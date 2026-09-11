use crate::ModelState;
use crate::models::transformer::StreamingTransformer;
use crate::modules::mlp::{LayerNorm, ModulationParams, SimpleMLPAdaLN};
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub fn lsd_decode(
    flow_net: &SimpleMLPAdaLN,
    modulations: &[Vec<ModulationParams>],
    x_0: &Tensor,
) -> Result<Tensor> {
    let mut current = x_0.clone();
    let num_steps = modulations.len();

    let step_factor = 1.0 / num_steps as f64;
    for step_mod in modulations {
        // Use forward_step_cached with pre-computed modulation batch for this ODE step
        let flow_dir = flow_net.forward_step_cached(&current, step_mod)?;
        current = (current + flow_dir.affine(step_factor, 0.0)?)?;
    }
    Ok(current)
}

#[derive(Clone)]
pub struct FlowLMModel {
    pub flow_net: SimpleMLPAdaLN,
    pub transformer: StreamingTransformer,
    pub input_linear: Linear,
    pub out_norm: LayerNorm,
    pub out_eos: Linear,
    pub bos_emb: Tensor,
    pub emb_mean: Tensor,
    pub emb_std: Tensor,
    pub ldim: usize,
    pub dim: usize,
    pub noise_clamp: Option<f32>,
    /// Learned `[1, 1, dim]` frame prepended to the voice-conditioning
    /// sequence when `insert_bos_before_voice` is set. Only present in v2
    /// (teacher) weights such as the Korean model. Matches upstream
    /// `FlowLMModel.bos_before_voice`.
    pub bos_before_voice: Option<Tensor>,
    /// Whether to prepend `bos_before_voice` during voice prompting.
    /// Matches upstream `flow_lm.insert_bos_before_voice`.
    pub insert_bos_before_voice: bool,
}

fn sample_noise(
    device: &candle_core::Device,
    shape: (usize, usize),
    temp: f32,
    clamp: Option<f32>,
    seed: Option<u64>,
) -> Result<Tensor> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let std = temp.sqrt();
    // Seeded RNG makes generation reproducible for benchmarks
    // (mirrors run_live_benchmark.py's torch.manual_seed(i) protocol).
    // Unseeded keeps the previous behavior via OS entropy.
    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let dist =
        rand_distr::Normal::new(0.0f32, std).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    match clamp {
        None => {
            let data: Vec<f32> = (0..shape.0 * shape.1)
                .map(|_| rand_distr::Distribution::sample(&dist, &mut rng))
                .collect();
            Tensor::from_vec(data, shape, device)
        }
        Some(limit) => {
            // Rejection sampling for truncated normal
            let count = shape.0 * shape.1;
            let mut data = Vec::with_capacity(count);

            while data.len() < count {
                let v = rand_distr::Distribution::sample(&dist, &mut rng);
                if v.abs() <= limit {
                    data.push(v);
                }
            }
            Tensor::from_vec(data, shape, device)
        }
    }
}

impl FlowLMModel {
    pub fn new(
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        ldim: usize,
        dim: usize,
        insert_bos_before_voice: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_linear = candle_nn::linear_no_bias(ldim, dim, vb.pp("input_linear"))?;
        let out_norm = LayerNorm::new(dim, 1e-5, true, vb.pp("out_norm"))?;
        let out_eos = candle_nn::linear(dim, 1, vb.pp("out_eos"))?;
        let bos_emb = vb.get(ldim, "bos_emb")?;
        let emb_mean = vb.get(ldim, "emb_mean")?;
        let emb_std = vb.get(ldim, "emb_std")?;
        // Only v2 (teacher) bundles carry this tensor; legacy student
        // weights do not. Require it when the config asks for it.
        let bos_before_voice = vb.get((1, 1, dim), "bos_before_voice").ok();
        if insert_bos_before_voice && bos_before_voice.is_none() {
            return Err(candle_core::Error::Msg(
                "config sets flow_lm.insert_bos_before_voice=true but the weights \
                 contain no flow_lm.bos_before_voice tensor"
                    .to_string(),
            ));
        }

        Ok(Self {
            flow_net,
            transformer,
            input_linear,
            out_norm,
            out_eos,
            bos_emb,
            emb_mean,
            emb_std,
            ldim,
            dim,
            noise_clamp: None, // Default to no clamp
            bos_before_voice,
            insert_bos_before_voice,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        sequence: &Tensor,
        text_embeddings: &Tensor,
        model_state: &mut ModelState,
        time_embeddings: &Tensor,
        temp: f32,
        eos_threshold: f32,
        step: usize,
        seed: Option<u64>,
    ) -> Result<(Tensor, bool)> {
        // sequence is [B, T, ldim]
        // text_embeddings is [B, S, dim]

        // Handle BOS (if NaN, use bos_emb) - simplistic check for NaN
        // In Candle we can use `Tensor::where_cond`
        // But for now let's assume sequence passed in doesn't have NaNs or handled upstream.
        // Original: sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)

        // Let's assume BOS is handled by caller for now or if sequence empty.

        let x = self.input_linear.forward(sequence)?;
        let s_len = text_embeddings.dims()[1];

        // Cat text embeddings and sequence embeddings only if text_embeddings is not empty
        let transformer_out_pre_norm = if s_len > 0 {
            let input = Tensor::cat(&[text_embeddings, &x], 1)?;
            let mut out = self.transformer.forward(&input, model_state, step)?;
            // Remove prefix (text embeddings length)
            out = out.narrow(1, s_len, out.dims()[1] - s_len)?;
            out
        } else {
            self.transformer.forward(&x, model_state, step)?
        };

        let transformer_out = self.out_norm.forward(&transformer_out_pre_norm)?;

        // Only use the last frame for generation
        let last_frame = transformer_out
            .narrow(1, transformer_out.dims()[1] - 1, 1)?
            .squeeze(1)?;

        let eos_score = self
            .out_eos
            .forward(&last_frame)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar::<f32>()?;
        let is_eos = eos_score > eos_threshold;

        // Generate noise with optional clamping.
        // A fixed seed makes the draw reproducible (benchmark protocol);
        // it is mixed with the AR step so every frame draws differently.
        let frame_seed = seed.map(|s| {
            s.wrapping_add(step as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
        });
        let noise = sample_noise(
            last_frame.device(),
            (last_frame.dims()[0], self.ldim),
            temp,
            self.noise_clamp,
            frame_seed,
        )?;

        // Pre-compute all modulations for this frame's ODE steps (8 steps * N blocks) in batch
        let c_emb = self.flow_net.embed_condition(&last_frame)?;
        let modulations = self
            .flow_net
            .precompute_modulations(&c_emb, time_embeddings)?;

        let next_latent = lsd_decode(&self.flow_net, &modulations, &noise)?;

        Ok((next_latent, is_eos))
    }
}
