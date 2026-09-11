//! onnx-bench: Rust ONNX benchmark — ORT step transformer + candle sampler/mimi.
//!
//!mirrors `bench_orig_onnx.py` §3 (hybrid) with the torch parts replaced by
//! the candle port: voice state comes from a precomputed torch dump
//! (`assets/vs.npz`, same pattern as deploy voice_cache.npz), text
//! embeddings from the candle conditioner, sampler (flow LSD) and Mimi
//! decode from candle. Step transformer runs in ONNX Runtime (`ort` crate).
//!
//! ```bash
//! export ORT_LIB_PATH=$HOME/test-assets/onnx_rs/ort_libs   # build only
//! export LD_LIBRARY_PATH=$HOME/miniconda3/lib/python3.14/site-packages/onnxruntime/capi
//! cargo run --release -p onnx-rs -- --onnx test-assets/pocket_step_uni256.onnx
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use clap::Parser;
use ndarray::Array;
use ort::value::Tensor as OrtTensor;
use pocket_tts::TTSModel;
use pocket_tts::config::load_config;
use pocket_tts::voice_state::init_states;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;
use std::fs::File;

/// Rust ONNX hybrid benchmark (ORT step + candle sampler/mimi).
#[derive(Parser, Debug)]
#[command(name = "onnx-bench", about = "ONNX Runtime step benchmark (Rust)")]
struct Args {
    /// ONNX step model path.
    #[arg(long, default_value = "test-assets/pocket_step_uni256.onnx")]
    onnx: String,
    /// Quant label for the report (e.g. FP32, INT8-dyn).
    #[arg(long, default_value = "FP32")]
    quant: String,
    /// Bundled candle variant for sampler/mimi/tokenizer.
    #[arg(long, default_value = "korean")]
    variant: String,
    /// Torch-dumped voice state (offs + caches).
    #[arg(long, default_value = "test-assets/onnx_rs/assets/vs.npz")]
    voice_state: String,
    #[arg(long, default_value = "안녕하세요. 한국어 음성 합성 모델입니다.")]
    text: String,
    #[arg(long, default_value = "/tmp/bench_onnx_rs")]
    outdir: String,
    #[arg(long, default_value_t = 3)]
    runs: usize,
    /// ORT intra-op threads (default: all cores).
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Serialize)]
struct BenchRow {
    model: String,
    backend: String,
    threads: String,
    quant: String,
    scope: String,
    status: String,
    gen_mean_s: Option<f64>,
    gen_std_s: Option<f64>,
    gen_min_s: Option<f64>,
    gen_max_s: Option<f64>,
    audio_mean_s: Option<f64>,
    speed_x: Option<f64>,
    rtf: Option<f64>,
    qa_peak: Option<f64>,
    qa_rms: Option<f64>,
    qa_cent: Option<f64>,
    note: String,
}

// --- minimal .npy/.npz reader (f32/i64 only) ---
fn parse_npy(bytes: &[u8]) -> Result<(Vec<f32>, Vec<i64>, Vec<usize>)> {
    anyhow::ensure!(&bytes[0..6] == b"\x93NUMPY", "bad npy magic");
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen])?;
    let descr = header.split("'descr':").nth(1).and_then(|s| s.split(',').next())
        .map(|s| s.trim().trim_matches('\'').to_string()).context("descr")?;
    let shape_src = header.split("'shape':").nth(1).and_then(|s| s.split(')').next()).context("shape")?;
    let shape: Vec<usize> = shape_src.split(['(', ',', ' ', ')']).filter(|t| !t.is_empty())
        .map(|t| t.parse::<usize>()).collect::<Result<_, _>>()?;
    let data = &bytes[10 + hlen..];
    let n: usize = shape.iter().product::<usize>().max(1);
    match descr.as_str() {
        "<f4" | "=f4" | "|f4" => Ok((
            (0..n).map(|i| f32::from_le_bytes(data[4 * i..4 * i + 4].try_into().unwrap())).collect(),
            vec![], shape,
        )),
        "<i8" | "=i8" | "|i8" => Ok((
            vec![],
            (0..n).map(|i| i64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap())).collect(),
            shape,
        )),
        d => anyhow::bail!("unsupported dtype {d}"),
    }
}

fn npz_f32(archive: &mut zip::ZipArchive<File>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let mut f = archive.by_name(&format!("{name}.npy"))?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut buf)?;
    let (v, _, s) = parse_npy(&buf)?;
    Ok((v, s))
}

fn npz_i64(archive: &mut zip::ZipArchive<File>, name: &str) -> Result<Vec<i64>> {
    let mut f = archive.by_name(&format!("{name}.npy"))?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut buf)?;
    let (_, v, _) = parse_npy(&buf)?;
    Ok(v)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// ort errors are neither Send nor Sync, so they cannot travel through
/// anyhow directly — format them into messages at the boundary.
fn oe<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("ort: {e:?}"))
}

/// Owned ORT tensor from an ndarray (error-normalized).
fn ot<T, D>(arr: ndarray::Array<T, D>) -> Result<OrtTensor<T>>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + 'static,
    D: ndarray::Dimension + 'static,
{
    oe(OrtTensor::from_array(arr))
}
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn qa_wav(path: &std::path::Path, sample_rate: u32) -> Result<(f64, f64, f64)> {
    let mut reader = hound::WavReader::open(path)?;
    let samples: Vec<f32> = match reader.spec().sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader.samples::<i32>().collect::<Result<Vec<_>, _>>()?
            .into_iter().map(|s| s as f32 / i32::MAX as f32).collect(),
    };
    anyhow::ensure!(!samples.is_empty(), "empty wav");
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs())) as f64;
    let rms = (samples.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
    // Mean spectral centroid over Hann-windowed frames (same as live_bench).
    const N: usize = 4096;
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut spectrum = fft.make_output_vec();
    let hann: Vec<f32> = (0..N)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (N - 1) as f32).cos()))
        .collect();
    let mut cent_sum = 0.0f64;
    let mut cent_n = 0usize;
    for frame in samples.chunks(N) {
        let mut buf = vec![0.0f32; N];
        for (i, &s) in frame.iter().enumerate() {
            buf[i] = s * hann[i];
        }
        fft.process(&mut buf, &mut spectrum)
            .map_err(|e| anyhow::anyhow!("fft: {e:?}"))?;
        let mut mag_sum = 0.0f64;
        let mut w_sum = 0.0f64;
        for (k, c) in spectrum.iter().enumerate() {
            let mag = c.norm() as f64;
            mag_sum += mag;
            w_sum += k as f64 * mag;
        }
        if mag_sum > 1e-9 {
            cent_sum += w_sum / mag_sum * sample_rate as f64 / N as f64;
            cent_n += 1;
        }
    }
    let cent = if cent_n > 0 { cent_sum / cent_n as f64 } else { 0.0 };
    Ok((peak, rms, cent))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ort::init().commit();
    let outdir = std::path::PathBuf::from(&args.outdir);
    std::fs::create_dir_all(&outdir)?;
    let threads_label = args.threads.map(|t| format!("{t}T")).unwrap_or_else(|| "auto".to_string());

    // ---- candle model (sampler/mimi/tokenizer) ----
    let cfg_path = TTSModel::config_path_for_variant(&args.variant)?;
    let config = load_config(&cfg_path)?;
    let model = TTSModel::load_from_config(config, None, 1, -4.0, None, &Device::Cpu)?;
    let sr = model.sample_rate;
    let dim = model.dim;
    let ldim = model.ldim;

    // ---- torch voice state dump ----
    let f = File::open(&args.voice_state)?;
    let mut archive = zip::ZipArchive::new(f)?;
    let n_layers = 24;
    let mut base_offs = Vec::with_capacity(n_layers);
    let mut base_caches = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        base_offs.push(npz_i64(&mut archive, &format!("off{i}"))?[0]);
        let (c, _shape) = npz_f32(&mut archive, &format!("cache{i}"))?;
        // Pad compact torch [2,1,L,16,64] caches to full 256 context with
        // NaN, exactly like the Python hybrid does before feeding ONNX.
        let l = c.len() / (2 * 16 * 64);
        let mut full = vec![f32::NAN; 2 * 256 * 16 * 64];
        for k in 0..2 {
            for h in 0..16 {
                for dd in 0..64 {
                    for t in 0..l {
                        full[((k * 256 + t) * 16 + h) * 64 + dd] =
                            c[((k * l + t) * 16 + h) * 64 + dd];
                    }
                }
            }
        }
        base_caches.push(full);
    }

    // ---- text ----
    let prepared = model.conditioner.prepare(&args.text, &Device::Cpu)?;
    let tc = prepared.dims()[1];
    let text_emb_t = model.conditioner.forward(&prepared)?;
    let text_emb = text_emb_t.squeeze(0)?.to_vec2::<f32>()?;
    let mg = ((tc as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;

    // ---- ORT session ----
    let mut builder = oe(ort::session::Session::builder())?;
    if let Some(t) = args.threads {
        builder = oe(builder.with_intra_threads(t))?;
        builder = oe(builder.with_inter_threads(1))?;
    }
    let mut session = oe(builder.commit_from_file(&args.onnx))?;
    let time_emb = model.flow_lm.flow_net.compute_time_embeddings(
        model.lsd_decode_steps, &model.device, DType::F32,
    )?;

    // ---- run closure ----
    let run_once = |seed: u64,
                    session: &mut ort::session::Session,
                    model: &TTSModel,
                    time_emb: &Tensor|
     -> Result<(f64, f64, Vec<f32>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise_dist = rand_distr::Normal::new(0.0f32, model.temp.sqrt())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut offs = base_offs.clone();
        let mut caches = base_caches.clone();
        let t0 = std::time::Instant::now();

        // prefill
        for t in 0..tc {
            let tok = Array::from_shape_vec((1, 1, ldim), vec![f32::NAN; ldim])?;
            let emb: Vec<f32> = text_emb[t].clone();
            let emb_arr = Array::from_shape_vec((1, 1, dim), emb)?;
            let flag = Array::from_shape_vec((1,), vec![true])?;
            let mut inputs: Vec<(String, ort::session::SessionInputValue)> = vec![
                ("token".to_string(), ot(tok)?.into()),
                ("emb".to_string(), ot(emb_arr)?.into()),
                ("is_text".to_string(), ot(flag)?.into()),
            ];
            for i in 0..n_layers {
                let c = Array::from_shape_vec((2, 1, 256, 16, 64), caches[i].clone())?;
                inputs.push((format!("off{i}"), ot(Array::from_shape_vec((1,), vec![offs[i]])?)?.into()));
                inputs.push((format!("cache{i}"), ot(c)?.into()));
            }
            let outs = oe(session.run(inputs))?;
            for i in 0..n_layers {
                let (_, v) = oe(outs[format!("new_cache{i}").as_str()].try_extract_tensor::<f32>())?;
                caches[i] = v.to_vec();
                offs[i] += 1;
            }
        }

        // AR loop (ORT step + candle sampler)
        let zero_emb = vec![0.0f32; dim];
        let mut token = vec![f32::NAN; ldim];
        let mut latents: Vec<Vec<f32>> = Vec::new();
        let mut eos_step: Option<usize> = None;
        // Python hybrid uses a fixed +3 (not the text-based guess).
        let frames_after = 3;
        for step_i in 0..mg {
            let tok = Array::from_shape_vec((1, 1, ldim), token.clone())?;
            let emb_arr = Array::from_shape_vec((1, 1, dim), zero_emb.clone())?;
            let flag = Array::from_shape_vec((1,), vec![false])?;
            let mut inputs: Vec<(String, ort::session::SessionInputValue)> = vec![
                ("token".to_string(), ot(tok)?.into()),
                ("emb".to_string(), ot(emb_arr)?.into()),
                ("is_text".to_string(), ot(flag)?.into()),
            ];
            for i in 0..n_layers {
                let c = Array::from_shape_vec((2, 1, 256, 16, 64), caches[i].clone())?;
                inputs.push((format!("off{i}"), ot(Array::from_shape_vec((1,), vec![offs[i]])?)?.into()));
                inputs.push((format!("cache{i}"), ot(c)?.into()));
            }
            let outs = oe(session.run(inputs))?;
            let (_, tout) = oe(outs["out"].try_extract_tensor::<f32>())?;
            let tout: Vec<f32> = tout.to_vec();
            if step_i == 0 && std::env::var("ONNX_DEBUG").is_ok() {
                let bytes: Vec<u8> = tout.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write("/tmp/rs_onnx_tout0.bin", &bytes).ok();
            }
            for i in 0..n_layers {
                let (_, v) = oe(outs[format!("new_cache{i}").as_str()].try_extract_tensor::<f32>())?;
                caches[i] = v.to_vec();
                offs[i] += 1;
            }
            // candle sampler
            let tout_t = Tensor::from_vec(tout, (1, 1, dim), &model.device)?;
            let last = tout_t.narrow(1, 0, 1)?.squeeze(1)?;
            let eos: f32 = model.flow_lm.out_eos.forward(&last)?.squeeze(0)?.squeeze(0)?.to_scalar()?;
            let c_emb = model.flow_lm.flow_net.embed_condition(&last)?;
            let mods = model.flow_lm.flow_net.precompute_modulations(&c_emb, time_emb)?;
            let noise: Vec<f32> = (0..ldim).map(|_| rand_distr::Distribution::sample(&noise_dist, &mut rng)).collect();
            let noise_t = Tensor::from_vec(noise, (1, ldim), &model.device)?;
            let lat_t = pocket_tts::models::flow_lm::lsd_decode(&model.flow_lm.flow_net, &mods, &noise_t)?;
            let lat: Vec<f32> = lat_t.squeeze(0)?.to_vec1::<f32>()?;
            if eos > model.eos_threshold && eos_step.is_none() {
                eos_step = Some(step_i);
            }
            if let Some(e) = eos_step {
                if step_i >= e + frames_after {
                    break;
                }
            }
            token = lat.clone();
            latents.push(lat);
        }

        // candle mimi decode
        let mut mimi_state = init_states(1, 1000);
        let mut chunks: Vec<f32> = Vec::new();
        for (fi, lat) in latents.iter().enumerate() {
            let lat_t = Tensor::from_vec(lat.clone(), (1, ldim), &model.device)?;
            let denorm = lat_t.broadcast_mul(&model.flow_lm.emb_std)?.broadcast_add(&model.flow_lm.emb_mean)?;
            let mimi_in = denorm.unsqueeze(1)?.transpose(1, 2)?;
            let q = model.mimi.quantize(&mimi_in)?;
            let fr = model.mimi.decode_from_latent(&q, &mut mimi_state, fi)?;
            let v: Vec<f32> = fr.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
            chunks.extend(v);
        }
        let dt = t0.elapsed().as_secs_f64();
        Ok((dt, chunks.len() as f64 / sr as f64, chunks))
    };

    // warmup + measured
    run_once(999, &mut session, &model, &time_emb)?;
    let mut gens = vec![];
    let mut durs = vec![];
    let mut first_wav = None;
    for i in 0..args.runs {
        let (dt, dur, audio) = run_once(i as u64, &mut session, &model, &time_emb)?;
        gens.push(dt);
        durs.push(dur);
        println!("run {i}: gen={dt:.3}s audio={dur:.3}s");
        if i == 0 {
            let p = outdir.join(format!("onnxrs_cpu{}T_{}.wav", threads_label, args.quant));
            let spec = hound::WavSpec { channels: 1, sample_rate: sr as u32, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
            let mut w = hound::WavWriter::create(&p, spec)?;
            for s in &audio {
                w.write_sample(*s)?;
            }
            w.finalize()?;
            first_wav = Some(p);
        }
    }
    let gm = mean(&gens);
    let dm = mean(&durs);
    let (pk, rms, cent) = match first_wav.as_deref() {
        Some(p) => qa_wav(p, sr as u32)?,
        None => (0.0, 0.0, 0.0),
    };
    let row = BenchRow {
        model: "onnx-rs".to_string(), backend: "CPU".to_string(), threads: threads_label.clone(),
        quant: args.quant.clone(), scope: "e2e".to_string(), status: "OK".to_string(),
        gen_mean_s: Some(round3(gm)), gen_std_s: None,
        gen_min_s: Some(round3(gens.iter().fold(f64::INFINITY, |a, &b| a.min(b)))),
        gen_max_s: Some(round3(gens.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))),
        audio_mean_s: Some(round3(dm)), speed_x: Some((dm / gm * 100.0).round() / 100.0),
        rtf: Some((gm / dm * 10000.0).round() / 10000.0),
        qa_peak: Some(round3(pk)), qa_rms: Some((rms * 10000.0).round() / 10000.0),
        qa_cent: Some(cent.round()), note: String::new(),
    };
    println!("[OK] onnx-rs | CPU | {} -> gen: {}s, {:.2}x", threads_label, row.gen_mean_s.unwrap(), dm / gm);
    std::fs::write(outdir.join("bench_onnx_rs.json"), serde_json::to_string_pretty(&row)?)?;
    Ok(())
}
