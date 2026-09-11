//! live_bench: Rust port of `run_live_benchmark.py` §1 (PyTorch e2e).
//!
//! Same measurement protocol, applied to the Candle model:
//! - model load + voice conditioning happen OUTSIDE the timer
//! - warmup: 1 run with seed 999 (never overlaps measured seeds)
//! - measured: N runs (default 3) with seeds 0..N
//! - gen time = e2e synthesis (text prefill + AR loop + sampler + Mimi)
//! - JSON schema matches the Python `add_result` keys
//!
//! Build & run (web-ui build script is skipped):
//! ```bash
//! cargo run --release -p pocket-tts-cli --no-default-features --bin live_bench -- \
//!   --variant korean --voice 2_0000.wav --outdir live_bench_wav
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use pocket_tts::TTSModel;
use pocket_tts::config::load_config;
use pocket_tts::weights::download_if_necessary;
use serde::Serialize;

// hgemm_ link stub for `pocket-tts/mkl` builds (see ../mkl_hgemm_stub.rs).
// Only needed when candle links static MKL; harmless otherwise.
#[path = "../mkl_hgemm_stub.rs"]
mod mkl_hgemm_stub;

/// Rust port of run_live_benchmark.py §1 (candle e2e, CPU/FP32).
#[derive(Parser, Debug)]
#[command(name = "live_bench", about = "Live e2e TTS benchmark (candle)")]
struct Args {
    /// Bundled model variant (e.g. korean). Ignored when --config is given.
    #[arg(long, default_value = "korean")]
    variant: String,

    /// Custom model config: local path or hf:// URL to a YAML file.
    #[arg(long)]
    config: Option<String>,

    /// Voice prompt wav for cloning.
    #[arg(long, default_value = "2_0000.wav")]
    voice: String,

    /// Text to synthesize.
    #[arg(long, default_value = "안녕하세요. 한국어 음성 합성 모델입니다.")]
    text: String,

    /// Output directory for wavs + JSON.
    #[arg(long, default_value = "live_bench_wav")]
    outdir: String,

    /// Measured runs (seeds 0..runs) after one warmup run (seed 999).
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// MKL/OMP thread count (sets RAYON/MKL/OMP_NUM_THREADS before load).
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

fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

fn std(xs: &[f64], m: f64) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some((xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt())
}

/// QA metrics for a mono f32 wav: peak, RMS, mean spectral centroid (Hz).
/// `qa_wav.qa` equivalent (python_backup/qa_wav.py is absent from the repo,
/// so this defines the Rust-side metric).
fn qa_wav(path: &std::path::Path, sample_rate: u32) -> Result<(f64, f64, f64)> {
    let mut reader = hound::WavReader::open(path)?;
    let samples: Vec<f32> = match reader.spec().sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|s| s as f32 / i32::MAX as f32)
            .collect(),
    };
    if samples.is_empty() {
        anyhow::bail!("empty wav: {}", path.display());
    }
    // De-interleave to mono (first channel) if needed.
    let ch = reader.spec().channels.max(1) as usize;
    let mono: Vec<f32> = if ch == 1 {
        samples
    } else {
        samples.into_iter().step_by(ch).collect()
    };

    let peak = mono.iter().fold(0.0f32, |m, &s| m.max(s.abs())) as f64;
    let rms = (mono.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / mono.len() as f64).sqrt();

    // Mean spectral centroid over Hann-windowed frames (n=4096, no overlap).
    const N: usize = 4096;
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut spectrum = fft.make_output_vec();
    let hann: Vec<f32> = (0..N)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (N - 1) as f32).cos()))
        .collect();
    let mut cent_sum = 0.0f64;
    let mut cent_n = 0usize;
    for frame in mono.chunks(N) {
        let mut buf = vec![0.0f32; N];
        for (i, &s) in frame.iter().enumerate() {
            buf[i] = s * hann[i];
        }
        fft.process(&mut buf, &mut spectrum)
            .map_err(|e| anyhow::anyhow!("fft: {:?}", e))?;
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
    let cent = if cent_n > 0 {
        cent_sum / cent_n as f64
    } else {
        0.0
    };
    Ok((peak, rms, cent))
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(t) = args.threads {
        // SAFETY: set before any compute threads are spawned.
        // RAYON_NUM_THREADS drives candle's gemm-crate matmuls (non-MKL
        // builds); MKL/OMP vars drive MKL builds. Set all three so
        // --threads is honored in every configuration.
        unsafe {
            std::env::set_var("RAYON_NUM_THREADS", t.to_string());
            std::env::set_var("OMP_NUM_THREADS", t.to_string());
            std::env::set_var("MKL_NUM_THREADS", t.to_string());
        }
    }
    let threads_label = args
        .threads
        .map(|t| format!("{t}T"))
        .unwrap_or_else(|| "auto".to_string());

    let outdir = std::path::PathBuf::from(&args.outdir);
    std::fs::create_dir_all(&outdir)?;

    // ---- Load (outside timer), same as Python §1 ----
    let t_load = std::time::Instant::now();
    let config = if let Some(cfg) = args.config.as_deref() {
        let p = download_if_necessary(cfg)
            .with_context(|| format!("Failed to fetch model config '{cfg}'"))?;
        load_config(&p)?
    } else {
        let p = TTSModel::config_path_for_variant(&args.variant)?;
        load_config(&p)?
    };
    let mut model = TTSModel::load_from_config(config, None, 1, -4.0, None, &candle_core::Device::Cpu)?;
    let voice_state = model
        .get_voice_state(&args.voice)
        .with_context(|| format!("Failed to encode voice '{}'", args.voice))?;
    let sr = model.sample_rate;
    eprintln!(
        "model loaded in {:.1}s (sample_rate={}Hz, voice ready)",
        t_load.elapsed().as_secs_f32(),
        sr
    );

    // ---- Warmup (seed 999, discarded) ----
    model.seed = Some(999);
    let _ = model.generate(&args.text, &voice_state)?;

    // ---- Measured runs (seeds 0..runs) ----
    let mut gens = Vec::new();
    let mut durs = Vec::new();
    let mut first_wav: Option<std::path::PathBuf> = None;
    for i in 0..args.runs {
        model.seed = Some(i as u64);
        let t0 = std::time::Instant::now();
        let audio = model.generate(&args.text, &voice_state)?;
        let dt = t0.elapsed().as_secs_f64();

        // audio: [C, T] (mono) — flatten channel-first.
        let (n_ch, n_samples) = match audio.dims() {
            [c, t] => (*c, *t),
            [t] => (1, *t),
            d => anyhow::bail!("unexpected audio dims: {:?}", d),
        };
        let flat = audio.flatten_all()?.to_vec1::<f32>()?;
        debug_assert_eq!(flat.len(), n_ch * n_samples);
        let mono: Vec<f32> = if n_ch == 1 {
            flat
        } else {
            flat.into_iter().step_by(n_ch).collect()
        };
        let dur = mono.len() as f64 / sr as f64;
        gens.push(dt);
        durs.push(dur);

        if i == 0 {
            let wav_path = outdir.join(format!("candle_cpu{threads_label}_FP32.wav"));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: sr as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut w = hound::WavWriter::create(&wav_path, spec)?;
            for s in &mono {
                w.write_sample(*s)?;
            }
            w.finalize()?;
            first_wav = Some(wav_path);
        }
        println!("run {i}: gen={dt:.3}s audio={dur:.3}s");
    }

    let gm = mean(&gens).unwrap();
    let gs = std(&gens, gm).unwrap();
    let gmin = gens.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let gmax = gens.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let dm = mean(&durs).unwrap();

    let (qa_peak, qa_rms, qa_cent) = match first_wav.as_deref() {
        Some(p) => {
            let (pk, rms, cent) = qa_wav(p, sr as u32)?;
            (Some(pk), Some(rms), Some(cent))
        }
        None => (None, None, None),
    };

    let row = BenchRow {
        model: "candle-rust".to_string(),
        backend: "CPU".to_string(),
        threads: threads_label.clone(),
        quant: "FP32".to_string(),
        scope: "e2e".to_string(),
        status: "OK".to_string(),
        gen_mean_s: Some(round3(gm)),
        gen_std_s: Some(round3(gs)),
        gen_min_s: Some(round3(gmin)),
        gen_max_s: Some(round3(gmax)),
        audio_mean_s: Some(round3(dm)),
        speed_x: Some(round2(dm / gm)),
        rtf: Some(round4(gm / dm)),
        qa_peak: qa_peak.map(round3),
        qa_rms: qa_rms.map(round4),
        qa_cent: qa_cent.map(|c| c.round()),
        note: String::new(),
    };
    let speed_str = format!("{:.2}x (RTF {:.3})", dm / gm, gm / dm);
    println!(
        "[OK] {:<14} | {:<6} | {:<6} | {:<12} | {:<8} -> gen: {}s, speed: {} | ",
        row.model, row.backend, row.threads, row.quant, row.scope, row.gen_mean_s.unwrap(),
        speed_str,
    );

    let out_json = outdir.join("live_benchmark_result_rs.json");
    std::fs::write(&out_json, serde_json::to_string_pretty(&row)?)?;
    println!("결과 JSON 저장: {}", out_json.display());
    Ok(())
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}
fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}
