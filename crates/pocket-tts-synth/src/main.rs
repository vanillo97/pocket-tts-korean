//! deploy-rs: Rust port of `deploy_package_fp32_fp32_256/synth.py`.
//!
//! Same behavior, same CLI (`--pkg/--text/--out/--seed`), same assets
//! (`models/*.xml`, `assets/params.json`, `voice_cache.npz`, `cond_embed.npy`,
//! `tokenizer.model`). Inference runs through the `openvino` crate
//! (runtime-linking: no link-time OpenVINO dependency).
//!
//! ```bash
//! cd test-assets/deploy_package_fp32_fp32_256_rs
//! cargo run --release -- --text "안녕하세요." --out out.wav
//! cargo run --release -- --pkg ../deploy_package_fp32_fp32_256 \
//!   --text "첫 문장입니다. 두 번째 문장입니다." --out out.wav --seed 1
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use openvino::{CompiledModel, ElementType, InferRequest, Shape, Tensor};
use pocket_tts::conditioners::text::LUTConditioner;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

/// Rust port of deploy_package/synth.py (OpenVINO, torch-free).
#[derive(Parser, Debug)]
#[command(name = "synth-rs", about = "pocket-tts OV synthesis (Rust)")]
struct Args {
    /// deploy_package path (models/ + assets/). Defaults to the sibling
    /// `../deploy_package_fp32_fp32_256` next to this crate.
    #[arg(long)]
    pkg: Option<String>,
    #[arg(long, default_value = "안녕하세요. 한국어 음성 합성 모델입니다.")]
    text: String,
    #[arg(long, default_value = "out.wav")]
    out: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// OpenVINO device: CPU, GPU, NPU, GPU.0, GPU.1, ...
    #[arg(long, default_value = "CPU")]
    device: String,
    /// Device for the mimi decoder (defaults to --device).
    /// The Python benchmark keeps mimi on CPU even for GPU runs.
    #[arg(long)]
    mimi_device: Option<String>,
    /// Inference threads (INFERENCE_NUM_THREADS). Default: device default.
    #[arg(long)]
    threads: Option<usize>,
    /// 구 패키지(fp32_int8_256, int4_int8_256) 호환: 문장 분할 없이
    /// 텍스트 전체를 한 번에 합성합니다.
    #[arg(long, default_value_t = false)]
    single: bool,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct Params {
    temp: f32,
    eos_threshold: f32,
    sample_rate: u32,
    ldim: usize,
    dim: usize,
    emb_std: Vec<f32>,
    emb_mean: Vec<f32>,
    max_len: usize,
    tokens_per_sec: f64,
    gen_pad_sec: f64,
    frame_rate: f64,
    pad_spaces: bool,
    rm_semi: bool,
    append_punct: bool,
    n_layers: usize,
    #[serde(default = "d256")]
    max_latents: usize,
}

fn d256() -> usize {
    256
}

// ---------------------------------------------------------------------------
// .npy / .npz (minimal reader: C-order, f4/i8/i4/u1)
// ---------------------------------------------------------------------------

enum NpyArray {
    F32(Vec<f32>, Vec<usize>),
    I64(Vec<i64>, Vec<usize>),
}

/// IEEE-754 binary16 -> f32 (for f16 voice caches in 1024-ctx packages).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let f = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = 0u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            (((127 - 15 - e) << 23) | ((m & 0x3FF) << 13)) | (sign << 31)
        }
    } else if exp == 31 {
        ((0xFF << 23) | (mant << 13)) | (sign << 31)
    } else {
        (((exp + 112) << 23) | (mant << 13)) | (sign << 31)
    };
    f32::from_bits(f)
}

fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    anyhow::ensure!(bytes.len() > 10, "npy too short");
    anyhow::ensure!(&bytes[0..6] == b"\x93NUMPY", "bad npy magic");
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen]).context("npy header utf8")?;
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().trim_matches('\'').to_string())
        .context("npy descr")?;
    let shape_src = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .context("npy shape")?;
    let shape: Vec<usize> = shape_src
        .split(['(', ',', ' ', ')'])
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<usize>().context("npy dim"))
        .collect::<Result<_>>()?;
    if header.contains("'fortran_order': True") && shape.len() > 1 {
        anyhow::bail!("fortran-order npy not supported");
    }
    let data = &bytes[10 + hlen..];
    let n: usize = shape.iter().product::<usize>().max(1);
    match descr.as_str() {
        "<f4" | "=f4" | "|f4" => {
            anyhow::ensure!(data.len() >= n * 4, "npy f32 truncated");
            let v: Vec<f32> = (0..n)
                .map(|i| f32::from_le_bytes(data[4 * i..4 * i + 4].try_into().unwrap()))
                .collect();
            Ok(NpyArray::F32(v, shape))
        }
        "<f2" | "=f2" | "|f2" => {
            anyhow::ensure!(data.len() >= n * 2, "npy f16 truncated");
            let v: Vec<f32> = (0..n)
                .map(|i| {
                    f16_to_f32(u16::from_le_bytes(
                        data[2 * i..2 * i + 2].try_into().unwrap(),
                    ))
                })
                .collect();
            Ok(NpyArray::F32(v, shape))
        }
        "<i8" | "=i8" | "|i8" => {
            anyhow::ensure!(data.len() >= n * 8, "npy i64 truncated");
            let v: Vec<i64> = (0..n)
                .map(|i| i64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap()))
                .collect();
            Ok(NpyArray::I64(v, shape))
        }
        "<i4" | "=i4" => {
            anyhow::ensure!(data.len() >= n * 4, "npy i32 truncated");
            let v: Vec<i64> = (0..n)
                .map(|i| i32::from_le_bytes(data[4 * i..4 * i + 4].try_into().unwrap()) as i64)
                .collect();
            Ok(NpyArray::I64(v, shape))
        }
        d => anyhow::bail!("unsupported npy dtype: {d}"),
    }
}

fn read_npz_f32(archive: &mut zip::ZipArchive<File>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    // .npz entries are stored with a `.npy` suffix inside the zip.
    let zip_name = format!("{name}.npy");
    let mut f = archive
        .by_name(&zip_name)
        .with_context(|| format!("npz missing: {zip_name}"))?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut buf)?;
    match parse_npy(&buf)? {
        NpyArray::F32(v, s) => Ok((v, s)),
        _ => anyhow::bail!("{name} is not f32"),
    }
}

fn read_npz_i64(archive: &mut zip::ZipArchive<File>, name: &str) -> Result<(Vec<i64>, Vec<usize>)> {
    let zip_name = format!("{name}.npy");
    let mut f = archive
        .by_name(&zip_name)
        .with_context(|| format!("npz missing: {zip_name}"))?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut buf)?;
    match parse_npy(&buf)? {
        NpyArray::I64(v, s) => Ok((v, s)),
        _ => anyhow::bail!("{name} is not i64"),
    }
}

// ---------------------------------------------------------------------------
// OpenVINO tensor helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_step(
    req: &mut InferRequest,
    t_token: &mut Reusable,
    t_emb: &mut Reusable,
    t_flag: &mut Reusable,
    t_offs: &mut [Reusable],
    t_caches: &mut [Reusable],
    off_names: &[String],
    cache_names: &[String],
    new_cache_names: &[String],
    tok: &[f32],
    emb: &[f32],
    is_text: bool,
    offs: &[i64],
    caches: &[Vec<f32>],
    cache_shapes: &[Vec<i64>],
    p: &Params,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    t_token.set_f32(&[1, 1, p.ldim as i64], tok)?;
    t_emb.set_f32(&[1, 1, p.dim as i64], emb)?;
    t_flag.set_bool(&[1], is_text)?;
    req.set_tensor("token", t_token.as_tensor())?;
    req.set_tensor("emb", t_emb.as_tensor())?;
    req.set_tensor("is_text", t_flag.as_tensor())?;
    for i in 0..p.n_layers {
        t_offs[i].set_i64(&[1], &[offs[i]])?;
        t_caches[i].set_f32(&cache_shapes[i], &caches[i])?;
        req.set_tensor(&off_names[i], t_offs[i].as_tensor())?;
        req.set_tensor(&cache_names[i], t_caches[i].as_tensor())?;
    }
    req.infer()?;
    let out = read_f32(req, "out")?;
    let mut new_caches = Vec::with_capacity(p.n_layers);
    for name in new_cache_names {
        new_caches.push(read_f32(req, name)?);
    }
    Ok((out, new_caches))
}

fn read_f32(req: &InferRequest, name: &str) -> Result<Vec<f32>> {
    Ok(req
        .get_tensor(name)
        .with_context(|| format!("missing output: {name}"))?
        .get_data::<f32>()?
        .to_vec())
}

/// Reusable OpenVINO tensor buffer: allocated once, refilled in place.
/// Per-step `Tensor::new` on 24 layers x caches (~50MB for 256ctx,
/// ~200MB for 1024ctx) dominated runtime; reuse removes allocator +
/// plugin-side reallocation overhead. Shape changes (mimi streaming
/// states only) fall back to recreation.
struct Reusable {
    tensor: Tensor,
    etype: ElementType,
    shape: Vec<i64>,
}

impl Reusable {
    fn new(etype: ElementType, shape: &[i64]) -> Result<Self> {
        Ok(Self {
            tensor: Tensor::new(etype, &Shape::new(shape)?)?,
            etype,
            shape: shape.to_vec(),
        })
    }

    fn ensure(&mut self, etype: ElementType, shape: &[i64]) -> Result<()> {
        if self.etype as u8 == etype as u8 && self.shape == shape {
            return Ok(());
        }
        self.tensor = Tensor::new(etype, &Shape::new(shape)?)?;
        self.etype = etype;
        self.shape = shape.to_vec();
        Ok(())
    }

    fn set_f32(&mut self, shape: &[i64], data: &[f32]) -> Result<()> {
        self.ensure(ElementType::F32, shape)?;
        self.tensor.get_data_mut::<f32>()?.copy_from_slice(data);
        Ok(())
    }

    fn set_i64(&mut self, shape: &[i64], data: &[i64]) -> Result<()> {
        self.ensure(ElementType::I64, shape)?;
        self.tensor.get_data_mut::<i64>()?.copy_from_slice(data);
        Ok(())
    }

    fn set_bool(&mut self, shape: &[i64], v: bool) -> Result<()> {
        self.ensure(ElementType::Boolean, shape)?;
        self.tensor.get_data_mut::<u8>()?.copy_from_slice(&[v as u8]);
        Ok(())
    }

    fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }
}

// ---------------------------------------------------------------------------
// Text (mirrors synth.py prepare_text_prompt / split_sentences / pack_chunks)
// ---------------------------------------------------------------------------

fn prepare_text_prompt(text: &str, pad_spaces: bool, rm_semi: bool, append_punct: bool) -> Result<(String, usize)> {
    let mut text = text.trim().to_string();
    anyhow::ensure!(!text.is_empty(), "empty");
    text = text.replace('\n', " ").replace('\r', " ").replace("  ", " ");
    if rm_semi {
        text = text.replace(';', ",");
    }
    let nw = text.split_whitespace().count();
    let frames_after = if nw <= 4 { 3 } else { 1 };
    if let Some(first) = text.chars().next() {
        if !first.is_uppercase() {
            text = format!("{}{}", first.to_uppercase(), &text[first.len_utf8()..]);
        }
    }
    if append_punct {
        if let Some(last) = text.chars().last() {
            if last.is_alphanumeric() {
                text.push('.');
            }
        }
    }
    if pad_spaces && text.split_whitespace().count() < 5 {
        text = format!("{}{}", " ".repeat(8), text);
    }
    Ok((text, frames_after))
}

/// Split on `(?<=[.?!…])\s+` or `\n+` (regex look-behind-free port).
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.trim().chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if ".?!…".contains(c) {
            let mut split = false;
            while let Some(&n) = chars.peek() {
                if n == '\n' || n.is_whitespace() {
                    split = true;
                    chars.next();
                } else {
                    break;
                }
            }
            if split {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur = String::new();
            }
        } else if c == '\n' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().to_string());
            }
            cur = String::new();
            while chars.peek().map(|n| *n == '\n' || n.is_whitespace()).unwrap_or(false) {
                chars.next();
            }
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn estimate_gen(tc: usize, p: &Params) -> usize {
    ((tc as f64 / p.tokens_per_sec + p.gen_pad_sec) * p.frame_rate).ceil() as usize
}

fn pack_chunks(sent_ids: &[Vec<u32>], words: &[usize], p: &Params, off0: usize) -> Vec<Vec<usize>> {
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_tc = 0usize;
    for (si, ids) in sent_ids.iter().enumerate() {
        let tc = ids.len();
        let trial = cur_tc + tc + if cur.is_empty() { 0 } else { 1 };
        let _ = words;
        if !cur.is_empty()
            && (off0 + trial + estimate_gen(trial, p) > p.max_len
                || estimate_gen(trial, p) > p.max_latents)
        {
            chunks.push(cur);
            cur = vec![si];
            cur_tc = tc;
        } else {
            cur.push(si);
            cur_tc = trial;
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

// ---------------------------------------------------------------------------
// Synthesis
// ---------------------------------------------------------------------------

struct Models {
    step: CompiledModel,
    sampler: CompiledModel,
    mimi: CompiledModel,
}

struct MimiInput {
    name: String,
    kind: MimiStateKind,
    shape: Vec<i64>,
}

#[derive(Clone, Copy, PartialEq)]
enum MimiStateKind {
    Off,
    Cache,
    Other,
}

struct MimiSlot {
    name: String,
    kind: MimiStateKind,
    buf: Reusable,
}

fn fresh_mimi_slots(inputs: &[MimiInput]) -> Result<Vec<MimiSlot>> {
    // State shapes evolve: like synth.py (`st[nm] = output`), each step's
    // output REPLACES the stored state, shape included (handled by
    // Reusable::ensure on update).
    let mut out = Vec::with_capacity(inputs.len());
    for inp in inputs {
        let n: usize = inp.shape.iter().map(|d| *d as usize).product();
        let buf = match inp.kind {
            MimiStateKind::Off => {
                let mut b = Reusable::new(ElementType::I64, &inp.shape)?;
                b.set_i64(&inp.shape, &vec![0i64; n])?;
                b
            }
            MimiStateKind::Cache => {
                let mut b = Reusable::new(ElementType::F32, &inp.shape)?;
                b.set_f32(&inp.shape, &vec![f32::NAN; n])?;
                b
            }
            MimiStateKind::Other => {
                let mut b = Reusable::new(ElementType::F32, &inp.shape)?;
                b.set_f32(&inp.shape, &vec![0.0f32; n])?;
                b
            }
        };
        out.push(MimiSlot {
            name: inp.name.clone(),
            kind: inp.kind,
            buf,
        });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn synth_ids(
    models: &mut Models,
    mimi_inputs: &[MimiInput],
    all_ids: &[u32],
    guess: usize,
    text_emb_table: &[f32],
    p: &Params,
    base_offs: &[i64],
    base_caches: &[Vec<f32>],
    cache_shapes: &[Vec<i64>],
    rng: &mut StdRng,
) -> Result<(Vec<f32>, usize)> {
    let tc = all_ids.len();
    let max_gen = estimate_gen(tc, p);
    anyhow::ensure!(base_offs[0] as usize + tc + max_gen <= p.max_len, "MAX_LEN 초과");
    anyhow::ensure!(max_gen <= p.max_latents, "청크가 너무 김: 문장을 나누세요");

    let mut offs: Vec<i64> = base_offs.to_vec();
    let mut caches: Vec<Vec<f32>> = base_caches.iter().map(|c| c.clone()).collect();
    let text_emb: Vec<f32> = all_ids
        .iter()
        .flat_map(|&id| {
            let base = id as usize * p.dim;
            text_emb_table[base..base + p.dim].iter().copied()
        })
        .collect();

    // Persistent reusable buffers: allocated once per chunk, refilled in
    // place every step (see Reusable).
    let mut step_req = models.step.create_infer_request()?;
    let mut t_token = Reusable::new(ElementType::F32, &[1, 1, p.ldim as i64])?;
    let mut t_emb = Reusable::new(ElementType::F32, &[1, 1, p.dim as i64])?;
    let mut t_flag = Reusable::new(ElementType::Boolean, &[1])?;
    let mut t_offs: Vec<Reusable> = (0..p.n_layers)
        .map(|_| Reusable::new(ElementType::I64, &[1]))
        .collect::<Result<_>>()?;
    let mut t_caches: Vec<Reusable> = cache_shapes
        .iter()
        .map(|s| Reusable::new(ElementType::F32, s))
        .collect::<Result<_>>()?;
    let off_names: Vec<String> = (0..p.n_layers).map(|i| format!("off{i}")).collect();
    let cache_names: Vec<String> = (0..p.n_layers).map(|i| format!("cache{i}")).collect();
    let new_cache_names: Vec<String> =
        (0..p.n_layers).map(|i| format!("new_cache{i}")).collect();
    let nan_tok = vec![f32::NAN; p.ldim];
    let zero_emb = vec![0.0f32; p.dim];

    // Text prefill.
    for t in 0..tc {
        let emb_slice = &text_emb[t * p.dim..(t + 1) * p.dim];
        let (_, nc) = run_step(
            &mut step_req,
            &mut t_token,
            &mut t_emb,
            &mut t_flag,
            &mut t_offs,
            &mut t_caches,
            &off_names,
            &cache_names,
            &new_cache_names,
            &nan_tok,
            emb_slice,
            true,
            &offs,
            &caches,
            cache_shapes,
            p,
        )?;
        caches = nc;
        for o in offs.iter_mut() {
            *o += 1;
        }
    }

    // AR loop + sampler.
    let mut sampler_req = models.sampler.create_infer_request()?;
    let mut t_cond = Reusable::new(ElementType::F32, &[1, p.dim as i64])?;
    let mut t_noise = Reusable::new(ElementType::F32, &[1, p.ldim as i64])?;
    let mut token = vec![f32::NAN; p.ldim];
    let mut latents: Vec<Vec<f32>> = Vec::new();
    let mut eos_step: Option<usize> = None;
    let std = p.temp.sqrt();
    let frames_after = guess + 2;
    let noise_dist =
        rand_distr::Normal::new(0.0f32, std).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    for s in 0..max_gen {
        let (tout, nc) = run_step(
            &mut step_req,
            &mut t_token,
            &mut t_emb,
            &mut t_flag,
            &mut t_offs,
            &mut t_caches,
            &off_names,
            &cache_names,
            &new_cache_names,
            &token,
            &zero_emb,
            false,
            &offs,
            &caches,
            cache_shapes,
            p,
        )?;
        caches = nc;
        for o in offs.iter_mut() {
            *o += 1;
        }
        // cond = tout[:, -1:] reshaped to (1, dim); tout is [1,1,dim].
        anyhow::ensure!(tout.len() == p.dim, "unexpected step out len");
        let noise: Vec<f32> = (0..p.ldim)
            .map(|_| rand_distr::Distribution::sample(&noise_dist, &mut *rng))
            .collect();
        t_cond.set_f32(&[1, p.dim as i64], &tout)?;
        t_noise.set_f32(&[1, p.ldim as i64], &noise)?;
        sampler_req.set_tensor("cond", t_cond.as_tensor())?;
        sampler_req.set_tensor("noise", t_noise.as_tensor())?;
        sampler_req.infer()?;
        let lat = read_f32(&sampler_req, "latent")?;
        let eos_v = read_f32(&sampler_req, "eos")?;
        if s < 25 && std::env::var("SYNTH_DEBUG").is_ok() {
            let mx = tout.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            eprintln!("DBG s={s} tout_max={mx:.3} eos={:.3}", eos_v[0]);
            if s == 0 {
                let bytes: Vec<u8> = tout.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write("/tmp/rs_tout0.bin", &bytes).ok();
            }
        }
        if eos_v[0] > p.eos_threshold && eos_step.is_none() {
            eos_step = Some(s);
        }
        if let Some(e) = eos_step {
            if s >= e + frames_after {
                break;
            }
        }
        token = lat.clone();
        latents.push(lat);
    }

    // Mimi decode (fresh state per chunk, buffers reused across latents).
    let mut mimi_req = models.mimi.create_infer_request()?;
    let mut t_latent = Reusable::new(ElementType::F32, &[1, 1, p.ldim as i64])?;
    let mut slots = fresh_mimi_slots(mimi_inputs)?;
    let mut chunks: Vec<f32> = Vec::new();
    for lat in &latents {
        let denorm: Vec<f32> = lat
            .iter()
            .enumerate()
            .map(|(i, &v)| v * p.emb_std[i] + p.emb_mean[i])
            .collect();
        t_latent.set_f32(&[1, 1, p.ldim as i64], &denorm)?;
        mimi_req.set_tensor("latent", t_latent.as_tensor())?;
        for slot in &slots {
            mimi_req.set_tensor(&slot.name, slot.buf.as_tensor())?;
        }
        mimi_req.infer()?;
        chunks.extend(read_f32(&mimi_req, "audio")?);
        for (j, slot) in slots.iter_mut().enumerate() {
            let t = mimi_req.get_tensor(&format!("n{j}"))?;
            let shape: Vec<i64> = t.get_shape()?.get_dimensions().to_vec();
            match slot.kind {
                MimiStateKind::Off => {
                    slot.buf.set_i64(&shape, t.get_data::<i64>()?)?;
                }
                _ => {
                    slot.buf.set_f32(&shape, t.get_data::<f32>()?)?;
                }
            }
        }
    }
    Ok((chunks, latents.len()))
}

fn default_pkg() -> Result<PathBuf> {
    // The binary may live in the workspace target/ dir or the crate's own
    // target/ dir; probe likely layouts before giving up.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            // workspace layout: <ws>/target/<profile>/synth-rs
            candidates.push(d.join("../../test-assets/deploy_package_fp32_fp32_256"));
            // standalone layout: <crate>/target/<profile>/synth-rs
            candidates.push(d.join("../../../deploy_package_fp32_fp32_256"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("test-assets/deploy_package_fp32_fp32_256"));
        candidates.push(cwd.join("deploy_package_fp32_fp32_256"));
    }
    for c in &candidates {
        if c.join("models").is_dir() && c.join("assets").is_dir() {
            return Ok(c.clone());
        }
    }
    anyhow::bail!(
        "cannot locate deploy_package_fp32_fp32_256 (tried: {}); pass --pkg explicitly",
        candidates
            .iter()
            .map(|c| c.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn element_kind(name: &str) -> MimiStateKind {
    if name.contains("off") {
        MimiStateKind::Off
    } else if name.contains("cache") {
        MimiStateKind::Cache
    } else {
        MimiStateKind::Other
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let pkg = match args.pkg {
        Some(p) => PathBuf::from(p),
        None => default_pkg()?,
    };
    anyhow::ensure!(pkg.join("models").is_dir(), "--pkg not a deploy package: {}", pkg.display());
    let models_dir = pkg.join("models");
    let assets_dir = pkg.join("assets");

    let params: Params = serde_json::from_str(
        &std::fs::read_to_string(assets_dir.join("params.json"))?,
    )?;
    let mut rng = StdRng::seed_from_u64(args.seed);

    let t0 = Instant::now();
    let mut core = openvino::Core::new()?;
    let dev = |s: &str| -> openvino::DeviceType<'static> {
        match s {
            "CPU" => openvino::DeviceType::CPU,
            "GPU" => openvino::DeviceType::GPU,
            "NPU" => openvino::DeviceType::NPU,
            other => openvino::DeviceType::Other(other.to_owned().into()),
        }
    };
    let read = |core: &mut openvino::Core, name: &str| -> Result<openvino::Model> {
        let xml = models_dir.join(format!("{name}.xml"));
        let bin = models_dir.join(format!("{name}.bin"));
        Ok(core.read_model_from_file(
            xml.to_str().context("non-utf8 path")?,
            bin.to_str().context("non-utf8 path")?,
        )?)
    };
    let step_model = read(&mut core, "step")?;
    let sampler_model = read(&mut core, "sampler")?;
    let mimi_model = read(&mut core, "mimi")?;
    if let Some(t) = args.threads {
        // INFERENCE_NUM_THREADS is a compile-time device default in this API
        // (post-compile set_property is rejected), so set it on the Core first.
        core.set_property(
            &dev(args.device.as_str()),
            &openvino::RwPropertyKey::InferenceNumThreads,
            &t.to_string(),
        )?;
    }
    let mut step = core.compile_model(&step_model, dev(args.device.as_str()))?;
    let mut sampler = core.compile_model(&sampler_model, dev(args.device.as_str()))?;
    let mimi_dev = args.mimi_device.as_deref().unwrap_or(args.device.as_str());
    let mut mimi = core.compile_model(&mimi_model, dev(mimi_dev))?;
    let mut models = Models { step, sampler, mimi };
    println!("OV 로드: {:.1}s", t0.elapsed().as_secs_f32());

    // Voice state (precomputed KV caches).
    let npz = File::open(assets_dir.join("voice_cache.npz"))?;
    let mut archive = zip::ZipArchive::new(npz)?;
    let (off0, _) = read_npz_i64(&mut archive, "off0")?;
    let mut base_offs = Vec::with_capacity(params.n_layers);
    let mut base_caches = Vec::with_capacity(params.n_layers);
    let mut cache_shapes = Vec::with_capacity(params.n_layers);
    for i in 0..params.n_layers {
        let (c, shape) = read_npz_f32(&mut archive, &format!("cache{i}"))?;
        base_offs.push(off0[0]);
        cache_shapes.push(shape.iter().map(|d| *d as i64).collect());
        base_caches.push(c);
    }
    let off0_val = off0[0] as usize;

    // Tokenizer (pure-Rust SentencePiece via pocket-tts) + cond embeddings.
    let tok_path = assets_dir.join("tokenizer.model");
    anyhow::ensure!(tok_path.is_file(), "missing tokenizer.model");
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let conditioner = LUTConditioner::new(4000, &tok_path, params.dim, params.dim, vb)?;
    let embed_bytes = std::fs::read(assets_dir.join("cond_embed.npy"))?;
    let (cond_embed, embed_shape) = match parse_npy(&embed_bytes)? {
        NpyArray::F32(v, s) => (v, s),
        _ => anyhow::bail!("cond_embed.npy is not f32"),
    };
    anyhow::ensure!(
        embed_shape.len() == 2 && embed_shape[1] == params.dim,
        "cond_embed shape mismatch: {:?}",
        embed_shape
    );
    if std::env::var("SYNTH_DEBUG").is_ok() {
        eprintln!("DBG embed_shape={embed_shape:?}");
    }

    // Mimi state layout (discovered from the IR, like synth.py).
    let n_mimi_in = models.mimi.get_input_size()?;
    let mut mimi_inputs = Vec::new();
    for i in 1..n_mimi_in {
        let node = models.mimi.get_input_by_index(i)?;
        let name = node.get_name()?;
        let shape: Vec<i64> = node.get_shape()?.get_dimensions().to_vec();
        mimi_inputs.push(MimiInput {
            kind: element_kind(&name),
            name,
            shape,
        });
    }

    // Sentences -> token ids -> chunks (신版), or whole text at once (구版).
    let t1 = Instant::now();
    let mut segs: Vec<f32> = Vec::new();
    if args.single {
        let (fmt, guess) =
            prepare_text_prompt(&args.text, params.pad_spaces, params.rm_semi, params.append_punct)?;
        let ids = conditioner.encode_ids(&fmt)?;
        let tc = ids.len();
        let max_gen = estimate_gen(tc, &params);
        println!("text tokens={tc}, max_gen={max_gen}, voice off={off0_val}");
        anyhow::ensure!(
            off0_val + tc + max_gen <= params.max_len,
            "MAX_LEN 초과: 더 긴 컨텍스트 모델 필요"
        );
        let (audio, nfr) = synth_ids(
            &mut models,
            &mimi_inputs,
            &ids,
            guess,
            &cond_embed,
            &params,
            &base_offs,
            &base_caches,
            &cache_shapes,
            &mut rng,
        )?;
        segs.extend(audio);
        println!("AR: {nfr} frames");
    } else {
        let sents = split_sentences(&args.text);
        anyhow::ensure!(!sents.is_empty(), "empty");
        let mut sent_ids = Vec::new();
        let mut sent_words = Vec::new();
        for s in &sents {
            let (fmt, _) = prepare_text_prompt(s, params.pad_spaces, params.rm_semi, params.append_punct)?;
            sent_ids.push(conditioner.encode_ids(&fmt)?);
            sent_words.push(fmt.split_whitespace().count());
        }
        let chunks = pack_chunks(&sent_ids, &sent_words, &params, off0_val);
        println!("문장 {}개 -> 청크 {}개", sents.len(), chunks.len());
        for (ci, chunk) in chunks.iter().enumerate() {
        // NOTE: synth.py joins sentences with `sp.encode(" ")[:1] or []`,
        // and SentencePiece returns [] for whitespace-only input, so the
        // separator is effectively absent. Our Unigram port would emit a
        // real id for " ", which would corrupt the prefill — concat directly.
        let mut flat: Vec<u32> = Vec::new();
        let mut words = 0usize;
        for &si in chunk {
            flat.extend_from_slice(&sent_ids[si]);
            words += sent_words[si];
        }
        let guess = if words <= 4 { 3 } else { 1 };
        let (audio, nfr) = synth_ids(
            &mut models,
            &mimi_inputs,
            &flat,
            guess,
            &cond_embed,
            &params,
            &base_offs,
            &base_caches,
            &cache_shapes,
            &mut rng,
        )?;
        segs.extend(audio);
        println!(
            "청크 {}/{}: {} frames, {:.2}s",
            ci + 1,
            chunks.len(),
            nfr,
            segs.len() as f64 / params.sample_rate as f64
        );
        }
    }
    let gen_t = t1.elapsed().as_secs_f64();
    let dur = segs.len() as f64 / params.sample_rate as f64;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: params.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&args.out, spec)?;
    for s in &segs {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    let peak = segs.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    println!("저장: {} ({dur:.2}s 오디오, {:.2}x)", args.out, dur / gen_t);
    println!("peak={peak:.3}");
    Ok(())
}
