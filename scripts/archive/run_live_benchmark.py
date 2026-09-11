#!/usr/bin/env python3
"""run_live_benchmark.py

실시간 라이브 벤치마크 실행 스크립트:
- 모델: 원본-PyTorch, ONNXRuntime, OpenVINO IR, 배포 패키지
- 백엔드: CPU (1T, 8T), GPU (iGPU GPU.0, dGPU GPU.1), NPU (Intel AI Boost, DeepX)
- 양자화: FP32, INT8, INT4(W4)
- 반복: warmup 1회(seed 999) + 실측 3회 (seed 0,1,2)

동일척도 원칙 (전 모델 공통):
- gen 시간 = e2e 합성 시간 (text prefill + AR 루프 + sampler + mimi 디코딩)
- 모델 로드/컴파일 시간은 제외 (타이머 밖에서 1회 수행)
- voice conditioning은 1회만 수행하고 측정 루프 밖으로 분리
"""

import os
import sys
import time
import json
import numpy as np
import scipy.io.wavfile
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE / "python_backup"))
from qa_wav import qa

TEXT = "안녕하세요. 한국어 음성 합성 모델입니다."
VOICE = str(WORKSPACE / "2_0000.wav")
CONFIG = "hf://seastar105/pocket-tts-korean-300m/korean.yaml"
OUTDIR = WORKSPACE / "live_bench_wav"
OUTDIR.mkdir(exist_ok=True, parents=True)

results = []

def add_result(model_name, backend, threads, quant, scope, status, gen_s, audio_s, qa_res=None, note=""):
    mean_gen = float(np.mean(gen_s)) if gen_s else None
    std_gen = float(np.std(gen_s)) if gen_s else None
    min_gen = float(np.min(gen_s)) if gen_s else None
    max_gen = float(np.max(gen_s)) if gen_s else None
    mean_dur = float(np.mean(audio_s)) if audio_s else None
    speed_x = float(mean_dur / mean_gen) if (mean_dur and mean_gen) else None
    rtf = float(mean_gen / mean_dur) if (mean_dur and mean_gen) else None

    r = {
        "model": model_name,
        "backend": backend,
        "threads": threads,
        "quant": quant,
        "scope": scope,
        "status": status,
        "gen_mean_s": round(mean_gen, 3) if mean_gen else None,
        "gen_std_s": round(std_gen, 3) if std_gen else None,
        "gen_min_s": round(min_gen, 3) if min_gen else None,
        "gen_max_s": round(max_gen, 3) if max_gen else None,
        "audio_mean_s": round(mean_dur, 3) if mean_dur else None,
        "speed_x": round(speed_x, 2) if speed_x else None,
        "rtf": round(rtf, 4) if rtf else None,
        "qa_peak": round(float(qa_res["peak"]), 3) if qa_res else None,
        "qa_rms": round(float(qa_res["rms"]), 4) if qa_res else None,
        "qa_cent": round(float(qa_res["cent"]), 0) if qa_res else None,
        "note": note,
    }
    results.append(r)
    speed_str = f"{r['speed_x']:.2f}x (RTF {r['rtf']:.3f})" if r['speed_x'] else "N/A"
    print(f"[{status}] {model_name:<14} | {backend:<6} | {threads:<6} | {quant:<12} | {scope:<8} -> gen: {r['gen_mean_s']}s, speed: {speed_str} | {note}")
    sys.stdout.flush()

# ================= 1. PyTorch 원본 =================
def run_torch_bench():
    print("\n>>> [1/4] PyTorch 원본 벤치마크 시작...")
    import torch
    from pocket_tts import TTSModel
    
    # CUDA 확인
    cuda_avail = torch.cuda.is_available()
    if not cuda_avail:
        add_result("원본-torch", "GPU", "multi", "FP32", "e2e", "N/A", [], [], note="PyTorch CPU-only 빌드 (CUDA 미탑재)")

    for quant in [False, True]:
        q_label = "INT8-dyn" if quant else "FP32"
        for th in [1, 8]:
            torch.set_num_threads(th)
            torch.set_grad_enabled(False)
            try:
                model = TTSModel.load_model(config=CONFIG, quantize=quant)
                model.eval()
                vs = model.get_state_for_audio_prompt(VOICE)
                sr = model.sample_rate

                # Warmup (실측 seed와 겹치지 않게 999)
                torch.manual_seed(999)
                _ = model.generate_audio(vs, TEXT, copy_state=True)

                gens, durs = [], []
                last_wav = None
                for i in range(3):
                    torch.manual_seed(i)
                    t0 = time.perf_counter()
                    audio = model.generate_audio(vs, TEXT, copy_state=True)
                    dt = time.perf_counter() - t0
                    dur = audio.numel() / sr
                    gens.append(dt)
                    durs.append(dur)
                    if i == 0:
                        last_wav = OUTDIR / f"torch_cpu{th}T_{q_label}.wav"
                        scipy.io.wavfile.write(str(last_wav), sr, audio.detach().cpu().numpy())

                qa_res = qa(str(last_wav)) if last_wav else None
                add_result("원본-torch", "CPU", f"{th}T", q_label, "e2e", "OK", gens, durs, qa_res)
                del vs, model
                import gc; gc.collect()
            except Exception as e:
                add_result("원본-torch", "CPU", f"{th}T", q_label, "e2e", "FAIL", [], [], note=str(e))

# ================= 2. OpenVINO IR =================
def run_ov_bench():
    print("\n>>> [2/4] OpenVINO IR 벤치마크 시작...")
    import openvino as ov
    import sentencepiece as spm
    core = ov.Core()

    step_models = {
        "FP32": str(WORKSPACE / "deploy_package_fp32_fp32_256" / "models" / "step.xml"),
        "INT8": str(WORKSPACE / "deploy_package_int8_fp32_256" / "models" / "step.xml"),
        "INT4": str(WORKSPACE / "deploy_package_int4_int8_256" / "models" / "step.xml"),
    }
    sampler_fp32_xml = str(WORKSPACE / "deploy_package_fp32_fp32_256" / "models" / "sampler.xml")
    sampler_int8_xml = str(WORKSPACE / "deploy_package_int4_int8_256" / "models" / "sampler.xml")
    mimi_xml = str(WORKSPACE / "deploy_package_fp32_fp32_256" / "models" / "mimi.xml")
    assets_dir = WORKSPACE / "deploy_package_int4_int8_256" / "assets"
    
    P = json.load(open(assets_dir / "params.json"))
    vc = np.load(assets_dir / "voice_cache.npz")
    base_offs = [vc["off0"].copy() for _ in range(P["n_layers"])]
    base_caches = [vc[f"cache{i}"] for i in range(P["n_layers"])]
    off0, MAX = int(vc["off0"][0]), P["max_len"]
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(assets_dir / "tokenizer.model"))
    embed = np.load(str(assets_dir / "cond_embed.npy"))

    fmt = TEXT.strip()
    if not fmt[0].isupper(): fmt = fmt[0].upper() + fmt[1:]
    if fmt[-1].isalnum(): fmt = fmt + "."
    ids = sp.encode(fmt, out_type=int)
    tc = len(ids)
    max_gen = int(np.ceil((tc / P["tokens_per_sec"] + P["gen_pad_sec"]) * P["frame_rate"]))

    # Test CPU 1T and 8T across FP32, INT8, INT4
    for q_label, step_xml in step_models.items():
        # INT4 패키지 실제 조합(step INT4 + sampler INT8)에 맞춤
        sampler_xml = sampler_int8_xml if q_label in ("INT8", "INT4") else sampler_fp32_xml
        for th in [1, 8]:
            cfg = {"INFERENCE_NUM_THREADS": 1} if th == 1 else {"INFERENCE_NUM_THREADS": 8}
            try:
                c_step = core.compile_model(step_xml, "CPU", cfg)
                c_sampler = core.compile_model(sampler_xml, "CPU", cfg)
                c_mimi = core.compile_model(mimi_xml, "CPU", cfg)

                def run_once(seed):
                    rng = np.random.default_rng(seed)
                    offs = [o.copy() for o in base_offs]
                    caches = [c.copy() for c in base_caches]
                    text_emb = embed[np.array(ids, dtype=np.int64)][None, :, :].astype(np.float32)
                    # e2e 동일척도: text prefill + AR + mimi 전체를 타이밍
                    t0 = time.perf_counter()

                    def run_s(tok, emb, flag):
                        feed = {"token": tok, "emb": emb, "is_text": flag}
                        for idx in range(P["n_layers"]):
                            feed[f"off{idx}"] = offs[idx]
                            feed[f"cache{idx}"] = caches[idx]
                        return c_step(feed)

                    nan_tok = np.full((1, 1, P["ldim"]), np.nan, dtype=np.float32)
                    zero_emb = np.zeros((1, 1, P["dim"]), dtype=np.float32)
                    for t in range(tc):
                        res = run_s(nan_tok, text_emb[:, t:t+1, :], np.ones(1, dtype=bool))
                        for idx in range(P["n_layers"]):
                            caches[idx] = np.asarray(res[c_step.output(f"new_cache{idx}")])
                            offs[idx] += 1

                    token = np.full((1, 1, P["ldim"]), np.nan, dtype=np.float32)
                    latents, eos_step = [], None
                    std = P["temp"] ** 0.5
                    for s in range(max_gen):
                        res = run_s(token, zero_emb, np.zeros(1, dtype=bool))
                        tout = np.asarray(res[c_step.output("out")])
                        for idx in range(P["n_layers"]):
                            caches[idx] = np.asarray(res[c_step.output(f"new_cache{idx}")])
                            offs[idx] += 1
                        noise = rng.normal(0, std, size=(1, P["ldim"])).astype(np.float32)
                        sr_ = c_sampler({"cond": tout[:, -1:].reshape(1, -1), "noise": noise})
                        lat = np.asarray(sr_[c_sampler.output("latent")])
                        eos = float(np.asarray(sr_[c_sampler.output("eos")])[0, 0])
                        if eos > P["eos_threshold"] and eos_step is None:
                            eos_step = s
                        if eos_step is not None and s >= eos_step + 3:
                            break
                        latents.append(lat)
                        token = lat[:, None, :]

                    mi = c_mimi.inputs
                    mnames = [inp.any_name for inp in mi[1:]]
                    st = {}
                    for inp in mi[1:]:
                        n, shp = inp.any_name, tuple(inp.shape)
                        if "off" in n: st[n] = np.zeros(shp, dtype=np.int64)
                        elif "cache" in n: st[n] = np.full(shp, np.nan, dtype=np.float32)
                        else: st[n] = np.zeros(shp, dtype=np.float32)
                    estd = np.array(P["emb_std"], dtype=np.float32)
                    emean = np.array(P["emb_mean"], dtype=np.float32)
                    chunks = []
                    for lat in latents:
                        f = dict(st)
                        f["latent"] = (lat.reshape(1, 1, -1) * estd + emean).astype(np.float32)
                        res = c_mimi(f)
                        chunks.append(np.asarray(res[c_mimi.output("audio")])[0, 0])
                        for j, nm in enumerate(mnames):
                            st[nm] = np.asarray(res[c_mimi.output(f"n{j}")])
                    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
                    gen_t = time.perf_counter() - t0  # e2e: prefill+AR+mimi 포함
                    return gen_t, len(audio) / P["sample_rate"], audio

                # warmup
                _ = run_once(999)
                gens, durs, last_wav = [], [], None
                for i in range(3):
                    g, d, a = run_once(i)
                    gens.append(g)
                    durs.append(d)
                    if i == 0:
                        last_wav = OUTDIR / f"ov_cpu{th}T_{q_label}.wav"
                        scipy.io.wavfile.write(str(last_wav), P["sample_rate"], a)
                qa_res = qa(str(last_wav)) if last_wav else None
                add_result("IR-OV", "CPU", f"{th}T", q_label, "e2e", "OK", gens, durs, qa_res)
            except Exception as e:
                add_result("IR-OV", "CPU", f"{th}T", q_label, "e2e", "FAIL", [], [], note=str(e))

    # Test GPU (장치명 자동 탐색: 구버전 GPU.0/GPU.1, 신버전 GPU)
    gpu_devs = [d for d in core.available_devices if d == "GPU" or d.startswith("GPU.")]
    if not gpu_devs:
        add_result("IR-OV", "GPU", "multi", "INT4", "e2e", "N/A", [], [], note="GPU 장치 없음")
    for dev in gpu_devs:
        if True:  # gpu_devs는 available_devices에서 탐색된 실존 장치
            try:
                c_step = core.compile_model(step_models["INT4"], dev)
                c_sampler = core.compile_model(sampler_int8_xml, dev)
                c_mimi = core.compile_model(mimi_xml, "CPU")

                def run_gpu(seed):
                    rng = np.random.default_rng(seed)
                    offs = [o.copy() for o in base_offs]
                    caches = [c.copy() for c in base_caches]
                    text_emb = embed[np.array(ids, dtype=np.int64)][None, :, :].astype(np.float32)
                    # e2e 동일척도: text prefill + AR + mimi 전체를 타이밍
                    t0 = time.perf_counter()

                    def run_s(tok, emb, flag):
                        feed = {"token": tok, "emb": emb, "is_text": flag}
                        for idx in range(P["n_layers"]):
                            feed[f"off{idx}"] = offs[idx]
                            feed[f"cache{idx}"] = caches[idx]
                        return c_step(feed)

                    nan_tok = np.full((1, 1, P["ldim"]), np.nan, dtype=np.float32)
                    zero_emb = np.zeros((1, 1, P["dim"]), dtype=np.float32)
                    for t in range(tc):
                        res = run_s(nan_tok, text_emb[:, t:t+1, :], np.ones(1, dtype=bool))
                        for idx in range(P["n_layers"]):
                            caches[idx] = np.asarray(res[c_step.output(f"new_cache{idx}")])
                            offs[idx] += 1

                    token = np.full((1, 1, P["ldim"]), np.nan, dtype=np.float32)
                    latents, eos_step = [], None
                    std = P["temp"] ** 0.5
                    for s in range(max_gen):
                        res = run_s(token, zero_emb, np.zeros(1, dtype=bool))
                        tout = np.asarray(res[c_step.output("out")])
                        for idx in range(P["n_layers"]):
                            caches[idx] = np.asarray(res[c_step.output(f"new_cache{idx}")])
                            offs[idx] += 1
                        noise = rng.normal(0, std, size=(1, P["ldim"])).astype(np.float32)
                        sr_ = c_sampler({"cond": tout[:, -1:].reshape(1, -1), "noise": noise})
                        lat = np.asarray(sr_[c_sampler.output("latent")])
                        eos = float(np.asarray(sr_[c_sampler.output("eos")])[0, 0])
                        if eos > P["eos_threshold"] and eos_step is None:
                            eos_step = s
                        if eos_step is not None and s >= eos_step + 3:
                            break
                        latents.append(lat)
                        token = lat[:, None, :]

                    mi = c_mimi.inputs
                    mnames = [inp.any_name for inp in mi[1:]]
                    st = {}
                    for inp in mi[1:]:
                        n, shp = inp.any_name, tuple(inp.shape)
                        if "off" in n: st[n] = np.zeros(shp, dtype=np.int64)
                        elif "cache" in n: st[n] = np.full(shp, np.nan, dtype=np.float32)
                        else: st[n] = np.zeros(shp, dtype=np.float32)
                    estd = np.array(P["emb_std"], dtype=np.float32)
                    emean = np.array(P["emb_mean"], dtype=np.float32)
                    chunks = []
                    for lat in latents:
                        f = dict(st)
                        f["latent"] = (lat.reshape(1, 1, -1) * estd + emean).astype(np.float32)
                        res = c_mimi(f)
                        chunks.append(np.asarray(res[c_mimi.output("audio")])[0, 0])
                        for j, nm in enumerate(mnames):
                            st[nm] = np.asarray(res[c_mimi.output(f"n{j}")])
                    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
                    gen_t = time.perf_counter() - t0  # e2e: prefill+AR+mimi 포함
                    return gen_t, len(audio) / P["sample_rate"], audio

                _ = run_gpu(999)
                gens, durs, last_wav = [], [], None
                for i in range(3):
                    g, d, a = run_gpu(i)
                    gens.append(g)
                    durs.append(d)
                    if i == 0:
                        last_wav = OUTDIR / f"ov_{dev}_w4.wav"
                        scipy.io.wavfile.write(str(last_wav), P["sample_rate"], a)
                qa_res = qa(str(last_wav)) if last_wav else None
                note_str = "품질 저하 주의(RMS 낮음)" if qa_res and qa_res["rms"] < 0.05 else ""
                add_result("IR-OV", dev, "multi", "INT4", "e2e", "OK", gens, durs, qa_res, note=note_str)
            except Exception as e:
                add_result("IR-OV", dev, "multi", "INT4", "e2e", "FAIL", [], [], note=f"{type(e).__name__}: {e}")

    # Test NPU (Intel AI Boost)
    if "NPU" in core.available_devices:
        try:
            print("  -> NPU 컴파일 및 실행 검증 시도 중 (Intel AI Boost)...")
            c_step = core.compile_model(step_models["INT4"], "NPU")
            feed_dummy = {
                "token": np.full((1, 1, P["ldim"]), np.nan, dtype=np.float32),
                "emb": np.zeros((1, 1, P["dim"]), dtype=np.float32),
                "is_text": np.zeros(1, dtype=bool)
            }
            for i in range(P["n_layers"]):
                feed_dummy[f"off{i}"] = np.array([41], dtype=np.int64)
                feed_dummy[f"cache{i}"] = np.full((2, 1, 256, 16, 64), np.nan, dtype=np.float32)
            # Try running 1 step
            _ = c_step(feed_dummy)
            add_result("IR-OV", "NPU", "multi", "INT4", "e2e", "OK", [], [], note="NPU 컴파일+1스텝 성공 (전체합성 미측정)")
        except Exception as e:
            err_msg = str(e).replace("\n", " ")[:90]
            add_result("IR-OV", "NPU", "multi", "INT4", "e2e", "FAIL", [], [], note=f"NPU 비호환 ({err_msg})")

# ================= 3. ONNX Runtime =================
def run_onnx_bench():
    print("\n>>> [3/4] ONNX Runtime 벤치마크 시작...")
    import onnxruntime as ort
    from pocket_tts import TTSModel
    from pocket_tts.models.flow_lm import lsd_decode, ot_decode
    from pocket_tts.modules.stateful_module import init_states, increment_steps
    from functools import partial
    import torch
    torch.set_grad_enabled(False)
    model = TTSModel.load_model(config=CONFIG)
    model.eval()
    flow = model.flow_lm
    decode_fn = ot_decode if flow.flow_type == "flow_matching" else lsd_decode
    # TEXT 포맷팅을 모델 prepare와 동일하게 처리
    vs = model.get_state_for_audio_prompt(VOICE)
    prepared = flow.conditioner.prepare(TEXT)
    tc = prepared.shape[1]
    mg = model._estimate_max_gen_len(tc)
    names = [ly.self_attn._module_absolute_name for ly in flow.transformer.layers]

    onnx_files = {
        "FP32": str(WORKSPACE / "pocket_step_uni256.onnx"),
        "INT8-dyn": str(WORKSPACE / "pocket_step_uni256_int8.onnx")
    }

    for q_label, onnx_p in onnx_files.items():
        if not os.path.exists(onnx_p):
            for th in [1, 8]:
                add_result("ONNX-ORT", "CPU", f"{th}T", q_label, "e2e", "N/A", [], [],
                           note=f"ONNX 파일 없음: {os.path.basename(onnx_p)} (해당 export 파이프라인 부재)")
            continue
        for th in [1, 8]:
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = th
                opts.inter_op_num_threads = 1
                sess = ort.InferenceSession(onnx_p, sess_options=opts, providers=["CPUExecutionProvider"])

                def run_onnx_once(seed):
                    rng = np.random.default_rng(seed)
                    # e2e 동일척도: state복제→캐시확장→text prefill→AR→mimi 전체를 타이밍
                    t0 = time.perf_counter()
                    st = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in vs.items()}
                    cur = model._flow_lm_current_end(st)
                    model._expand_kv_cache(st, cur + tc + mg)
                    model._run_flow_lm_and_increment_step(model_state=st, text_tokens=prepared)
                    o_curr, c_curr = [], []
                    for n in names:
                        s = st[n]
                        o_curr.append(int(s["offset"].view(-1)[0].item()))
                        c = s["cache"][:, :1].detach().numpy().astype(np.float32)
                        pad = np.full((2, 1, 256 - c.shape[2]) + c.shape[3:], np.nan, dtype=np.float32)
                        c_curr.append(np.concatenate([c, pad], axis=2))
                    token = np.full((1, 1, flow.ldim), np.nan, dtype=np.float32)
                    latents, eos_step = [], None
                    std = model.temp ** 0.5
                    zero_emb = np.zeros((1, 1, flow.dim), dtype=np.float32)
                    f_flag = np.zeros(1, dtype=bool)
                    steps_per = int(model.mimi.encoder_frame_rate / model.mimi.frame_rate)
                    for step_i in range(mg):
                        feed = {"token": token, "emb": zero_emb, "is_text": f_flag}
                        for i, (o, c) in enumerate(zip(o_curr, c_curr)):
                            feed[f"off{i}"] = np.array([o], dtype=np.int64)
                            feed[f"cache{i}"] = c
                        res = sess.run(None, feed)
                        tout = torch.from_numpy(np.asarray(res[0]))
                        for i in range(len(c_curr)):
                            c_curr[i] = np.asarray(res[1 + i])
                            o_curr[i] += 1
                        noise = torch.empty((1, flow.ldim))
                        torch.nn.init.normal_(noise, mean=0.0, std=std)
                        # torch RNG 대신 numpy RNG 일관성 위해 numpy 기반 noise 사용 시 아래로 교체 가능
                        # noise = torch.from_numpy(rng.normal(0, std, size=(1, flow.ldim)).astype(np.float32))
                        lat = decode_fn(partial(flow.flow_net, tout[:, -1]), noise, 1)
                        eos = float(flow.out_eos(tout[:, -1].to(torch.float32)).item())
                        if eos > model.eos_threshold and eos_step is None:
                            eos_step = step_i
                        if eos_step is not None and step_i >= eos_step + 3:
                            break
                        latents.append(lat)
                        token = lat.detach().numpy().astype(np.float32)[:, None, :]
                    # mimi 디코딩을 e2e 타이밍에 포함
                    if latents:
                        mimi_state = init_states(model.mimi, batch_size=1, sequence_length=mg * steps_per)
                        chunks = []
                        for lat in latents:
                            inp = lat * flow.emb_std + flow.emb_mean
                            # inp shape: [1, 1, ldim] -> decode_from_latent expects [B, T, C]
                            fr = model.mimi.decode_from_latent(inp, mimi_state)
                            increment_steps(model.mimi, mimi_state, increment=steps_per)
                            chunks.append(fr[0, 0].detach().cpu())
                        audio = torch.cat(chunks, dim=0).numpy() if chunks else np.zeros(0, dtype=np.float32)
                    else:
                        audio = np.zeros(0, dtype=np.float32)
                    gen_t = time.perf_counter() - t0  # e2e: prefill+AR+mimi 포함
                    return gen_t, len(audio) / model.sample_rate, audio, model.sample_rate

                _ = run_onnx_once(999)
                gens, durs, last_wav = [], [], None
                for i in range(3):
                    g, d, a, sr_ = run_onnx_once(i)
                    gens.append(g)
                    durs.append(d)
                    if i == 0:
                        last_wav = OUTDIR / f"onnx_cpu{th}T_{q_label}.wav"
                        scipy.io.wavfile.write(str(last_wav), sr_, a)
                qa_res = qa(str(last_wav)) if last_wav else None
                add_result("ONNX-ORT", "CPU", f"{th}T", q_label, "e2e", "OK", gens, durs, qa_res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                add_result("ONNX-ORT", "CPU", f"{th}T", q_label, "e2e", "FAIL", [], [], note=str(e))

# ================= 4. Deploy Packages (End-to-End) =================
def run_deploy_packages_bench():
    print("\n>>> [4/4] 배포 패키지 End-to-End 벤치마크 시작 (e2e 동일척도: wall - OV로드)...")
    import re
    import subprocess
    pkgs = sorted([d for d in WORKSPACE.glob("deploy_package_*") if (d / "synth.py").exists()])
    py_bin = sys.executable

    def quant_label(pkg_name):
        core = pkg_name.replace("deploy_package_", "")
        *rest, ctx = core.split("_")
        base = "_".join(rest).upper().replace("_", "/")
        return f"{base}-{ctx}ctx"

    for pkg_path in pkgs:
        pkg_name = pkg_path.name
        q_desc = quant_label(pkg_name)
        gens, durs = [], []
        last_wav = None
        fail_note = ""
        for i in range(3):
            out_f = OUTDIR / f"{pkg_name}_r{i}.wav"
            t0 = time.perf_counter()
            r = subprocess.run([py_bin, str(pkg_path / "synth.py"), "--pkg", str(pkg_path),
                                "--text", TEXT, "--out", str(out_f), "--seed", str(i)],
                               capture_output=True, text=True)
            wall_t = time.perf_counter() - t0
            if r.returncode != 0:
                fail_note = (r.stderr or r.stdout)[-200:]
                break
            # e2e 합성시간 = 프로세스 전체 - 모델 로드/컴파일(타이머 밖扱)
            m = re.search(r"OV 로드:\s*([\d.]+)s", r.stdout)
            load_t = float(m.group(1)) if m else 0.0
            try:
                sr_, data = scipy.io.wavfile.read(str(out_f))
                audio_dur = len(data) / sr_
            except Exception as e:
                fail_note = str(e)[:100]
                break
            gens.append(wall_t - load_t)
            durs.append(audio_dur)
            if i == 0:
                last_wav = OUTDIR / f"{pkg_name}.wav"
                if last_wav.exists():
                    last_wav.unlink()
                out_f.rename(last_wav)
            else:
                out_f.unlink(missing_ok=True)
        if len(gens) == 3:
            qa_res = qa(str(last_wav)) if last_wav else None
            add_result("IR-Deploy-E2E", "CPU", "multi", q_desc, "e2e", "OK", gens, durs, qa_res)
        else:
            add_result("IR-Deploy-E2E", "CPU", "multi", q_desc, "e2e", "FAIL", [], [], note=fail_note)

def main():
    t_start = time.perf_counter()
    run_torch_bench()
    run_ov_bench()
    run_onnx_bench()
    run_deploy_packages_bench()
    t_total = time.perf_counter() - t_start

    print(f"\n==========================================")
    print(f"전체 라이브 벤치마크 완료! 소요 시간: {t_total:.1f}초")
    out_json = WORKSPACE / "live_benchmark_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"결과 JSON 저장: {out_json}")

if __name__ == "__main__":
    main()
