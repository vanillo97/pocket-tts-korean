#!/usr/bin/env python3
"""bench_orig_onnx.py — 원본(PyTorch) + ONNX 벤치마크 (Rust 하네스와 동일 조건).

run_live_benchmark.py §1/§3를 test-assets 경로 + 독립 qa로 수정한 버전.
- TEXT/VOICE/CONFIG, warmup 999 + 실측 3회(seed 0,1,2), e2e 타이밍 동일
- JSON 스키마: live_bench.rs BenchRow와 동일 키
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile

WORKSPACE = Path(__file__).resolve().parent
TEXT = "안녕하세요. 한국어 음성 합성 모델입니다."
VOICE = str(WORKSPACE / "2_0000.wav")
CONFIG = "hf://seastar105/pocket-tts-korean-300m/korean.yaml"
OUTDIR = Path("/tmp/bench_py")
OUTDIR.mkdir(exist_ok=True, parents=True)

results = []


def qa(path):
    """peak/rms/centroid — live_bench.rs qa_wav와 동일 정의."""
    sr, data = scipy.io.wavfile.read(str(path))
    a = np.asarray(data, dtype=np.float64)
    if a.ndim > 1:
        a = a[:, 0]
    peak = float(np.abs(a).max())
    rms = float(np.sqrt((a ** 2).mean()))
    n = 4096
    hann = np.hanning(n)
    spec = np.abs(np.fft.rfft(a[: len(a) // n * n].reshape(-1, n) * hann, axis=1))
    mag = spec.sum(axis=1)
    ok = mag > 1e-9
    if not ok.any():
        return {"peak": peak, "rms": rms, "cent": 0.0}
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    cent = float(((spec[ok] * freqs).sum(axis=1) / mag[ok]).mean())
    return {"peak": peak, "rms": rms, "cent": cent}


def add_result(model_name, backend, threads, quant, scope, status, gen_s, audio_s,
               qa_res=None, note=""):
    mean_gen = float(np.mean(gen_s)) if gen_s else None
    std_gen = float(np.std(gen_s)) if gen_s else None
    r = {
        "model": model_name, "backend": backend, "threads": threads, "quant": quant,
        "scope": scope, "status": status,
        "gen_mean_s": round(mean_gen, 3) if mean_gen else None,
        "gen_std_s": round(std_gen, 3) if std_gen else None,
        "gen_min_s": round(float(np.min(gen_s)), 3) if gen_s else None,
        "gen_max_s": round(float(np.max(gen_s)), 3) if gen_s else None,
        "audio_mean_s": round(float(np.mean(audio_s)), 3) if audio_s else None,
        "speed_x": round(float(np.mean(audio_s)) / mean_gen, 2) if gen_s else None,
        "rtf": round(mean_gen / float(np.mean(audio_s)), 4) if gen_s else None,
        "qa_peak": round(float(qa_res["peak"]), 3) if qa_res else None,
        "qa_rms": round(float(qa_res["rms"]), 4) if qa_res else None,
        "qa_cent": round(float(qa_res["cent"]), 0) if qa_res else None,
        "note": note,
    }
    results.append(r)
    speed = f"{r['speed_x']:.2f}x (RTF {r['rtf']:.3f})" if r["speed_x"] else "N/A"
    print(f"[{status}] {model_name:<14} | {backend:<6} | {threads:<6} | {quant:<12} | {scope:<8} -> gen: {r['gen_mean_s']}s, speed: {speed} | {note}")
    sys.stdout.flush()


def run_torch_bench():
    print("\n>>> [1/2] PyTorch 원본 벤치마크 시작...")
    import torch
    from pocket_tts import TTSModel

    if not torch.cuda.is_available():
        add_result("원본-torch", "GPU", "multi", "FP32", "e2e", "N/A", [], [],
                   note="PyTorch CPU-only 빌드 (CUDA 미탑재)")

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

                torch.manual_seed(999)
                _ = model.generate_audio(vs, TEXT, copy_state=True)

                gens, durs, last_wav = [], [], None
                for i in range(3):
                    torch.manual_seed(i)
                    t0 = time.perf_counter()
                    audio = model.generate_audio(vs, TEXT, copy_state=True)
                    gens.append(time.perf_counter() - t0)
                    durs.append(audio.numel() / sr)
                    if i == 0:
                        last_wav = OUTDIR / f"torch_cpu{th}T_{q_label}.wav"
                        scipy.io.wavfile.write(str(last_wav), sr, audio.detach().cpu().numpy())

                add_result("원본-torch", "CPU", f"{th}T", q_label, "e2e", "OK",
                           gens, durs, qa(str(last_wav)))
                del vs, model
                gc.collect()
            except Exception as e:
                add_result("원본-torch", "CPU", f"{th}T", q_label, "e2e", "FAIL",
                           [], [], note=str(e)[:200])


def run_onnx_bench():
    print("\n>>> [2/2] ONNX Runtime 벤치마크 시작...")
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
    vs = model.get_state_for_audio_prompt(VOICE)
    prepared = flow.conditioner.prepare(TEXT)
    tc = prepared.shape[1]
    mg = model._estimate_max_gen_len(tc)
    names = [ly.self_attn._module_absolute_name for ly in flow.transformer.layers]

    onnx_files = {
        "FP32": str(WORKSPACE / "test-assets" / "pocket_step_uni256.onnx"),
        "INT8-dyn": str(WORKSPACE / "test-assets" / "pocket_step_uni256_int8.onnx"),
    }

    for q_label, onnx_p in onnx_files.items():
        if not Path(onnx_p).exists():
            for th in [1, 8]:
                add_result("ONNX-ORT", "CPU", f"{th}T", q_label, "e2e", "N/A", [], [],
                           note=f"ONNX 파일 없음: {Path(onnx_p).name}")
            continue
        for th in [1, 8]:
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = th
                opts.inter_op_num_threads = 1
                sess = ort.InferenceSession(onnx_p, sess_options=opts,
                                            providers=["CPUExecutionProvider"])

                def run_onnx_once(seed):
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
                        pad = np.full((2, 1, 256 - c.shape[2]) + c.shape[3:], np.nan,
                                      dtype=np.float32)
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
                        lat = decode_fn(partial(flow.flow_net, tout[:, -1]), noise, 1)
                        eos = float(flow.out_eos(tout[:, -1].to(torch.float32)).item())
                        if eos > model.eos_threshold and eos_step is None:
                            eos_step = step_i
                        if eos_step is not None and step_i >= eos_step + 3:
                            break
                        latents.append(lat)
                        token = lat.detach().numpy().astype(np.float32)[:, None, :]
                    if latents:
                        mimi_state = init_states(model.mimi, batch_size=1,
                                                sequence_length=mg * steps_per)
                        chunks = []
                        for lat in latents:
                            inp = (lat * flow.emb_std + flow.emb_mean)[:, None, :]
                            fr = model.mimi.decode_from_latent(inp, mimi_state)
                            increment_steps(model.mimi, mimi_state, increment=steps_per)
                            chunks.append(fr[0, 0].detach().cpu())
                        audio = torch.cat(chunks, dim=0).numpy() if chunks else np.zeros(0, dtype=np.float32)
                    else:
                        audio = np.zeros(0, dtype=np.float32)
                    return time.perf_counter() - t0, len(audio) / model.sample_rate, audio, model.sample_rate

                _ = run_onnx_once(999)
                gens, durs, last_wav = [], [], None
                for i in range(3):
                    g, d, a, sr_ = run_onnx_once(i)
                    gens.append(g)
                    durs.append(d)
                    if i == 0:
                        last_wav = OUTDIR / f"onnx_cpu{th}T_{q_label}.wav"
                        scipy.io.wavfile.write(str(last_wav), sr_, a)
                add_result("ONNX-ORT", "CPU", f"{th}T", q_label, "e2e", "OK",
                           gens, durs, qa(str(last_wav)))
            except Exception as e:
                import traceback
                traceback.print_exc()
                add_result("ONNX-ORT", "CPU", f"{th}T", q_label, "e2e", "FAIL",
                           [], [], note=str(e)[:200])


def main():
    t_start = time.perf_counter()
    run_torch_bench()
    run_onnx_bench()
    print(f"\n전체 완료! 소요 시간: {time.perf_counter() - t_start:.1f}초")
    out_json = OUTDIR / "bench_orig_onnx.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"결과 JSON 저장: {out_json}")


if __name__ == "__main__":
    main()
