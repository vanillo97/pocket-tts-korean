# Pocket-TTS 한국어 음성 합성 벤치마크 자료 (Rust 실측, 2026-09-11)

모델(original/ONNX/IR) × 백엔드(CPU/NPU/GPU) × 양자화 × 스레드 전 매트릭스를
Rust 하네스(`live_bench`, `onnx-bench`, `synth-rs`)로 실측하고,
Python 대조군(원본-torch, ONNX-ORT hybrid)은 동일 조건으로 오늘 재측정함.

- 공통 조건: 텍스트 `"안녕하세요. 한국어 음성 합성 모델입니다."`, voice `2_0000.wav`,
  config `hf://seastar105/pocket-tts-korean-300m/korean.yaml`
- 공통 프로토콜: 모델 로드/컴파일 제외 e2e 타이밍 (text prefill + AR 루프 + sampler + mimi),
  warmup 1회(seed 999) + 실측 3회(seed 0,1,2)
- speed_x = 오디오길이 / gen, RTF = gen / 오디오길이 (RTF<1 = 실시간 이상)

## 1. 하드웨어 / 소프트웨어 환경 (실측 머신)

| 항목 | 값 |
|---|---|
| CPU | Intel Core Ultra 9 285 (Arrow Lake-S, 24코어), L3 36MiB |
| RAM | 61Gi (가용 55Gi), Ubuntu 26.04.1, kernel 7.0.0-30 |
| GPU | iGPU Arrow Lake-S (GPU.0) + Battlemage G31 dGPU ×2 (GPU.1/GPU.2, GPU.2는 phantom) |
| NPU | Intel Arrow Lake NPU (AI Boost, `/dev/accel0`) + DEEPX DX_M1 (OV 미노출) |
| OV 노출 장치 | `CPU, GPU.0, GPU.1, GPU.2, NPU` (openvino 2026.3.1) |
| torch | 2.13.0+cu130 (CUDA False → CPU-only로 동작, GPU 측정 불가) |
| onnxruntime | 1.29.0 (`CPUExecutionProvider`만, GPU EP 없음) |
| Rust | `target/release/{live_bench, onnx-bench, synth-rs}` (2026-09-11 빌드) |

## 2. 원본 모델 (original) — Rust candle vs Python torch

| 모델 | 스레드 | 양자화 | gen 평균(s) | 오디오(s) | 속도 | RTF | QA rms |
|---|---|---|---|---|---|---|---|
| 원본-torch (Python, 오늘 재측정) | 1T | FP32 | 3.169 | 4.32 | 1.36x | 0.736 | 0.0757 |
| 원본-torch | 8T | FP32 | 1.426 | 4.32 | 3.03x | 0.330 | 0.0757 |
| 원본-torch | 1T | INT8-dyn | 1.480 | 4.64 | 3.14x | 0.319 | 0.0794 |
| 원본-torch | 8T | INT8-dyn | 0.780 | 4.40 | 5.64x | 0.177 | 0.0792 |
| candle-rust (`live_bench`) | 1T | FP32 | 3.961 | 4.43 | 1.12x | 0.895 | 0.0735 |
| candle-rust (`live_bench`) | 8T | FP32 | 3.907 | 4.43 | 1.13x | 0.883 | 0.0735 |
| 원본-torch | GPU | FP32 | N/A | — | — | — | — (CPU-only 빌드) |

- candle FP32는 torch 1T급(3.9s vs 3.2s)이지만 **스레드 확장 없음**(1T≈8T, 메모리 바운드).
  Python 없이 실시간 합성이 필요할 때 유효한 원본급 대안.
- torch 8T INT8-dyn (0.78s, 5.64x)이 전체 최속.

## 3. ONNX 모델 — Rust hybrid vs Python hybrid

| 모델 | 스레드 | 양자화 | gen 평균(s) | 오디오(s) | 속도 | RTF | QA rms |
|---|---|---|---|---|---|---|---|
| ONNX-ORT hybrid (Python, 오늘 재측정) | 1T | FP32 | 5.972 | 4.19 | 0.70x | 1.426 | 0.0826 |
| ONNX-ORT hybrid | 8T | FP32 | 4.440 | 4.96 | 1.12x | 0.895 | 0.0883 |
| ONNX-ORT hybrid | 1T | INT8-dyn | 4.115 | 4.77 | 1.16x | 0.862 | 0.0896 |
| ONNX-ORT hybrid | 8T | INT8-dyn | 2.972 | 4.24 | 1.43x | 0.701 | 0.0843 |
| onnx-bench (Rust: ORT step + candle sampler/mimi) | 1T | FP32 | 8.306 | 4.59 | 0.55x | 1.811 | 0.0873 |
| onnx-bench (Rust) | 8T | FP32 | 6.073 | 4.59 | 0.76x | 1.324 | 0.0873 |
| onnx-bench (Rust) | 1T | INT8-dyn | 5.344 | 4.48 | 0.84x | 1.193 | 0.0847 |
| onnx-bench (Rust) | 8T | INT8-dyn | 4.958 | 4.48 | 0.90x | 1.107 | 0.0847 |

- ONNX는 step만 ONNX이고 sampler/mimi가 torch/candle이라 전 모델 중 최하위권.
- Rust가 Python보다 느린 이유: 매 프레임 candle sampler/mimi 실행 + 스텝 입력
  51개를 매번 ndarray로 재구성(캐시 50MB 복사). 첫 AR 스텝 출력은 Python과 1e-6 일치.
- ORT에 GPU EP가 없어 GPU/NPU 측정 불가.

## 4. IR 모델 — Rust `synth-rs` (OpenVINO)

| 양자화 (step/sampler-ctx) | 1T gen(s) | 8T gen(s) | 오디오(s) | 8T 속도 | 1T→8T | rms |
|---|---|---|---|---|---|---|
| FP32/FP32-256 | 7.30 | 3.33 | 4.51 | 1.35x | 2.19배 | 0.0830 |
| FP32/INT8-256 † | 8.17 | 3.77 | 5.20 | 1.38x | 2.17배 | 0.0794 |
| INT4/FP32-256 | 5.26 | 2.57 | 5.07 | 1.97x | 2.05배 | 0.0724 |
| INT4/INT8-1024 | 19.96 | 7.76 | 5.25 | 0.68x | 2.57배 | 0.0770 |
| INT4/INT8-256 † | 5.12 | 2.51 | 4.99 | 1.99x | 2.04배 | 0.0746 |
| INT8/FP32-256 | 5.60 | 2.71 | 5.17 | 1.91x | 2.07배 | 0.0811 |
| INT8/INT8-256 | 5.00 | 2.41 | 4.59 | 1.90x | 2.07배 | 0.0842 |

† 구 패키지: `--single` 모드. († 포함 전 패키지 동일 텍스트·시드, 1T/8T 출력 wav 동일 = 결정성 확인)

- 순위: **INT8/INT8-256 최속(2.41s)** > INT4/INT8-256 (2.51s) > INT4/FP32 (2.57s).
  INT8이 INT4보다 약간 빠름.
- 1024ctx는 8T에서도 실시간 미달(7.76s, 0.68x) — 실사용 비권장.
- 1T→8T 가속비 2.0~2.6배 (Python OV와 동급).

## 5. 백엔드 (GPU/NPU, INT4/INT8-256, mimi=CPU)

| 백엔드 | gen 평균(s) | 오디오(s) | 속도 | rms | 비고 |
|---|---|---|---|---|---|
| CPU 8T (대조) | 2.51 | 4.99 | 1.99x | 0.0746 | — |
| GPU.0 (iGPU) | 6.00 | 5.12 | 0.85x | 0.0744 | CPU 8T보다 느림 |
| GPU.1 (dGPU) | 3.18 | 4.96 | 1.56x | 0.0827 | iGPU의 1.9배, CPU 8T에는 못 미침 |
| GPU.2 | FAIL | — | — | — | phantom 장치 (기존 측정 `CL_INVALID_KERNEL_ARGS`) |
| NPU | FAIL(품질) | 5.68 | 0.26x | NaN | 합성은 끝나나(약 21s) 출력에 NaN 포함 — 실사용 불가 |

- mimi는 CPU 고정 (Python 벤치와 동일 구성).
- GPU마다 AR 경로(프레임 수)가 달라 오디오 길이가 다름 (HW 커널 수치차, 정상).

## 6. 분석

1. **모델**: torch 8T INT8 (0.78s) > IR-OV CPU 8T INT8 (2.41s) > candle (3.9s) ≈ torch 1T > ONNX hybrid.
   Rust만으로는 `synth-rs` INT8/INT8-256 8T가 최고속.
2. **백엔드**: CPU 8T가 iGPU보다 빠름. dGPU(GPU.1)는 CPU 점유를 낮춰야 할 때만 유효.
   NPU는 수치 붕괴(NaN), torch/ORT GPU는 EP·빌드 부재로 N/A.
3. **스레드**: OV 1T→8T 2.0~2.6배. candle은 스레드 효과 없음. ONNX-Rust는 1.1~1.4배로 낮음
   (candle sampler/mimi 병목).
4. **양자화**: OV 배포는 INT8/INT8-256이 최속·품질 정상. torch는 FP32→INT8-dyn 1.8배 향상.
5. **품질**: CPU/GPU 전 조건 rms 0.065~0.090 정상. NPU만 NaN으로 제외.
6. **주의**: 동일 텍스트도 AR 확률 경로(EOS 시점)가 시드·양자화·백엔드별로 달라
   오디오 길이가 3.9~5.7s로 변동 — speed_x는 참고용, gen 시간(RTF) 중심 비교 권장.

## 7. 권장 조합

- **최고속도 (연구/오프라인)**: 원본-torch CPU 8T INT8-dyn (0.78s, 5.64x)
- **Python 없이 최고속**: `synth-rs` INT8/INT8-256 CPU 8T (2.41s, 1.90x)
- **Python 없이 원본급**: `live_bench` candle CPU (3.9s, 실시간)
- **ONNX 필요 시**: Python hybrid INT8 8T (2.97s) — Rust onnx-bench (4.96s)보다 빠름
- **dGPU 보유 시**: `synth-rs` GPU.1 + mimi CPU (3.18s) — CPU 점유를 낮춰야 할 때
- **피해야 할 조합**: 1024ctx, GPU.0 단독, NPU, GPU.2, OV CPU 1T FP32

## 8. 산출물 및 재현

```bash
# 원본 (candle, Rust) — 1T/8T
./target/release/live_bench --variant korean --voice 2_0000.wav --threads 8 --outdir /tmp/lb_8T
# 원본 (torch, Python 대조군) — 1T/8T × FP32/INT8-dyn
python3 bench_orig_onnx.py          # 결과: /tmp/bench_py/bench_orig_onnx.json
# ONNX (Rust)
LD_LIBRARY_PATH=test-assets/onnx_rs/ort_libs ./target/release/onnx-bench \
  --onnx test-assets/pocket_step_uni256_int8.onnx --quant INT8-dyn --threads 8
# IR (Rust, 전 패키지; 구 2종은 --single)
./target/release/synth-rs --pkg test-assets/deploy_package_int8_int8_256 \
  --threads 8 --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav --seed 0
# IR GPU
./target/release/synth-rs --pkg test-assets/deploy_package_int4_int8_256 --single \
  --device GPU.1 --mimi-device CPU --text "..." --out out.wav --seed 0
```

- 측정 wav: `/tmp/lb_*` (candle), `/tmp/ox_*` (onnx-bench), `/tmp/synth_bench/` (IR 14종+GPU+NPU),
  `/tmp/bench_py/` (torch + Python ONNX hybrid)
- Rust 바이너리: `target/release/{live_bench, onnx-bench, synth-rs}`
- 관련 기존 자료: `BENCHMARK_REPORT_AI6.md` (Python 전수, 09-10),
  `BENCHMARK_REPORT_RS_bak.md` (Rust 전수, 09-11 — 오늘 재측정치와 ±5% 내 일치)
