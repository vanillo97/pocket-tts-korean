# Pocket-TTS 한국어 음성 합성 벤치마크 자료 (Rust 실측, 2026-09-11 재측정)

모델(original/ONNX/IR) × 백엔드(CPU/NPU/GPU) × 양자화 × 스레드 전 매트릭스를
Rust 하네스(`live_bench`, `onnx-bench`, `synth-rs`)로 실측하고,
Python 대조군(원본-torch, ONNX-ORT hybrid)은 동일 조건으로 재측정함.

- 공통 조건: 텍스트 `"안녕하세요. 한국어 음성 합성 모델입니다."`, voice `2_0000.wav`,
  config `hf://seastar105/pocket-tts-korean-300m/korean.yaml`
- 공통 프로토콜: 모델 로드/컴파일 제외 e2e 타이밍 (text prefill + AR 루프 + sampler + mimi),
  warmup 1회(seed 999) + 실측 3회(seed 0,1,2)
- speed_x = 오디오길이 / gen, RTF = gen / 오디오길이 (RTF<1 = 실시간 이상)

> **재측정 변경점**: ① `live_bench` MKL 빌드 (`--features pocket-tts/mkl`,
> static MKL의 `hgemm_` 미제공 → `mkl_hgemm_stub.rs`로 링크 해결).
> ② `--threads`가 기존 OMP/MKL 변수만 설정해 실효가 없었던 문제를 수정 —
> `RAYON_NUM_THREADS`도 함께 설정 (candle gemm 백엔드가 참조).
> 구버전 리포트의 "candle 스레드 효과 없음" 결론은 측정 오류였음 (1T 표시로 24T 실행).
> ③ `onnx-bench`도 candle 병목 구간에 RAYON 스레드 적용.

## 1. 하드웨어 / 소프트웨어 환경 (실측 머신)

| 항목 | 값 |
|---|---|
| CPU | Intel Core Ultra 9 285 (Arrow Lake-S, 24코어), L3 36MiB |
| RAM | 61Gi, Ubuntu 26.04.1, kernel 7.0.0-30 |
| GPU | iGPU Arrow Lake-S (GPU.0) + Battlemage G31 dGPU ×2 (GPU.1/GPU.2, GPU.2는 phantom) |
| NPU | Intel Arrow Lake NPU (AI Boost, `/dev/accel0`) + DEEPX DX_M1 (OV 미노출) |
| OV 노출 장치 | `CPU, GPU.0, GPU.1, GPU.2, NPU` (openvino 2026.3.1) |
| torch | 2.13.0+cu130 (CUDA False → CPU-only로 동작, GPU 측정 불가) |
| onnxruntime | 1.29.0 (`CPUExecutionProvider`만, GPU EP 없음) |
| Rust | `target/release/{live_bench (MKL), onnx-bench, synth-rs}` |

## 2. 원본 모델 (original) — Rust candle(MKL) vs Python torch

| 모델 | 스레드 | 양자화 | gen 평균(s) | 오디오(s) | 속도 | RTF | QA rms |
|---|---|---|---|---|---|---|---|
| 원본-torch (Python) | 1T | FP32 | 3.234 | 4.32 | 1.34x | 0.749 | 0.0757 |
| 원본-torch | 8T | FP32 | 1.455 | 4.32 | 2.97x | 0.337 | 0.0757 |
| 원본-torch | 1T | INT8-dyn | 1.520 | 4.64 | 3.05x | 0.328 | 0.0794 |
| 원본-torch | 8T | INT8-dyn | 0.809 | 4.40 | 5.44x | 0.184 | 0.0792 |
| candle-rust MKL (`live_bench`) | 1T | FP32 | 3.687 | 4.43 | 1.20x | 0.833 | 0.0734 |
| candle-rust MKL (`live_bench`) | 8T | FP32 | 1.811 | 4.43 | 2.44x | 0.409 | 0.0736 |
| 원본-torch | GPU | FP32 | N/A | — | — | — | — (CPU-only 빌드) |

- candle MKL 빌드는 **1T→8T 2.04배** 스케일 (수정 전 리포트의 "스레드 효과 없음"은
  `--threads` 미반영 측정 오류). 8T 1.81s로 torch 8T FP32 (1.46s)에 근접.
- torch 8T INT8-dyn (0.81s, 5.44x)이 전체 최속 (변동 없음).

## 3. ONNX 모델 — Rust hybrid vs Python hybrid

| 모델 | 스레드 | 양자화 | gen 평균(s) | 오디오(s) | 속도 | RTF | QA rms |
|---|---|---|---|---|---|---|---|
| ONNX-ORT hybrid (Python) | 1T | FP32 | 5.907 | 4.19 | 0.71x | 1.411 | 0.0826 |
| ONNX-ORT hybrid | 8T | FP32 | 4.462 | 4.96 | 1.11x | 0.900 | 0.0883 |
| ONNX-ORT hybrid | 1T | INT8-dyn | 4.237 | 4.77 | 1.13x | 0.888 | 0.0896 |
| ONNX-ORT hybrid | 8T | INT8-dyn | 3.029 | 4.24 | 1.40x | 0.714 | 0.0843 |
| onnx-bench (Rust: ORT step + candle sampler/mimi) | 1T | FP32 | 8.496 | 4.59 | 0.54x | 1.852 | 0.0873 |
| onnx-bench (Rust) | 8T | FP32 | 6.141 | 4.59 | 0.75x | 1.339 | 0.0873 |
| onnx-bench (Rust) | 1T | INT8-dyn | 5.489 | 4.48 | 0.82x | 1.225 | 0.0847 |
| onnx-bench (Rust) | 8T | INT8-dyn | 5.031 | 4.48 | 0.89x | 1.123 | 0.0847 |

- RAYON 수정 후 Rust도 스레드 효과가 측정됨 (FP32 1.38배, INT8 1.09배) —
  여전히 낮음 (candle sampler/mimi가 병목, ORT step만 스케일).
- ONNX는 step만 ONNX이고 sampler/mimi가 torch/candle이라 전 모델 중 최하위권 (변동 없음).
- ORT에 GPU EP가 없어 GPU/NPU 측정 불가.

## 4. IR 모델 — Rust `synth-rs` (OpenVINO)

| 양자화 (step/sampler-ctx) | 1T gen(s) | 8T gen(s) | 오디오(s) | 8T 속도 | 1T→8T | rms |
|---|---|---|---|---|---|---|
| FP32/FP32-256 | 7.30 | 3.33 | 4.51 | 1.35x | 2.19배 | 0.0830 |
| FP32/INT8-256 † | 8.17 | 3.73 | 5.20 | 1.39x | 2.19배 | 0.0794 |
| INT4/FP32-256 | 5.35 | 2.56 | 5.07 | 1.98x | 2.09배 | 0.0724 |
| INT4/INT8-1024 | 19.9 | 7.88 | 5.25 | 0.67x | 2.53배 | 0.0770 |
| INT4/INT8-256 † | 5.14 | 2.50 | 4.99 | 2.00x | 2.06배 | 0.0746 |
| INT8/FP32-256 | 5.68 | 2.71 | 5.17 | 1.91x | 2.10배 | 0.0811 |
| INT8/INT8-256 | 5.00 | 2.42 | 4.59 | 1.90x | 2.07배 | 0.0842 |

† 구版 패키지: `--single` 모드. († 포함 전 패키지 동일 텍스트·시드, 1T/8T 출력 wav 동일 = 결정성 확인)

- 이전 측정과 ±3% 내 재현. 순위 유지: **INT8/INT8-256 최속(2.42s)**.
- 1024ctx는 8T에서도 실시간 미달(7.88s) — 실사용 비권장.

## 5. 백엔드 (GPU/NPU, INT4/INT8-256, mimi=CPU)

| 백엔드 | gen 평균(s) | 오디오(s) | 속도 | rms | 비고 |
|---|---|---|---|---|---|
| CPU 8T (대조) | 2.50 | 4.99 | 2.00x | 0.0746 | — |
| GPU.0 (iGPU) | 5.63 | 5.12 | 0.91x | 0.0744 | CPU 8T보다 느림 |
| GPU.1 (dGPU) | 3.25 | 4.96 | 1.53x | 0.0827 | iGPU의 1.7배, CPU 8T에는 못 미침 |
| GPU.2 | FAIL | — | — | — | phantom 장치 (기존 측정 `CL_INVALID_KERNEL_ARGS`) |
| NPU | FAIL(품질) | 5.68 | 0.27x | NaN | 합성은 끝나나(약 21s) 출력에 NaN 포함 — 실사용 불가 |

- 이전 측정과 동일 결론 (NPU NaN 재현됨).

## 6. 분석

1. **모델**: torch 8T INT8 (0.81s) > candle-MKL 8T (1.81s) > IR-OV CPU 8T INT8 (2.42s) >
   Python ONNX hybrid (3.03s) > Rust onnx-bench (5.03s).
   **MKL 적용으로 candle이 IR-OV보다 빨라짐** (수정 전과 순위 역전).
2. **백엔드**: CPU 8T가 iGPU보다 빠름. dGPU(GPU.1)는 CPU 점유를 낮춰야 할 때만 유효.
   NPU는 수치 붕괴(NaN), torch/ORT GPU는 EP·빌드 부재로 N/A.
3. **스레드**: candle-MKL 1T→8T **2.04배** (수정으로 정상 측정).
   OV 1T→8T 2.1~2.5배. ONNX-Rust는 1.1~1.4배로 낮음 (candle sampler/mimi 병목).
   MKL 24T는 8T보다 느림 (오버서브스크립션) — 8T 권장.
4. **양자화**: OV 배포는 INT8/INT8-256이 최속·품질 정상. torch는 FP32→INT8-dyn 1.9배 향상.
5. **품질**: CPU/GPU 전 조건 rms 0.065~0.090 정상. NPU만 NaN으로 제외.
6. **주의**: 동일 텍스트도 AR 확률 경로(EOS 시점)가 시드·양자화·백엔드별로 달라
   오디오 길이가 3.9~5.7s로 변동 — speed_x는 참고용, gen 시간(RTF) 중심 비교 권장.

## 7. 권장 조합

- **최고속도 (연구/오프라인)**: 원본-torch CPU 8T INT8-dyn (0.81s, 5.44x)
- **Python 없이 최고속**: `live_bench` candle-MKL CPU 8T (1.81s, 2.44x)
  — MKL 빌드 필요: `cargo build --release -p pocket-tts-bench --features pocket-tts/mkl`
- **배포 실전 (OV)**: `synth-rs` INT8/INT8-256 CPU 8T (2.42s, 1.90x)
- **ONNX 필요 시**: Python hybrid INT8 8T (3.03s) — Rust onnx-bench (5.03s)보다 빠름
- **dGPU 보유 시**: `synth-rs` GPU.1 + mimi CPU (3.25s) — CPU 점유를 낮춰야 할 때
- **피해야 할 조합**: 1024ctx, GPU.0 단독, NPU, GPU.2, OV CPU 1T FP32

## 8. 산출물 및 재현

```bash
# 원본 (candle MKL, Rust) — 1T/8T (--threads는 RAYON/MKL/OMP 모두 설정)
cargo build --release -p pocket-tts-bench --bin live_bench --features pocket-tts/mkl
./target/release/live_bench --variant korean --voice 2_0000.wav --threads 8 --outdir /tmp/lb_8T
# 원본 (torch, Python 대조군) — 1T/8T × FP32/INT8-dyn
python3 bench_orig_onnx.py          # 결과: /tmp/bench_py/bench_orig_onnx.json
# ONNX (Rust)
LD_LIBRARY_PATH=test-assets/onnx_rs/ort_libs ./target/release/onnx-bench \
  --onnx test-assets/pocket_step_uni256_int8.onnx --quant INT8-dyn --threads 8
# IR (Rust, 전 패키지; 구版 2종은 --single)
./target/release/synth-rs --pkg test-assets/deploy_package_int8_int8_256 \
  --threads 8 --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav --seed 0
# IR GPU
./target/release/synth-rs --pkg test-assets/deploy_package_int4_int8_256 --single \
  --device GPU.1 --mimi-device CPU --text "..." --out out.wav --seed 0
```

- 측정 wav: `/tmp/lb2_*` (candle-MKL), `/tmp/ox2_*` (onnx-bench), `/tmp/synth2/` (IR 14종+GPU+NPU),
  `/tmp/bench_py/` (torch + Python ONNX hybrid)
- Rust 바이너리: `target/release/{live_bench (MKL), onnx-bench, synth-rs}`
- 코드 변경: `live_bench.rs`/`onnx-bench.rs` RAYON 스레드 반영, `mkl_hgemm_stub.rs` 신규
  (static MKL의 `hgemm_` 미제공으로 링크 실패 → FP32 전용 stub으로 해결)
