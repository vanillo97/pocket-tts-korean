# 벤치 실행 메뉴얼

`run_live_benchmark.py`의 Rust판 하네스 3종 실행법. 공통 프로토콜과 결과 해석은
`BENCHMARK_REPORT_RS.md` 참조.

공통 프로토콜(전 하네스 동일): 텍스트 `"안녕하세요. 한국어 음성 합성 모델입니다."`,
voice `2_0000.wav`, 로드/컴파일 제외 e2e 타이밍, warmup 1회(seed 999) + 실측 3회(seed 0,1,2).

## 0. 사전 준비

```bash
# 빌드 (web-ui 없이; web/dist 빌드 실패 회피)
cargo build --release -p pocket-tts-cli --no-default-features
cargo build --release -p pocket-tts-synth -p pocket-tts-bench -p pocket-tts-cli --no-default-features
```

- OpenVINO: conda/pip 설치 자동 탐색 (추가 환경변수 불필요).
- ONNX: 실행 시 `LD_LIBRARY_PATH=test-assets/onnx_rs/ort_libs` 필요 (ORT 1.29).
- 모델 가중치: 한국어 candle 모델은 HF 공개라 토큰 불필요 (初回 다운로드 후 캐시).
  영어 `b6369a24`는 gated라 `HF_TOKEN` 필요.
- 디스크: `target/` 약 11GB 사용. `--release`로만 빌드할 것 (`target/debug` 금지).

## 1. live_bench (candle e2e, 원본 모델)

```bash
cargo run --release -p pocket-tts-bench --bin live_bench -- \
  --variant korean --voice 2_0000.wav --outdir live_bench_wav
```

옵션: `--config <로컬경로|hf://...>` (variant 대신), `--text`, `--runs` (기본 3),
`--threads <N>` (MKL/OMP 스레드, 기본 auto), `--quant` 없음(FP32 고정).

출력: `{outdir}/candle_cpu<T>_FP32.wav` (첫 실행분) + `live_benchmark_result_rs.json`
(gen_mean/std/min/max, audio_mean_s, speed_x, rtf, qa_peak/rms/cent).

특징: `--threads`와 무관하게 결정적 (시드 고정 시 wav sha256 일치).
candle은 1T/8T가 거의 동일하다 (스레드 확장 없음).

## 2. synth-rs (IR 배포 패키지 7종)

```bash
./target/release/synth-rs --pkg test-assets/deploy_package_int8_int8_256 \
  --threads 8 --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav --seed 0
```

옵션:

| 옵션 | 설명 |
|---|---|
| `--pkg` | 패키지 경로 (생략 시 fp32_fp32_256 자동 탐색) |
| `--device` | `CPU`(기본), `GPU`, `NPU`, `GPU.0`, `GPU.1`, … |
| `--threads` | `INFERENCE_NUM_THREADS` (생략 시 장치 기본값) |
| `--mimi-device` | mimi 전용 장치 (기본 `--device`와 동일) |
| `--single` | 구版 2종(`fp32_int8_256`, `int4_int8_256`)용 단일 텍스트 모드 |
| `--seed` | 난수 시드 (기본 0) |

패키지별 권장 호출:

```bash
# 신版 5종 (문장 분할+청킹 내장)
for p in fp32_fp32_256 int4_fp32_256 int4_int8_1024 int8_fp32_256 int8_int8_256; do
  ./target/release/synth-rs --pkg test-assets/deploy_package_$p --threads 8 \
    --text "안녕하세요. 한국어 음성 합성 모델입니다." --out ${p}.wav --seed 0
done
# 구版 2종 (--single 필수)
for p in fp32_int8_256 int4_int8_256; do
  ./target/release/synth-rs --pkg test-assets/deploy_package_$p --threads 8 --single \
    --text "안녕하세요. 한국어 음성 합성 모델입니다." --out ${p}.wav --seed 0
done
# GPU (mimi는 CPU 권장 — 전부 GPU 시 rms 저하)
./target/release/synth-rs --pkg test-assets/deploy_package_int4_int8_256 --single \
  --device GPU.1 --mimi-device CPU --text "..." --out out.wav
```

출력 줄: `OV 로드: Xs` (컴파일, gen에서 제외), `청크/AR: N frames`, `저장: ... (오디오, 속도)`,
`peak=...`. gen 시간 = 전체 wall − `OV 로드` (리포트 §3.4 정의와 동일).

## 3. onnx-bench (ONNX step + candle sampler/mimi)

```bash
export LD_LIBRARY_PATH=test-assets/onnx_rs/ort_libs
cargo run --release -p pocket-tts-bench --bin onnx-bench -- --onnx test-assets/pocket_step_uni256_int8.onnx \
  --quant INT8-dyn --threads 8
```

옵션: `--onnx` (기본 `test-assets/pocket_step_uni256.onnx`), `--quant` (리포트 라벨),
`--variant` (candle sampler/mimi용, 기본 korean), `--voice-state`
(기본 `test-assets/onnx_rs/assets/vs.npz`, torch 덤프), `--text/--outdir/--runs/--threads`.
출력: `{outdir}/onnxrs_cpu<T>_<quant>.wav` + `bench_onnx_rs.json`.

## 4. Python 대조군 (원본-torch)

```bash
python3 -c "import bench_orig_onnx as b; b.run_torch_bench()"
```

`pocket-tts==3.1.0`, torch CPU, `test-assets` 경로 수정版. 결과는
`BENCHMARK_REPORT_AI6.md` §3.1과 ±3% 이내로 재현되어야 한다.

## 5. 전체 매트릭스 재현 (예시)

```bash
TEXT="안녕하세요. 한국어 음성 합성 모델입니다."
# candle 1T/8T
for th in 1 8; do ./target/release/live_bench --variant korean \
  --voice 2_0000.wav --outdir /tmp/lb_${th}t --threads $th; done
# OV 7종 × 1T/8T × seed 0,1,2 (위 §2 루프에 seed 루프 추가)
# ONNX FP32/INT8 × 1T/8T
export LD_LIBRARY_PATH=test-assets/onnx_rs/ort_libs
for spec in "pocket_step_uni256.onnx:FP32" "pocket_step_uni256_int8.onnx:INT8-dyn"; do
  f="${spec%%:*}"; q="${spec##*:}"
  for th in 1 8; do ./target/release/onnx-bench --onnx test-assets/$f \
    --quant $q --threads $th --outdir /tmp/ox_${q}_${th}t; done
done
```

## 6. 결과 해석 주의

- 오디오 길이는 시드·양자화·장치별 AR 확률 경로(EOS 시점)로 4~5s 변동한다 (정상).
  비교 중심은 gen 시간(RTF), speed_x는 참고용.
- 10초 초과 입력: step ctx 256 + mimi 128 latents 한도로 양쪽(Rust/Python) 모두
  `ScatterElementsUpdate` 오류 (패키지 자체 한도, 버그 아님).
- 1024ctx 패키지는 양쪽 모두 2~3배 느리다 (실사용 비권장).
- GPU는 장치별 수치차로 프레임 수가 달라진다 (HW 커널 차이, 정상).
- NPU 전체합성은 장치 hung으로 실패한다 (AI6에서도 미지원).

## 7. 문제 해결

| 증상 | 원인/대처 |
|---|---|
| `VERS_1.29.0 not found` (onnx-bench) | `LD_LIBRARY_PATH=test-assets/onnx_rs/ort_libs` 미설정 |
| `No space left on device` 빌드 | `target/debug` 삭제, `--release`만 사용 |
| `npz missing` / dtype 오류 | 패키지 `assets/` 확인 (1024ctx는 f16, 자동 변환됨) |
| 첫 실행만 느림 | OS 페이지 캐시/원타임 컴파일 캐시, warmup으로 분리 측정됨 |
