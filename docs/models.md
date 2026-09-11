# 모델 실행 메뉴얼

저장소에서 실행 가능한 모델별 사용법. 성능 수치는 `BENCHMARK_REPORT_RS.md` 참조.

## 1. 모델 목록

| 모델 | 소스 | 실행 수단 | 양자화/컨텍스트 |
|---|---|---|---|
| 한국어 300M (24L teacher) | `seastar105/pocket-tts-korean-300m` | candle `generate`/`live_bench`, torch | FP32 |
| 영어 student (6L) | `kyutai/pocket-tts` (gated) | candle `generate --variant b6369a24` | FP32 |
| OV IR 7종 | `test-assets/deploy_package_*` | `synth-rs` | FP32/INT8/INT4, 256/1024ctx |
| ONNX step 2종 | `test-assets/pocket_step_uni256*.onnx` | `onnx-bench`, torch hybrid | FP32/INT8-dyn |
| torch 원본 | `pocket-tts==3.1.0` pip | `bench_orig_onnx.py`, 직접 코드 | FP32/INT8-dyn |

## 2. Candle 실행 (Rust 네이티브)

```bash
# 한국어 (HF 공개, 토큰 불필요)
cargo run --release -p pocket-tts-cli --no-default-features -- generate \
  --variant korean --voice 2_0000.wav \
  --text "안녕하세요. 한국어 음성 합성 테스트입니다." --output out.wav
# 임의 HF 설정
cargo run --release -p pocket-tts-cli --no-default-features -- generate \
  --config hf://seastar105/pocket-tts-korean-300m/korean.yaml \
  --voice my_voice.wav --text "반갑습니다." --output out.wav
# 영어 (gated — HF_TOKEN 필요, 이용약관 동의 필수)
export HF_TOKEN="hf_..."
cargo run --release -p pocket-tts-cli --no-default-features -- generate \
  --text "Hello, world!"
```

- `--temperature` 생략 시 config 권장값 (한국어 0.3).
- `--voice`: 사전 정의명(`alba` 등, 영어만) 또는 `.wav` (자동 리샘플, 10초 이내 권장,
  본인 동의 음성만).
- 서버 모드: `cargo run --release -p pocket-tts-cli --no-default-features -- serve`
  (자세히는 `docs/serve.md`). `--config/--variant`로 모델 선택.

## 3. OV 배포 패키지 실행 (7종)

```bash
# 최속 조합 (INT8/INT8, 8T)
./target/release/synth-rs --pkg test-assets/deploy_package_int8_int8_256 --threads 8 \
  --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav --seed 0
# 구版 2종은 --single 필수
./target/release/synth-rs --pkg test-assets/deploy_package_fp32_int8_256 --threads 8 \
  --single --text "안녕하세요." --out out.wav
# GPU (mimi는 CPU 권장)
./target/release/synth-rs --pkg test-assets/deploy_package_int4_int8_256 --single \
  --device GPU.1 --mimi-device CPU --text "안녕하세요." --out out.wav
```

양자화 선택 가이드: 속도 INT8/INT8 > INT4/INT8 ≈ INT8/FP32 > FP32 (2.6~3.6s @8T),
품질 차이 없음(rms 0.077~0.087). 1024ctx는 8초대로 비권장.
Python版 동일 동작: `python test-assets/deploy_package_X/synth.py --text ... --out ... --seed 0`
(필요 패키지: openvino, numpy, scipy, sentencepiece).

## 4. ONNX 실행

```bash
export LD_LIBRARY_PATH=test-assets/onnx_rs/ort_libs
./target/release/onnx-bench --onnx test-assets/pocket_step_uni256_int8.onnx \
  --quant INT8-dyn --threads 8 --text "안녕하세요." --outdir out_onnx
```

- step만 ONNX, sampler/Mimi는 candle (하이브리드). voice state는 torch 덤프
  (`test-assets/onnx_rs/assets/vs.npz`) 사용.
- torch 하이브리드 직접 실행은 `bench_orig_onnx.py` §3 코드 참조
  (필요: `pocket-tts==3.1.0`, torch, onnxruntime).

## 5. Torch 원본 실행

```bash
pip install "pocket-tts==3.1.0" torch scipy  # CPU
```

```python
from pocket_tts import TTSModel
model = TTSModel.load_model(config="hf://seastar105/pocket-tts-korean-300m/korean.yaml")
vs = model.get_state_for_audio_prompt("./2_0000.wav")  # 동의 음성만
audio = model.generate_audio(vs, "안녕하세요. 한국어 음성 합성 모델입니다.")
import scipy.io.wavfile
scipy.io.wavfile.write("out.wav", model.sample_rate, audio.detach().cpu().numpy())
```

## 6. 음성 변경 (화자 등록)

- candle: `--voice 새화자.wav` 로 바로 변경 (매 실행 인코딩).
- OV 패키지: `assets/voice_cache.npz` 고정. 변경 시
  `python test-assets/deploy_package_X/tools/enroll_ov.py --voice 새화자.wav --pkg ...`
  (torch 불필요).

## 7. 문제 해결

| 증상 | 원인/대처 |
|---|---|
| `status code 401` (가중치) | `export HF_TOKEN` 오타 확인 (`HF_TOKER` 아님), 토큰 유효성 |
| `status code 403` (가중치) | HF 모델 페이지 이용약관 동의 + gated 읽기 권한 토큰 |
| `Web UI assets not found` 빌드 | `--no-default-features` 추가 또는 `crates/pocket-tts-cli/web`에서 `npm run build` |
| `ScatterElementsUpdate` 합성 실패 | 10초 초과 입력 (ctx 256 / mimi 128 latents 한도), 텍스트를 나누세요 |
| mimi-GPU 음성 작음 (rms 0.03) | `--mimi-device CPU` 사용 |
| NPU `DEVICE_LOST` | 전체합성 미지원, CPU/GPU 사용 |
| `VERS_1.29.0 not found` | `LD_LIBRARY_PATH=test-assets/onnx_rs/ort_libs` 설정 |
| 빌드 디스크 부족 | `target/debug` 삭제, `--release`만 사용 (약 11GB 필요) |
