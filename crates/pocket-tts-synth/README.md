# pocket-tts-synth — pocket-tts 배포 패키지의 Rust 버전

`test-assets/deploy_package_*/synth.py` 7종을 하나의 Rust 바이너리(`synth-rs`)로
포팅한 크레이트. 동일 CLI(`--pkg/--text/--out/--seed`), 동일 에셋(`models/*.xml`,
`assets/`), 동일 합성 알고리즘으로 동작합니다. torch·Python 불필요.

## 실행

```bash
# 신版 패키지 5종 (문장 분할+청킹)
cargo run --release -p pocket-tts-synth -- --pkg test-assets/deploy_package_fp32_fp32_256 \
  --text "안녕하세요." --out out.wav
# 구版 패키지 2종 (fp32_int8_256, int4_int8_256: 단일 텍스트 합성)
cargo run --release -p pocket-tts-synth -- --pkg test-assets/deploy_package_fp32_int8_256 \
  --text "안녕하세요." --out out.wav --single
```

`--pkg` 생략 시 워크스페이스(`target/release` 기준) 또는 현재 디렉토리에서
`deploy_package_fp32_fp32_256`을 자동 탐색합니다. 전 패키지 검증됨
(자세한 수치: 루트 `BENCHMARK_REPORT_RS.md`).

## 구현 메모

- 추론: `openvino` 크레이트 0.11 (`runtime-linking` — 빌드 시 OpenVINO
  링크 불필요, 실행 시 설치된 OpenVINO를 자동 탐색; conda/pip 설치 지원).
- 토크나이저: 워크스페이스의 `pocket-tts`가 내장한 pure-Rust SentencePiece
  로더를 재사용하므로 `tokenizer.model` 해석이 Python과 토큰 단위로 일치합니다.
- `.npz`/`.npy` 리더는 자작 미니 파서(`src/main.rs`의 `parse_npy`)를 사용합니다
  (C-order, f32/i64/f16 — 1024ctx 패키지의 f16 voice cache 포함).
- mimi 스트리밍 상태는 `synth.py`와 동일하게 매 스텝 출력(`n{j}`)으로 교체합니다
  (shape 포함, i64 off 대응).
- 입력 텐서 구조·상태 초기화는 IR에서 동적 탐색하므로 256ctx/1024ctx,
  FP32/INT8/INT4 모두 동일 코드로 동작합니다.

## 성능 메모

매 스텝 입력 텐서를 재할당하면 24층 KV 캐시 복사(~50MB/스텝, 1024ctx는
~200MB/스텝)가 병목이 됩니다. `Reusable` 버퍼(할당 1회 + `get_data_mut`
제자리 갱신, shape 변경 시에만 재생성)로 15~37% 단축했고, 그 결과
FP32/1024ctx는 Python版보다 근소하게 빨라졌습니다 (수치: 루트
`BENCHMARK_REPORT_RS.md`).

## Parity 상태 (검증됨)

- `s=0` 첫 AR 스텝의 transformer 출력(`out`)이 Python과 **비트 단위로 동일**합니다.
- 이후 프레임은 난수 생성기가 달라서 분기됩니다 (numpy PCG64 vs Rust ChaCha):
  동일 입력·동일 시드라도 프레임 수/음성이 1:1로 같지는 않습니다.
- 한 가지 의도적 일치: 신版 `synth.py`는 문장 결합 시 `sp.encode(" ")[:1] or []`를
  사용하는데, SentencePiece는 공백 전용 입력에 `[]`를 반환하므로 구분자가
  사실상 없습니다. Rust Unigram 포트는 `" "`에 실제 ID를 부여하므로,
  prefill 오염을 막기 위해 구분자 없이 바로 이어붙입니다.
- `SYNTH_DEBUG=1` 환경변수로 처음 25 스텝의 `tout_max`/`eos`를 출력합니다.

## 알려진 한계 (Python版과 동일)

- step ctx 256 + mimi 128 latents(약 10초) 초과 입력은 양쪽 모두
  `ScatterElementsUpdate` 오류로 동일하게 실패합니다 (패키지 자체 한도).
- `tools/enroll_ov.py`(화자 등록)는 포팅 범위 밖입니다.
