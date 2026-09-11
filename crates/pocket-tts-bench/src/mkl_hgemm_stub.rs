//! Link stub for MKL's half-precision GEMM (`hgemm_`).
//!
//! `candle-core` with the `mkl` feature unconditionally references the
//! Fortran `hgemm_` symbol, but the static MKL libraries packaged by
//! `intel-mkl-src 0.8.1` do not provide it, so final binaries fail to link
//! with `undefined symbol: hgemm_`.
//!
//! This stub satisfies the linker. It is never executed: the pocket-tts
//! models run entirely in FP32, and candle only calls `hgemm` for F16
//! matmuls. If it were ever reached, the process aborts instead of
//! silently computing garbage (release profile uses `panic = "abort"`).

/// MKL Fortran `hgemm_` stub. Zero parameters on purpose: the symbol only
/// needs to exist for the linker; the body never runs (see module docs).
#[unsafe(no_mangle)]
#[allow(dead_code)]
pub extern "C" fn hgemm_() {
    panic!("hgemm_ called: FP16 MKL GEMM is not available in this build");
}
