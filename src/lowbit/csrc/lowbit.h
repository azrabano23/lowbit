/*
 * lowbit: fused dequantize + GEMM kernels for low-bit LLM weight formats.
 *
 * All GEMMs compute  Y[M,N] = X[M,K] . W[N,K]^T   (nn.Linear convention:
 * W has one row per output feature).  K is always the *padded* reduction
 * length Kp (a multiple of the format's block size); the Python side pads
 * X with zeros so padded weight codes never contribute.
 *
 * Packed weight layouts (row n of W occupies Kp/2 contiguous bytes):
 *
 *   int4, group 128 : group g = bytes [64g, 64g+64). Byte j holds element
 *                     128g+j in its low nibble and 128g+64+j in its high
 *                     nibble. Codes are unsigned 0..15; value = (q - z) * s
 *                     with z = zeros[n,g] (or 8 when zeros == NULL) and
 *                     s = scales[n,g] (fp32).
 *   MXFP4, block 32 : block b = bytes [16b, 16b+16). Byte j holds element
 *                     32b+j (low nibble) and 32b+16+j (high nibble). Codes
 *                     are OCP E2M1 (sign | 2-bit exp | 1-bit mantissa);
 *                     value = e2m1(q) * 2^(e8m0[n,b] - 127).
 */
#ifndef LOWBIT_H
#define LOWBIT_H

#include <stddef.h>
#include <stdint.h>

#define LB_ISA_SCALAR 0
#define LB_ISA_AVX2 1
#define LB_ISA_AVX512 2

#define LB_GROUP 128   /* int4 group size (elements) */
#define LB_MXBLOCK 32  /* MXFP4 block size (elements) */
#define LB_MAX_MR 8
#define LB_MAX_NR 4

/* Tuning / dispatch parameters. Zero or negative fields mean "default". */
typedef struct {
    int isa;      /* -1 auto, else LB_ISA_*; clamped to what the CPU supports */
    int mr;       /* rows of X per micro-kernel (1,2,4,8) */
    int nr;       /* rows of W per micro-kernel (1,2,4)  ("unroll") */
    int nb;       /* N-tile per thread task (rows of W) */
    int kb;       /* K-block in elements (multiple of 128), 0 = whole K */
    int order;    /* 0: m-blocks outer inside a tile, 1: n outer */
    int nthreads; /* 0 = all available */
} lb_params;

/* micro-kernel signatures: accumulate (+=) into y[i*ldy + j], i<MR, j<NR,
 * over the K range [k0, k1) (both multiples of 128). Pointers are
 * pre-offset to the first row of the tile. */
typedef void (*lb_w4a16_fn)(const float *x, int ldx, const uint8_t *qw, int ldq,
                            const float *s, const uint8_t *z, int lds, int k0,
                            int k1, float *y, int ldy);
typedef void (*lb_mxfp4_fn)(const float *x, int ldx, const uint8_t *qw, int ldq,
                            const uint8_t *e, int lde, int k0, int k1, float *y,
                            int ldy);
typedef void (*lb_w4a8_fn)(const int8_t *xq, int ldx, const float *xs,
                           const int32_t *asum, int lda, const uint8_t *qw,
                           int ldq, const float *s, const uint8_t *z, int lds,
                           int k0, int k1, float *y, int ldy);

/* per-ISA micro-kernel tables (return NULL for unsupported tiles) */
lb_w4a16_fn lb_w4a16_micro_scalar(int mr, int nr);
lb_mxfp4_fn lb_mxfp4_micro_scalar(int mr, int nr);
lb_w4a8_fn lb_w4a8_micro_scalar(int mr, int nr);
void lb_w4a8_groupacc_scalar(const int8_t *xq, const int32_t *asum, int M, int Kp,
                             const uint8_t *qw, const uint8_t *z, int N,
                             int32_t *out);
#ifdef LB_X86
lb_w4a16_fn lb_w4a16_micro_avx2(int mr, int nr);
lb_mxfp4_fn lb_mxfp4_micro_avx2(int mr, int nr);
lb_w4a8_fn lb_w4a8_micro_avx2(int mr, int nr);
void lb_w4a8_groupacc_avx2(const int8_t *xq, const int32_t *asum, int M, int Kp,
                           const uint8_t *qw, const uint8_t *z, int N,
                           int32_t *out);
lb_w4a16_fn lb_w4a16_micro_avx512(int mr, int nr);
lb_mxfp4_fn lb_mxfp4_micro_avx512(int mr, int nr);
lb_w4a8_fn lb_w4a8_micro_vnni(int mr, int nr);
void lb_w4a8_groupacc_vnni(const int8_t *xq, const int32_t *asum, int M, int Kp,
                           const uint8_t *qw, const uint8_t *z, int N,
                           int32_t *out);
#endif

/* E8M0 -> fp32 (0xFF is NaN per the OCP MX spec; 0x00 is 2^-127). */
static inline float lb_e8m0_to_f32(uint8_t e) {
    union { uint32_t u; float f; } v;
    v.u = e == 0 ? 0x00400000u : (e == 255 ? 0x7FC00000u : (uint32_t)e << 23);
    return v.f;
}

/* 256-entry E8M0 -> fp32 table (filled by a library constructor); a table
 * lookup is two loads and no ALU work in the inner loop. */
extern float lb_e8m0_lut[256];

/* ---- public API (exported from the shared library) ---- */
int lb_version(void);
int lb_cpu_isa(void);       /* best ISA the hardware supports */
int lb_active_isa(void);    /* after applying the LOWBIT_ISA env cap */
int lb_has_vnni(void);
int lb_has_openmp(void);
int lb_max_threads(void);
/* effective ISA a kernel ("w4a16","mxfp4","w4a8" = 0,1,2) will run with */
int lb_effective_isa(int kind, int requested);

int lb_gemm_w4a16(const float *x, int M, int Kp, const uint8_t *qw,
                  const float *scales, const uint8_t *zeros, int N, float *y,
                  const lb_params *p);
int lb_gemm_mxfp4(const float *x, int M, int Kp, const uint8_t *qw,
                  const uint8_t *e8m0, int N, float *y, const lb_params *p);
/* x is fp32; per-token int8 quantization happens inside (and is timed). */
int lb_gemm_w4a8(const float *x, int M, int Kp, const uint8_t *qw,
                 const float *scales, const uint8_t *zeros, int N, float *y,
                 const lb_params *p);
/* same, with activations already quantized (xq [M,Kp], xs [M], asum [M,G]) */
int lb_gemm_w4a8_q(const int8_t *xq, const float *xs, const int32_t *asum,
                   int M, int Kp, const uint8_t *qw, const float *scales,
                   const uint8_t *zeros, int N, float *y, const lb_params *p);
/* per-token symmetric int8 quantization; asum = per-group sum of codes */
void lb_quant_act_int8(const float *x, int M, int Kp, int8_t *xq, float *xs,
                       int32_t *asum);
/* exact integer accumulators acc[m,n,g] = sum_k (q - z) * xq  (debug/test) */
int lb_w4a8_group_acc(const int8_t *xq, const int32_t *asum, int M, int Kp,
                      const uint8_t *qw, const uint8_t *zeros, int N,
                      int32_t *out, int isa);

#endif
