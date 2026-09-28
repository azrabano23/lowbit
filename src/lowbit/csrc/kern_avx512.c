/* AVX-512F/BW kernels: W4A16 (int4 g128, fp32 act) and MXFP4 (fp32 act).
 * Compiled with -mavx512f -mavx512bw -mfma; only called after runtime check.
 *
 * Dequantization happens in registers, never in memory. Both formats have
 * only 16 possible values per group/block, so we build a 16-entry fp32
 * table per (row, group) once and decode with a single vpermps per 16
 * elements:
 *   int4 : table[q] = (q - z) * s                 (one sub+mul per group)
 *   MXFP4: table[q] = e2m1(q) * 2^(e-127)         (one mul per block; exact)
 *   16 packed bytes -> vpmovzxbd (16 dwords, 2 nibbles each) ->
 *   lo = vpermps(v, table)   (vpermps reads only the low 4 index bits)
 *   hi = vpermps(v >> 4, table)
 * Each (row of X, row of W) pair keeps two accumulators (lo/hi halves)
 * so the FMA dependency chain is not the bottleneck at MR = NR = 1. */
#include <immintrin.h>

#include "lowbit.h"

/* software prefetch distance for the packed weight stream (bytes ahead) */
#ifndef LB_PF
#define LB_PF 1024
#endif
#define PF(p) do { if (LB_PF) _mm_prefetch((const char *)(p) + LB_PF, _MM_HINT_T0); } while (0)

#define AI static inline __attribute__((always_inline))
#define UNROLL _Pragma("GCC unroll 16")

AI void w4a16_body(const int MR, const int NR, const float *x, int ldx,
                   const uint8_t *qw, int ldq, const float *s, const uint8_t *z,
                   int lds, int k0, int k1, float *y, int ldy) {
    __m512 accl[LB_MAX_MR][LB_MAX_NR], acch[LB_MAX_MR][LB_MAX_NR];
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++) {
        accl[i][j] = _mm512_setzero_ps();
        acch[i][j] = _mm512_setzero_ps();
    }
    const __m512 iota = _mm512_setr_ps(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15);
    for (int g = k0 / LB_GROUP; g < k1 / LB_GROUP; g++) {
        /* per-group 16-entry table lut[q] = (q - z) * s: the whole
         * dequantization of an element is then one vpermps */
        __m512 lut[LB_MAX_NR];
        const uint8_t *wp[LB_MAX_NR];
        UNROLL for (int j = 0; j < NR; j++) {
            const float sc = s[(size_t)j * lds + g];
            const float zz = z ? (float)z[(size_t)j * lds + g] : 8.0f;
            lut[j] = _mm512_mul_ps(_mm512_sub_ps(iota, _mm512_set1_ps(zz)),
                                   _mm512_set1_ps(sc));
            wp[j] = qw + (size_t)j * ldq + (size_t)g * 64;
            PF(wp[j]);
        }
        const float *xg = x + (size_t)g * LB_GROUP;
        UNROLL for (int c = 0; c < 4; c++) {
            __m512 wl[LB_MAX_NR], wh[LB_MAX_NR];
            UNROLL for (int j = 0; j < NR; j++) {
                const __m512i v = _mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i *)(wp[j] + c * 16)));
                /* vpermps reads only the low 4 index bits: no mask needed */
                wl[j] = _mm512_permutexvar_ps(v, lut[j]);
                wh[j] = _mm512_permutexvar_ps(_mm512_srli_epi32(v, 4), lut[j]);
            }
            UNROLL for (int i = 0; i < MR; i++) {
                const __m512 xl = _mm512_loadu_ps(xg + (size_t)i * ldx + c * 16);
                const __m512 xh = _mm512_loadu_ps(xg + (size_t)i * ldx + 64 + c * 16);
                UNROLL for (int j = 0; j < NR; j++) {
                    accl[i][j] = _mm512_fmadd_ps(wl[j], xl, accl[i][j]);
                    acch[i][j] = _mm512_fmadd_ps(wh[j], xh, acch[i][j]);
                }
            }
        }
    }
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++)
        y[(size_t)i * ldy + j] +=
            _mm512_reduce_add_ps(_mm512_add_ps(accl[i][j], acch[i][j]));
}

AI void mxfp4_body(const int MR, const int NR, const float *x, int ldx,
                   const uint8_t *qw, int ldq, const uint8_t *e, int lde, int k0,
                   int k1, float *y, int ldy) {
    const __m512 lut = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f,
                                      6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f,
                                      -3.0f, -4.0f, -6.0f);
    __m512 accl[LB_MAX_MR][LB_MAX_NR], acch[LB_MAX_MR][LB_MAX_NR];
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++) {
        accl[i][j] = _mm512_setzero_ps();
        acch[i][j] = _mm512_setzero_ps();
    }
    for (int b = k0 / LB_MXBLOCK; b < k1 / LB_MXBLOCK; b++) {
        __m512 wl[LB_MAX_NR], wh[LB_MAX_NR];
        if ((b & 3) == 0)
            UNROLL for (int j = 0; j < NR; j++) PF(qw + (size_t)j * ldq + (size_t)b * 16);
        UNROLL for (int j = 0; j < NR; j++) {
            /* fold the power-of-two block scale into the table (exact) */
            const __m512 tab = _mm512_mul_ps(
                lut, _mm512_set1_ps(lb_e8m0_lut[e[(size_t)j * lde + b]]));
            const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                (const __m128i *)(qw + (size_t)j * ldq + (size_t)b * 16)));
            wl[j] = _mm512_permutexvar_ps(v, tab);
            wh[j] = _mm512_permutexvar_ps(_mm512_srli_epi32(v, 4), tab);
        }
        const float *xb = x + (size_t)b * LB_MXBLOCK;
        UNROLL for (int i = 0; i < MR; i++) {
            const __m512 xl = _mm512_loadu_ps(xb + (size_t)i * ldx);
            const __m512 xh = _mm512_loadu_ps(xb + (size_t)i * ldx + 16);
            UNROLL for (int j = 0; j < NR; j++) {
                accl[i][j] = _mm512_fmadd_ps(wl[j], xl, accl[i][j]);
                acch[i][j] = _mm512_fmadd_ps(wh[j], xh, acch[i][j]);
            }
        }
    }
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++)
        y[(size_t)i * ldy + j] +=
            _mm512_reduce_add_ps(_mm512_add_ps(accl[i][j], acch[i][j]));
}

#define DEF_TILE(MR, NR)                                                       \
    static void w4a16_##MR##x##NR(const float *x, int ldx, const uint8_t *qw,   \
                                  int ldq, const float *s, const uint8_t *z,    \
                                  int lds, int k0, int k1, float *y, int ldy) { \
        w4a16_body(MR, NR, x, ldx, qw, ldq, s, z, lds, k0, k1, y, ldy);         \
    }                                                                           \
    static void mxfp4_##MR##x##NR(const float *x, int ldx, const uint8_t *qw,   \
                                  int ldq, const uint8_t *e, int lde, int k0,   \
                                  int k1, float *y, int ldy) {                  \
        mxfp4_body(MR, NR, x, ldx, qw, ldq, e, lde, k0, k1, y, ldy);            \
    }
#include "tiles.inc"
#undef DEF_TILE

lb_w4a16_fn lb_w4a16_micro_avx512(int mr, int nr) {
#define DEF_TILE(MR, NR) if (mr == MR && nr == NR) return w4a16_##MR##x##NR;
#include "tiles.inc"
#undef DEF_TILE
    return 0;
}

lb_mxfp4_fn lb_mxfp4_micro_avx512(int mr, int nr) {
#define DEF_TILE(MR, NR) if (mr == MR && nr == NR) return mxfp4_##MR##x##NR;
#include "tiles.inc"
#undef DEF_TILE
    return 0;
}
