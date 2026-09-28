/* AVX-512 VNNI W4A8 kernel: int4 weights (g128) x int8 per-token activations.
 * Compiled with -mavx512f -mavx512bw -mavx512vnni.
 *
 * vpdpbusd multiplies *unsigned* 8-bit by *signed* 8-bit bytes, sums groups
 * of four into int32 lanes. Our weight codes are already unsigned (q in
 * 0..15, value q - z), so they go in the u8 operand and the int8
 * activations in the s8 operand, with no bias trick on the activations:
 *
 *     sum_k (q_k - z) a_k  =  vpdpbusd(q, a)  -  z * sum_k a_k
 *
 * sum_k a_k per group is computed once per token when quantizing X and
 * the correction is seeded into lane 0 of the int32 accumulator, so the
 * group total (horizontal sum of the 16 lanes) is exact integer arithmetic.
 * Max |lane| = 8 * 15 * 128 + 15 * 128 * 128, far below 2^31. */
#include <immintrin.h>

#include "lowbit.h"

/* software prefetch distance for the packed weight stream (bytes ahead) */
#ifndef LB_PF
#define LB_PF 1024
#endif
#define PF(p) do { if (LB_PF) _mm_prefetch((const char *)(p) + LB_PF, _MM_HINT_T0); } while (0)

#define AI static inline __attribute__((always_inline))
#define UNROLL _Pragma("GCC unroll 16")

/* one group (128 elements) of one W row against one X row -> 16 int32 lanes
 * whose sum is exactly sum_k (q_k - z) * a_k */
AI __m512i group_vec(__m512i lo, __m512i hi, const int8_t *a, int32_t corr) {
    __m512i acc = _mm512_zextsi128_si512(_mm_cvtsi32_si128(corr));
    acc = _mm512_dpbusd_epi32(acc, lo, _mm512_loadu_si512((const void *)a));
    acc = _mm512_dpbusd_epi32(acc, hi, _mm512_loadu_si512((const void *)(a + 64)));
    return acc;
}

AI void unpack(const uint8_t *wp, __m512i *lo, __m512i *hi) {
    const __m512i m4 = _mm512_set1_epi8(15);
    const __m512i w = _mm512_loadu_si512((const void *)wp);
    *lo = _mm512_and_si512(w, m4);                        /* elements 0..63  */
    *hi = _mm512_and_si512(_mm512_srli_epi16(w, 4), m4);  /* elements 64..127 */
}

AI void w4a8_body(const int MR, const int NR, const int8_t *xq, int ldx,
                  const float *xs, const int32_t *asum, int lda,
                  const uint8_t *qw, int ldq, const float *s, const uint8_t *z,
                  int lds, int k0, int k1, float *y, int ldy) {
    __m512 facc[LB_MAX_MR][LB_MAX_NR];
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++)
        facc[i][j] = _mm512_setzero_ps();
    for (int g = k0 / LB_GROUP; g < k1 / LB_GROUP; g++) {
        __m512i lo[LB_MAX_NR], hi[LB_MAX_NR];
        __m512 sv[LB_MAX_NR];
        int32_t zz[LB_MAX_NR];
        UNROLL for (int j = 0; j < NR; j++) {
            PF(qw + (size_t)j * ldq + (size_t)g * 64);
            unpack(qw + (size_t)j * ldq + (size_t)g * 64, &lo[j], &hi[j]);
            sv[j] = _mm512_set1_ps(s[(size_t)j * lds + g]);
            zz[j] = z ? z[(size_t)j * lds + g] : 8;
        }
        UNROLL for (int i = 0; i < MR; i++) {
            const int8_t *a = xq + (size_t)i * ldx + (size_t)g * LB_GROUP;
            const int32_t as = asum[(size_t)i * lda + g];
            UNROLL for (int j = 0; j < NR; j++) {
                const __m512i acc = group_vec(lo[j], hi[j], a, -zz[j] * as);
                facc[i][j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), sv[j], facc[i][j]);
            }
        }
    }
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++)
        y[(size_t)i * ldy + j] += xs[i] * _mm512_reduce_add_ps(facc[i][j]);
}

#define DEF_TILE(MR, NR)                                                        \
    static void w4a8_##MR##x##NR(const int8_t *xq, int ldx, const float *xs,     \
                                 const int32_t *asum, int lda, const uint8_t *qw, \
                                 int ldq, const float *s, const uint8_t *z,       \
                                 int lds, int k0, int k1, float *y, int ldy) {    \
        w4a8_body(MR, NR, xq, ldx, xs, asum, lda, qw, ldq, s, z, lds, k0, k1, y,  \
                  ldy);                                                          \
    }
#include "tiles.inc"
#undef DEF_TILE

lb_w4a8_fn lb_w4a8_micro_vnni(int mr, int nr) {
#define DEF_TILE(MR, NR) if (mr == MR && nr == NR) return w4a8_##MR##x##NR;
#include "tiles.inc"
#undef DEF_TILE
    return 0;
}

void lb_w4a8_groupacc_vnni(const int8_t *xq, const int32_t *asum, int M, int Kp,
                           const uint8_t *qw, const uint8_t *z, int N,
                           int32_t *out) {
    const int G = Kp / LB_GROUP;
    for (int n = 0; n < N; n++)
        for (int g = 0; g < G; g++) {
            __m512i lo, hi;
            unpack(qw + (size_t)n * (Kp / 2) + (size_t)g * 64, &lo, &hi);
            const int32_t zz = z ? z[(size_t)n * G + g] : 8;
            for (int m = 0; m < M; m++)
                out[((size_t)m * N + n) * G + g] = _mm512_reduce_add_epi32(
                    group_vec(lo, hi, xq + (size_t)m * Kp + (size_t)g * LB_GROUP,
                              -zz * asum[(size_t)m * G + g]));
        }
}
