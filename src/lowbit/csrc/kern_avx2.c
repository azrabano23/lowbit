/* AVX2 + FMA kernels (the fallback for CPUs without AVX-512, e.g. many CI
 * runners). Same algorithms as kern_avx512.c / kern_vnni.c at 8 lanes.
 *
 * W4A8 without VNNI uses vpmaddubsw (u8 x s8 -> s16 pair sums) + vpmaddwd.
 * vpmaddubsw saturates at int16, but with 4-bit unsigned codes one pair is
 * at most 2 * 15 * 128 = 3840 in magnitude and we add four such vectors
 * before widening: 4 * 3840 = 15360 < 32767, so the result is exact. */
#include <immintrin.h>

#include "lowbit.h"

/* software prefetch distance for the packed weight stream (bytes ahead) */
#ifndef LB_PF
#define LB_PF 1024
#endif
#define PF(p) do { if (LB_PF) _mm_prefetch((const char *)(p) + LB_PF, _MM_HINT_T0); } while (0)

#define AI static inline __attribute__((always_inline))
#define UNROLL _Pragma("GCC unroll 16")

AI float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

AI int32_t hsum8i(__m256i v) {
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4E));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xB1));
    return _mm_cvtsi128_si32(a);
}

AI __m256i load8_u8_to_i32(const uint8_t *p) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
}

AI void w4a16_body(const int MR, const int NR, const float *x, int ldx,
                   const uint8_t *qw, int ldq, const float *s, const uint8_t *z,
                   int lds, int k0, int k1, float *y, int ldy) {
    __m256 accl[LB_MAX_MR][LB_MAX_NR], acch[LB_MAX_MR][LB_MAX_NR];
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++) {
        accl[i][j] = _mm256_setzero_ps();
        acch[i][j] = _mm256_setzero_ps();
    }
    const __m256i m4 = _mm256_set1_epi32(15);
    for (int g = k0 / LB_GROUP; g < k1 / LB_GROUP; g++) {
        __m256 sv[LB_MAX_NR], bv[LB_MAX_NR];
        const uint8_t *wp[LB_MAX_NR];
        UNROLL for (int j = 0; j < NR; j++) {
            const float sc = s[(size_t)j * lds + g];
            const float zz = z ? (float)z[(size_t)j * lds + g] : 8.0f;
            sv[j] = _mm256_set1_ps(sc);
            bv[j] = _mm256_set1_ps(-zz * sc);
            wp[j] = qw + (size_t)j * ldq + (size_t)g * 64;
            PF(wp[j]);
        }
        const float *xg = x + (size_t)g * LB_GROUP;
        UNROLL for (int c = 0; c < 8; c++) {
            __m256 wl[LB_MAX_NR], wh[LB_MAX_NR];
            UNROLL for (int j = 0; j < NR; j++) {
                const __m256i v = load8_u8_to_i32(wp[j] + c * 8);
                wl[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m4)),
                                        sv[j], bv[j]);
                wh[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(v, 4)),
                                        sv[j], bv[j]);
            }
            UNROLL for (int i = 0; i < MR; i++) {
                const __m256 xl = _mm256_loadu_ps(xg + (size_t)i * ldx + c * 8);
                const __m256 xh = _mm256_loadu_ps(xg + (size_t)i * ldx + 64 + c * 8);
                UNROLL for (int j = 0; j < NR; j++) {
                    accl[i][j] = _mm256_fmadd_ps(wl[j], xl, accl[i][j]);
                    acch[i][j] = _mm256_fmadd_ps(wh[j], xh, acch[i][j]);
                }
            }
        }
    }
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++)
        y[(size_t)i * ldy + j] += hsum8(_mm256_add_ps(accl[i][j], acch[i][j]));
}

AI void mxfp4_body(const int MR, const int NR, const float *x, int ldx,
                   const uint8_t *qw, int ldq, const uint8_t *e, int lde, int k0,
                   int k1, float *y, int ldy) {
    /* vpermps only has 8 entries: decode the magnitude from the low 3 bits
     * with a per-block table |e2m1| * 2^(e-127), then xor the sign (bit 3)
     * into the float sign bit. */
    const __m256 lut = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f);
    const __m256i sgn = _mm256_set1_epi32((int)0x80000000u);
    __m256 accl[LB_MAX_MR][LB_MAX_NR], acch[LB_MAX_MR][LB_MAX_NR];
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++) {
        accl[i][j] = _mm256_setzero_ps();
        acch[i][j] = _mm256_setzero_ps();
    }
    for (int b = k0 / LB_MXBLOCK; b < k1 / LB_MXBLOCK; b++) {
        __m256 sc[LB_MAX_NR];
        if ((b & 3) == 0)
            UNROLL for (int j = 0; j < NR; j++) PF(qw + (size_t)j * ldq + (size_t)b * 16);
        UNROLL for (int j = 0; j < NR; j++)
            sc[j] = _mm256_mul_ps(lut, _mm256_set1_ps(lb_e8m0_lut[e[(size_t)j * lde + b]]));
        const float *xb = x + (size_t)b * LB_MXBLOCK;
        UNROLL for (int c = 0; c < 2; c++) {
            __m256 wl[LB_MAX_NR], wh[LB_MAX_NR];
            UNROLL for (int j = 0; j < NR; j++) {
                const __m256i v =
                    load8_u8_to_i32(qw + (size_t)j * ldq + (size_t)b * 16 + c * 8);
                const __m256 ml = _mm256_permutevar8x32_ps(sc[j], v);
                const __m256 sl = _mm256_castsi256_ps(
                    _mm256_and_si256(_mm256_slli_epi32(v, 28), sgn));
                const __m256i h = _mm256_srli_epi32(v, 4);
                const __m256 mh = _mm256_permutevar8x32_ps(sc[j], h);
                const __m256 sh = _mm256_castsi256_ps(
                    _mm256_and_si256(_mm256_slli_epi32(v, 24), sgn));
                wl[j] = _mm256_xor_ps(ml, sl);
                wh[j] = _mm256_xor_ps(mh, sh);
            }
            UNROLL for (int i = 0; i < MR; i++) {
                const __m256 xl = _mm256_loadu_ps(xb + (size_t)i * ldx + c * 8);
                const __m256 xh = _mm256_loadu_ps(xb + (size_t)i * ldx + 16 + c * 8);
                UNROLL for (int j = 0; j < NR; j++) {
                    accl[i][j] = _mm256_fmadd_ps(wl[j], xl, accl[i][j]);
                    acch[i][j] = _mm256_fmadd_ps(wh[j], xh, acch[i][j]);
                }
            }
        }
    }
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++)
        y[(size_t)i * ldy + j] += hsum8(_mm256_add_ps(accl[i][j], acch[i][j]));
}

/* 128-element group: lo0 = elems 0..31, lo1 = 32..63, hi0 = 64..95, hi1 = 96..127 */
AI void unpack(const uint8_t *wp, __m256i w[4]) {
    const __m256i m4 = _mm256_set1_epi8(15);
    const __m256i a = _mm256_loadu_si256((const __m256i *)wp);
    const __m256i b = _mm256_loadu_si256((const __m256i *)(wp + 32));
    w[0] = _mm256_and_si256(a, m4);
    w[1] = _mm256_and_si256(b, m4);
    w[2] = _mm256_and_si256(_mm256_srli_epi16(a, 4), m4);
    w[3] = _mm256_and_si256(_mm256_srli_epi16(b, 4), m4);
}

AI __m256i group_vec(const __m256i w[4], const int8_t *a, int32_t corr) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i p = _mm256_maddubs_epi16(w[0], _mm256_loadu_si256((const __m256i *)a));
    p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w[1], _mm256_loadu_si256((const __m256i *)(a + 32))));
    p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w[2], _mm256_loadu_si256((const __m256i *)(a + 64))));
    p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w[3], _mm256_loadu_si256((const __m256i *)(a + 96))));
    return _mm256_add_epi32(_mm256_madd_epi16(p, ones),
                            _mm256_setr_epi32(corr, 0, 0, 0, 0, 0, 0, 0));
}

AI void w4a8_body(const int MR, const int NR, const int8_t *xq, int ldx,
                  const float *xs, const int32_t *asum, int lda,
                  const uint8_t *qw, int ldq, const float *s, const uint8_t *z,
                  int lds, int k0, int k1, float *y, int ldy) {
    __m256 facc[LB_MAX_MR][LB_MAX_NR];
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++)
        facc[i][j] = _mm256_setzero_ps();
    for (int g = k0 / LB_GROUP; g < k1 / LB_GROUP; g++) {
        __m256i w[LB_MAX_NR][4];
        __m256 sv[LB_MAX_NR];
        int32_t zz[LB_MAX_NR];
        UNROLL for (int j = 0; j < NR; j++) {
            PF(qw + (size_t)j * ldq + (size_t)g * 64);
            unpack(qw + (size_t)j * ldq + (size_t)g * 64, w[j]);
            sv[j] = _mm256_set1_ps(s[(size_t)j * lds + g]);
            zz[j] = z ? z[(size_t)j * lds + g] : 8;
        }
        UNROLL for (int i = 0; i < MR; i++) {
            const int8_t *a = xq + (size_t)i * ldx + (size_t)g * LB_GROUP;
            const int32_t as = asum[(size_t)i * lda + g];
            UNROLL for (int j = 0; j < NR; j++) {
                const __m256i acc = group_vec(w[j], a, -zz[j] * as);
                facc[i][j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv[j], facc[i][j]);
            }
        }
    }
    UNROLL for (int i = 0; i < MR; i++) UNROLL for (int j = 0; j < NR; j++)
        y[(size_t)i * ldy + j] += xs[i] * hsum8(facc[i][j]);
}

#define DEF_TILE(MR, NR)                                                        \
    static void w4a16_##MR##x##NR(const float *x, int ldx, const uint8_t *qw,    \
                                  int ldq, const float *s, const uint8_t *z,     \
                                  int lds, int k0, int k1, float *y, int ldy) {  \
        w4a16_body(MR, NR, x, ldx, qw, ldq, s, z, lds, k0, k1, y, ldy);          \
    }                                                                            \
    static void mxfp4_##MR##x##NR(const float *x, int ldx, const uint8_t *qw,    \
                                  int ldq, const uint8_t *e, int lde, int k0,    \
                                  int k1, float *y, int ldy) {                   \
        mxfp4_body(MR, NR, x, ldx, qw, ldq, e, lde, k0, k1, y, ldy);             \
    }                                                                            \
    static void w4a8_##MR##x##NR(const int8_t *xq, int ldx, const float *xs,      \
                                 const int32_t *asum, int lda, const uint8_t *qw, \
                                 int ldq, const float *s, const uint8_t *z,       \
                                 int lds, int k0, int k1, float *y, int ldy) {    \
        w4a8_body(MR, NR, xq, ldx, xs, asum, lda, qw, ldq, s, z, lds, k0, k1, y,  \
                  ldy);                                                          \
    }
#include "tiles.inc"
#undef DEF_TILE

lb_w4a16_fn lb_w4a16_micro_avx2(int mr, int nr) {
#define DEF_TILE(MR, NR) if (mr == MR && nr == NR) return w4a16_##MR##x##NR;
#include "tiles.inc"
#undef DEF_TILE
    return 0;
}
lb_mxfp4_fn lb_mxfp4_micro_avx2(int mr, int nr) {
#define DEF_TILE(MR, NR) if (mr == MR && nr == NR) return mxfp4_##MR##x##NR;
#include "tiles.inc"
#undef DEF_TILE
    return 0;
}
lb_w4a8_fn lb_w4a8_micro_avx2(int mr, int nr) {
#define DEF_TILE(MR, NR) if (mr == MR && nr == NR) return w4a8_##MR##x##NR;
#include "tiles.inc"
#undef DEF_TILE
    return 0;
}

void lb_w4a8_groupacc_avx2(const int8_t *xq, const int32_t *asum, int M, int Kp,
                           const uint8_t *qw, const uint8_t *z, int N,
                           int32_t *out) {
    const int G = Kp / LB_GROUP;
    for (int n = 0; n < N; n++)
        for (int g = 0; g < G; g++) {
            __m256i w[4];
            unpack(qw + (size_t)n * (Kp / 2) + (size_t)g * 64, w);
            const int32_t zz = z ? z[(size_t)n * G + g] : 8;
            for (int m = 0; m < M; m++)
                out[((size_t)m * N + n) * G + g] = hsum8i(
                    group_vec(w, xq + (size_t)m * Kp + (size_t)g * LB_GROUP,
                              -zz * asum[(size_t)m * G + g]));
        }
}
