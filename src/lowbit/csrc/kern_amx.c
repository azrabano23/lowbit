/* AMX-INT8 W4A8 kernel (Sapphire Rapids and later). Compiled with
 * -mamx-tile -mamx-int8 -mavx512f -mavx512bw; only used when CPUID reports
 * AMX-TILE + AMX-INT8 and the kernel grants the tile-data permission.
 *
 * TDPBSUD multiplies signed int8 (tile A = activations) by unsigned int8
 * (tile B = weight codes 0..15) in groups of four and accumulates int32:
 *     C[m][n] += sum_{i<4} A[m][4r+i] * B[r][4n+i]
 * so B must hold, per row r, four consecutive k for each of 16 columns n
 * (the "VNNI" layout). We therefore use a one-time repacked weight layout
 * (lb_amx_pack in Python): per block of 16 output rows and per group of
 * 128 k, 1 KiB where byte (r, 4c+i) holds code(n0+c, 128g+4r+i) in its low
 * nibble and code(n0+c, 128g+64+4r+i) in its high nibble. At run time a
 * group is unpacked with vpand / vpsrlw into two 16x64 B tiles in L1.
 *
 * Per group: C = A0.B0 + A1.B1 (exact int32), then
 *     y[m][n] += s[n] * C[m][n] - (s[n] * z[n]) * sum_k a[m][k]
 * (the zero-point correction, same identity as the VNNI kernel). */
#define _GNU_SOURCE /* syscall() */
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cpuid.h>

#include "lowbit.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18
#define UNROLL _Pragma("GCC unroll 16")

typedef struct {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __attribute__((packed, aligned(64))) tilecfg_t;

/* tile registers (the intrinsics stringify their argument, so literals only):
 * tmm0 = C, tmm1/2 = A (k 0..63 / 64..127), tmm3/4 = B (low / high nibbles) */
enum { TC = 0, TA0 = 1, TA1 = 2, TB0 = 3, TB1 = 4 };

int lb_amx_available(void) {
    static int state = -1;
    if (state >= 0) return state;
    unsigned a, b, c, d;
    state = 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return state;
    if (!((d >> 24) & 1) || !((d >> 25) & 1)) return state; /* AMX-TILE, AMX-INT8 */
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) return state;
    state = 1;
    return state;
}

static void config(int mb) {
    tilecfg_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette_id = 1;
    cfg.rows[TC] = (uint8_t)mb;  cfg.colsb[TC] = 64;
    cfg.rows[TA0] = (uint8_t)mb; cfg.colsb[TA0] = 64;
    cfg.rows[TA1] = (uint8_t)mb; cfg.colsb[TA1] = 64;
    cfg.rows[TB0] = 16;          cfg.colsb[TB0] = 64;
    cfg.rows[TB1] = 16;          cfg.colsb[TB1] = 64;
    _tile_loadconfig(&cfg);
}

static inline void unpack_group(const uint8_t *pk, uint8_t *b0, uint8_t *b1) {
    const __m512i m4 = _mm512_set1_epi8(15);
    UNROLL for (int r = 0; r < 16; r++) {
        const __m512i v = _mm512_loadu_si512((const void *)(pk + r * 64));
        _mm512_store_si512((void *)(b0 + r * 64), _mm512_and_si512(v, m4));
        _mm512_store_si512((void *)(b1 + r * 64),
                           _mm512_and_si512(_mm512_srli_epi16(v, 4), m4));
    }
}

/* one 16-column block of W against rows [m0, m0+mb) of X, all groups */
static void block(const int8_t *xq, int Kp, const float *xs, const float *asumf,
                  const uint8_t *qwa, const float *sa, const float *sza, int nblk,
                  int m0, int mb, int N, float *y, uint8_t *bufB, int32_t *bufC) {
    const int G = Kp / LB_GROUP;
    __m512 acc[16];
    UNROLL for (int m = 0; m < 16; m++) acc[m] = _mm512_setzero_ps();
    for (int g = 0; g < G; g++) {
        const size_t off = (size_t)nblk * G + g;
        unpack_group(qwa + off * 1024, bufB, bufB + 1024);
        _tile_loadd(3, bufB, 64);
        _tile_loadd(4, bufB + 1024, 64);
        const int8_t *a = xq + (size_t)m0 * Kp + (size_t)g * LB_GROUP;
        _tile_loadd(1, a, Kp);
        _tile_loadd(2, a + 64, Kp);
        _tile_zero(0);
        _tile_dpbsud(0, 1, 3);
        _tile_dpbsud(0, 2, 4);
        _tile_stored(0, bufC, 64);
        const __m512 sv = _mm512_loadu_ps(sa + off * 16);
        const __m512 szv = _mm512_loadu_ps(sza + off * 16);
        UNROLL for (int m = 0; m < 16; m++) {
            if (m < mb) {
                const __m512 c = _mm512_cvtepi32_ps(_mm512_load_si512((const void *)(bufC + m * 16)));
                acc[m] = _mm512_fmadd_ps(c, sv, acc[m]);
                acc[m] = _mm512_fnmadd_ps(szv, _mm512_set1_ps(asumf[(size_t)(m0 + m) * G + g]),
                                          acc[m]);
            }
        }
    }
    const int n0 = nblk * 16;
    const int nv = N - n0 < 16 ? N - n0 : 16;
    const __mmask16 mask = (__mmask16)((1u << nv) - 1u);
    UNROLL for (int m = 0; m < 16; m++)
        if (m < mb)
            _mm512_mask_storeu_ps(y + (size_t)(m0 + m) * N + n0, mask,
                                  _mm512_mul_ps(acc[m], _mm512_set1_ps(xs[m0 + m])));
}

int lb_gemm_w4a8_amx(const float *x, int M, int Kp, const uint8_t *qwa,
                     const float *sa, const float *sza, int N, float *y,
                     int nthreads) {
    if (M <= 0 || N <= 0 || Kp <= 0 || Kp % LB_GROUP) return -1;
    if (!lb_amx_available()) return -4;
    const int G = Kp / LB_GROUP;
    int8_t *xq = aligned_alloc(64, (((size_t)M * Kp + 63) / 64) * 64);
    float *xs = malloc(sizeof(float) * (size_t)M);
    int32_t *asum = malloc(sizeof(int32_t) * (size_t)M * G);
    float *asumf = malloc(sizeof(float) * (size_t)M * G);
    if (!xq || !xs || !asum || !asumf) {
        free(xq); free(xs); free(asum); free(asumf);
        return -3;
    }
    lb_quant_act_int8(x, M, Kp, xq, xs, asum);
    for (size_t i = 0; i < (size_t)M * G; i++) asumf[i] = (float)asum[i];
    const int nblocks = (N + 15) / 16;
    const int mblocks = (M + 15) / 16;
    int nt = nthreads > 0 ? nthreads : lb_max_threads();
    (void)nt;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt) if (nt > 1 && nblocks > 1)
#endif
    {
        uint8_t bufB[2048] __attribute__((aligned(64)));
        int32_t bufC[256] __attribute__((aligned(64)));
        int cur = -1;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int t = 0; t < nblocks * mblocks; t++) {
            const int nb = t / mblocks, mbk = t % mblocks;
            const int m0 = mbk * 16, mb = M - m0 < 16 ? M - m0 : 16;
            if (mb != cur) {
                config(mb);
                cur = mb;
            }
            block(xq, Kp, xs, asumf, qwa, sa, sza, nb, m0, mb, N, y, bufB, bufC);
        }
        _tile_release();
    }
    free(xq); free(xs); free(asum); free(asumf);
    return 3;
}

/* exact int32 group accumulators via the tile path (tests) */
int lb_w4a8_group_acc_amx(const int8_t *xq, const int32_t *asum, int M, int Kp,
                          const uint8_t *qwa, const uint8_t *za, int N,
                          int32_t *out) {
    if (M <= 0 || N <= 0 || Kp <= 0 || Kp % LB_GROUP) return -1;
    if (!lb_amx_available()) return -4;
    const int G = Kp / LB_GROUP;
    uint8_t bufB[2048] __attribute__((aligned(64)));
    int32_t bufC[256] __attribute__((aligned(64)));
    int cur = -1;
    for (int nb = 0; nb < (N + 15) / 16; nb++)
        for (int m0 = 0; m0 < M; m0 += 16) {
            const int mb = M - m0 < 16 ? M - m0 : 16;
            if (mb != cur) {
                config(mb);
                cur = mb;
            }
            for (int g = 0; g < G; g++) {
                const size_t off = (size_t)nb * G + g;
                unpack_group(qwa + off * 1024, bufB, bufB + 1024);
                _tile_loadd(3, bufB, 64);
                _tile_loadd(4, bufB + 1024, 64);
                const int8_t *a = xq + (size_t)m0 * Kp + (size_t)g * LB_GROUP;
                _tile_loadd(1, a, Kp);
                _tile_loadd(2, a + 64, Kp);
                _tile_zero(0);
                _tile_dpbsud(0, 1, 3);
                _tile_dpbsud(0, 2, 4);
                _tile_stored(0, bufC, 64);
                for (int m = 0; m < mb; m++)
                    for (int c = 0; c < 16 && nb * 16 + c < N; c++)
                        out[((size_t)(m0 + m) * N + nb * 16 + c) * G + g] =
                            bufC[m * 16 + c] - (int32_t)za[off * 16 + c] * asum[(size_t)(m0 + m) * G + g];
            }
        }
    _tile_release();
    return 3;
}
