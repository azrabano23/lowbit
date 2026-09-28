/* Runtime CPU dispatch, tiling / threading driver and activation quantizer.
 * Compiled with baseline flags only (no -mavx*), so it is safe to run on
 * any x86-64 (or, without LB_X86, any) CPU. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lowbit.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int lb_version(void) { return 1; }

float lb_e8m0_lut[256];

__attribute__((constructor)) static void lb_init_tables(void) {
    for (int e = 0; e < 256; e++) lb_e8m0_lut[e] = lb_e8m0_to_f32((uint8_t)e);
}

int lb_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int lb_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int lb_cpu_isa(void) {
#ifdef LB_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))
        return LB_ISA_AVX512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return LB_ISA_AVX2;
#endif
    return LB_ISA_SCALAR;
}

int lb_has_vnni(void) {
#ifdef LB_X86
    __builtin_cpu_init();
    return lb_cpu_isa() == LB_ISA_AVX512 && __builtin_cpu_supports("avx512vnni");
#else
    return 0;
#endif
}

/* LOWBIT_ISA=scalar|avx2|avx512 caps the ISA (read once per process). */
int lb_active_isa(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int isa = lb_cpu_isa();
    const char *env = getenv("LOWBIT_ISA");
    if (env && *env) {
        int cap = isa;
        if (!strcmp(env, "scalar")) cap = LB_ISA_SCALAR;
        else if (!strcmp(env, "avx2")) cap = LB_ISA_AVX2;
        else if (!strcmp(env, "avx512")) cap = LB_ISA_AVX512;
        if (cap < isa) isa = cap;
    }
    cached = isa;
    return isa;
}

enum { K_W4A16 = 0, K_MXFP4 = 1, K_W4A8 = 2 };

int lb_effective_isa(int kind, int requested) {
    int isa = lb_active_isa();
    if (requested >= 0 && requested < isa) isa = requested;
    /* the AVX-512 W4A8 kernel needs VNNI; otherwise use the AVX2 maddubs one */
    if (kind == K_W4A8 && isa == LB_ISA_AVX512 && !lb_has_vnni()) isa = LB_ISA_AVX2;
    return isa;
}

typedef struct {
    int kind, M, N, Kp, G;
    const float *x;
    const int8_t *xq;
    const float *xs;
    const int32_t *asum;
    const uint8_t *qw, *z, *e;
    const float *s;
    float *y;
    void *fn[LB_MAX_MR + 1][LB_MAX_NR + 1];
} job_t;

static void *lookup(int kind, int isa, int mr, int nr) {
    switch (isa) {
#ifdef LB_X86
    case LB_ISA_AVX512:
        if (kind == K_W4A16) return (void *)lb_w4a16_micro_avx512(mr, nr);
        if (kind == K_MXFP4) return (void *)lb_mxfp4_micro_avx512(mr, nr);
        return (void *)lb_w4a8_micro_vnni(mr, nr);
    case LB_ISA_AVX2:
        if (kind == K_W4A16) return (void *)lb_w4a16_micro_avx2(mr, nr);
        if (kind == K_MXFP4) return (void *)lb_mxfp4_micro_avx2(mr, nr);
        return (void *)lb_w4a8_micro_avx2(mr, nr);
#endif
    default:
        if (kind == K_W4A16) return (void *)lb_w4a16_micro_scalar(mr, nr);
        if (kind == K_MXFP4) return (void *)lb_mxfp4_micro_scalar(mr, nr);
        return (void *)lb_w4a8_micro_scalar(mr, nr);
    }
}

static void call_micro(const job_t *J, int mr, int nr, int m0, int n0, int k0,
                       int k1) {
    const int Kp = J->Kp, G = J->G, N = J->N, ldq = Kp / 2;
    void *f = J->fn[mr][nr];
    float *y = J->y + (size_t)m0 * N + n0;
    switch (J->kind) {
    case K_W4A16:
        ((lb_w4a16_fn)f)(J->x + (size_t)m0 * Kp, Kp, J->qw + (size_t)n0 * ldq, ldq,
                         J->s + (size_t)n0 * G, J->z ? J->z + (size_t)n0 * G : 0,
                         G, k0, k1, y, N);
        break;
    case K_MXFP4: {
        const int lde = Kp / LB_MXBLOCK;
        ((lb_mxfp4_fn)f)(J->x + (size_t)m0 * Kp, Kp, J->qw + (size_t)n0 * ldq, ldq,
                         J->e + (size_t)n0 * lde, lde, k0, k1, y, N);
        break;
    }
    default:
        ((lb_w4a8_fn)f)(J->xq + (size_t)m0 * Kp, Kp, J->xs + m0,
                        J->asum + (size_t)m0 * G, G, J->qw + (size_t)n0 * ldq, ldq,
                        J->s + (size_t)n0 * G, J->z ? J->z + (size_t)n0 * G : 0, G,
                        k0, k1, y, N);
    }
}

/* largest compiled tile size <= want that fits the remaining rows */
static int fit(int want, int rem, void *const *avail, int stride) {
    for (int t = want < rem ? want : rem; t > 1; t--)
        if (avail[t * stride]) return t;
    return 1;
}

static int run(job_t *J, const lb_params *pin) {
    lb_params p = {-1, 0, 0, 0, 0, 0, 0};
    if (pin) p = *pin;
    const int isa = lb_effective_isa(J->kind, p.isa);
    int mr = p.mr > 0 ? p.mr : (J->M >= 4 ? 4 : 1);
    int nr = p.nr > 0 ? p.nr : 2;
    if (isa == LB_ISA_SCALAR) mr = nr = 1;
    if (mr > LB_MAX_MR) mr = LB_MAX_MR;
    if (nr > LB_MAX_NR) nr = LB_MAX_NR;
    memset(J->fn, 0, sizeof J->fn);
    for (int a = 1; a <= LB_MAX_MR; a++)
        for (int b = 1; b <= LB_MAX_NR; b++) J->fn[a][b] = lookup(J->kind, isa, a, b);
    if (!J->fn[mr][nr]) return -2; /* tile not compiled for this ISA */
    int nb = p.nb > 0 ? p.nb : 64;
    if (nb < nr) nb = nr;
    int kstep = p.kb > 0 ? ((p.kb + LB_GROUP - 1) / LB_GROUP) * LB_GROUP : J->Kp;
    if (kstep > J->Kp) kstep = J->Kp;
    int nt = p.nthreads > 0 ? p.nthreads : lb_max_threads();
    const int M = J->M, N = J->N, Kp = J->Kp;
    const int ntiles = (N + nb - 1) / nb;
    memset(J->y, 0, sizeof(float) * (size_t)M * N);
    (void)nt;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt) if (nt > 1 && ntiles > 1)
#endif
    for (int t = 0; t < ntiles; t++) {
        const int n0 = t * nb, n1 = n0 + nb < N ? n0 + nb : N;
        for (int k0 = 0; k0 < Kp; k0 += kstep) {
            const int k1 = k0 + kstep < Kp ? k0 + kstep : Kp;
            if (p.order == 0) {
                for (int m = 0; m < M;) {
                    const int me = fit(mr, M - m, &J->fn[0][1], LB_MAX_NR + 1);
                    for (int n = n0; n < n1;) {
                        int ne = nr < n1 - n ? nr : n1 - n;
                        while (!J->fn[me][ne]) ne--;
                        call_micro(J, me, ne, m, n, k0, k1);
                        n += ne;
                    }
                    m += me;
                }
            } else {
                for (int n = n0; n < n1;) {
                    int ne = nr < n1 - n ? nr : n1 - n;
                    while (!J->fn[1][ne]) ne--;
                    for (int m = 0; m < M;) {
                        int me = fit(mr, M - m, &J->fn[0][ne], LB_MAX_NR + 1);
                        call_micro(J, me, ne, m, n, k0, k1);
                        m += me;
                    }
                    n += ne;
                }
            }
        }
    }
    return isa;
}

static int check_shape(int M, int Kp, int N, int block) {
    if (M <= 0 || N <= 0 || Kp <= 0 || Kp % block) return -1;
    return 0;
}

int lb_gemm_w4a16(const float *x, int M, int Kp, const uint8_t *qw,
                  const float *scales, const uint8_t *zeros, int N, float *y,
                  const lb_params *p) {
    if (check_shape(M, Kp, N, LB_GROUP)) return -1;
    job_t J = {0};
    J.kind = K_W4A16; J.M = M; J.N = N; J.Kp = Kp; J.G = Kp / LB_GROUP;
    J.x = x; J.qw = qw; J.s = scales; J.z = zeros; J.y = y;
    return run(&J, p);
}

int lb_gemm_mxfp4(const float *x, int M, int Kp, const uint8_t *qw,
                  const uint8_t *e8m0, int N, float *y, const lb_params *p) {
    /* the driver's K blocking works in multiples of 128, which is a
     * multiple of 32, so any Kp % 32 == 0 is fine: the last block is short */
    if (check_shape(M, Kp, N, LB_MXBLOCK)) return -1;
    job_t J = {0};
    J.kind = K_MXFP4; J.M = M; J.N = N; J.Kp = Kp; J.G = Kp / LB_MXBLOCK;
    J.x = x; J.qw = qw; J.e = e8m0; J.y = y;
    return run(&J, p);
}

void lb_quant_act_int8(const float *x, int M, int Kp, int8_t *xq, float *xs,
                       int32_t *asum) {
    const int G = Kp / LB_GROUP;
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * Kp;
        float amax = 0.0f;
        for (int k = 0; k < Kp; k++) {
            const float a = fabsf(xr[k]);
            amax = a > amax ? a : amax;
        }
        const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
        xs[m] = sc;
        int8_t *qr = xq + (size_t)m * Kp;
        for (int g = 0; g < G; g++) {
            int32_t sum = 0;
            for (int k = g * LB_GROUP; k < (g + 1) * LB_GROUP; k++) {
                /* round half to even (as np.rint) via the 1.5*2^23 trick;
                 * exact for |v| < 2^22 and vectorizable, unlike nearbyintf */
                float q = (xr[k] / sc + 12582912.0f) - 12582912.0f;
                q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
                qr[k] = (int8_t)q;
                sum += qr[k];
            }
            asum[(size_t)m * G + g] = sum;
        }
    }
}

int lb_gemm_w4a8_q(const int8_t *xq, const float *xs, const int32_t *asum,
                   int M, int Kp, const uint8_t *qw, const float *scales,
                   const uint8_t *zeros, int N, float *y, const lb_params *p) {
    if (check_shape(M, Kp, N, LB_GROUP)) return -1;
    job_t J = {0};
    J.kind = K_W4A8; J.M = M; J.N = N; J.Kp = Kp; J.G = Kp / LB_GROUP;
    J.xq = xq; J.xs = xs; J.asum = asum; J.qw = qw; J.s = scales; J.z = zeros;
    J.y = y;
    return run(&J, p);
}

int lb_gemm_w4a8(const float *x, int M, int Kp, const uint8_t *qw,
                 const float *scales, const uint8_t *zeros, int N, float *y,
                 const lb_params *p) {
    if (check_shape(M, Kp, N, LB_GROUP)) return -1;
    const int G = Kp / LB_GROUP;
    /* 64-byte aligned scratch for the quantized activations */
    int8_t *xq = aligned_alloc(64, (((size_t)M * Kp + 63) / 64) * 64);
    float *xs = malloc(sizeof(float) * (size_t)M);
    int32_t *asum = malloc(sizeof(int32_t) * (size_t)M * G);
    int rc = -3;
    if (xq && xs && asum) {
        lb_quant_act_int8(x, M, Kp, xq, xs, asum);
        rc = lb_gemm_w4a8_q(xq, xs, asum, M, Kp, qw, scales, zeros, N, y, p);
    }
    free(xq); free(xs); free(asum);
    return rc;
}

int lb_w4a8_group_acc(const int8_t *xq, const int32_t *asum, int M, int Kp,
                      const uint8_t *qw, const uint8_t *zeros, int N,
                      int32_t *out, int isa_req) {
    if (check_shape(M, Kp, N, LB_GROUP)) return -1;
    const int isa = lb_effective_isa(K_W4A8, isa_req);
#ifdef LB_X86
    if (isa == LB_ISA_AVX512) {
        lb_w4a8_groupacc_vnni(xq, asum, M, Kp, qw, zeros, N, out);
        return isa;
    }
    if (isa == LB_ISA_AVX2) {
        lb_w4a8_groupacc_avx2(xq, asum, M, Kp, qw, zeros, N, out);
        return isa;
    }
#endif
    lb_w4a8_groupacc_scalar(xq, asum, M, Kp, qw, zeros, N, out);
    return LB_ISA_SCALAR;
}
