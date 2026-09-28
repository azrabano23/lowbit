/* Portable scalar reference kernels (plain C99, no intrinsics).
 * Built with -fno-tree-vectorize so it is an honest scalar baseline.
 * Only the 1x1 micro-tile exists; the driver forces mr = nr = 1. */
#include "lowbit.h"

static const float E2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                               4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                               -2.0f, -3.0f, -4.0f, -6.0f};

static void w4a16_1x1(const float *x, int ldx, const uint8_t *qw, int ldq,
                      const float *s, const uint8_t *z, int lds, int k0, int k1,
                      float *y, int ldy) {
    (void)ldx; (void)ldq; (void)lds; (void)ldy;
    float acc = 0.0f;
    for (int g = k0 / LB_GROUP; g < k1 / LB_GROUP; g++) {
        const uint8_t *wp = qw + (size_t)g * 64;
        const float *xg = x + (size_t)g * LB_GROUP;
        const float sc = s[g];
        const float zz = z ? (float)z[g] : 8.0f;
        float part = 0.0f;
        for (int j = 0; j < 64; j++) {
            const uint8_t b = wp[j];
            part += ((float)(b & 15) - zz) * xg[j];
            part += ((float)(b >> 4) - zz) * xg[64 + j];
        }
        acc += part * sc;
    }
    y[0] += acc;
}

static void mxfp4_1x1(const float *x, int ldx, const uint8_t *qw, int ldq,
                      const uint8_t *e, int lde, int k0, int k1, float *y,
                      int ldy) {
    (void)ldx; (void)ldq; (void)lde; (void)ldy;
    float acc = 0.0f;
    for (int b = k0 / LB_MXBLOCK; b < k1 / LB_MXBLOCK; b++) {
        const uint8_t *wp = qw + (size_t)b * 16;
        const float *xb = x + (size_t)b * LB_MXBLOCK;
        float part = 0.0f;
        for (int j = 0; j < 16; j++) {
            part += E2M1[wp[j] & 15] * xb[j];
            part += E2M1[wp[j] >> 4] * xb[16 + j];
        }
        acc += part * lb_e8m0_lut[e[b]];
    }
    y[0] += acc;
}

static inline int32_t w4a8_group_dot(const uint8_t *wp, const int8_t *a,
                                     int32_t zz, int32_t asum) {
    /* sum_k q_k * a_k over unsigned codes, then the zero-point correction:
     * sum (q - z) a = sum q a - z * sum a  (the same identity VNNI uses). */
    int32_t acc = 0;
    for (int j = 0; j < 64; j++) {
        acc += (int32_t)(wp[j] & 15) * a[j];
        acc += (int32_t)(wp[j] >> 4) * a[64 + j];
    }
    return acc - zz * asum;
}

static void w4a8_1x1(const int8_t *xq, int ldx, const float *xs,
                     const int32_t *asum, int lda, const uint8_t *qw, int ldq,
                     const float *s, const uint8_t *z, int lds, int k0, int k1,
                     float *y, int ldy) {
    (void)ldx; (void)lda; (void)ldq; (void)lds; (void)ldy;
    float acc = 0.0f;
    for (int g = k0 / LB_GROUP; g < k1 / LB_GROUP; g++) {
        const int32_t zz = z ? z[g] : 8;
        const int32_t d = w4a8_group_dot(qw + (size_t)g * 64,
                                         xq + (size_t)g * LB_GROUP, zz, asum[g]);
        acc += (float)d * s[g];
    }
    y[0] += acc * xs[0];
}

lb_w4a16_fn lb_w4a16_micro_scalar(int mr, int nr) {
    return (mr == 1 && nr == 1) ? w4a16_1x1 : 0;
}
lb_mxfp4_fn lb_mxfp4_micro_scalar(int mr, int nr) {
    return (mr == 1 && nr == 1) ? mxfp4_1x1 : 0;
}
lb_w4a8_fn lb_w4a8_micro_scalar(int mr, int nr) {
    return (mr == 1 && nr == 1) ? w4a8_1x1 : 0;
}

void lb_w4a8_groupacc_scalar(const int8_t *xq, const int32_t *asum, int M,
                             int Kp, const uint8_t *qw, const uint8_t *z, int N,
                             int32_t *out) {
    const int G = Kp / LB_GROUP;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int g = 0; g < G; g++) {
                const int32_t zz = z ? z[(size_t)n * G + g] : 8;
                out[((size_t)m * N + n) * G + g] = w4a8_group_dot(
                    qw + (size_t)n * (Kp / 2) + (size_t)g * 64,
                    xq + (size_t)m * Kp + (size_t)g * LB_GROUP, zz,
                    asum[(size_t)m * G + g]);
            }
}
