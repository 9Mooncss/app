/* Exact filtered k-NN kernels, single thread.
 *
 * Data layout (prepared in Python, fknn/fastscan.py):
 *   Xs        float32 [n][d]   vectors sorted by label (each label's points contiguous)
 *   lstart    int64  [L+1]     label l occupies rows lstart[l] .. lstart[l+1]-1 of Xs
 *   codes     uint8  [nb][d][32]  4-bit VA codes (one per byte), blocks of 32 slots;
 *                              each label's points occupy whole blocks (padding slots)
 *   slot_row  int64  [nb*32]   row in Xs of each slot (-1 = padding)
 *   lblock    int64  [L+1]     label l occupies blocks lblock[l] .. lblock[l+1]-1
 *
 * bf_batch: scans every allowed row, exact L2 (the strongest flat baseline: contiguous).
 * va_batch: (1) per query table T[d][16] of squared cell gaps, rounded DOWN to uint8 with
 *           one scale -> lower bounds stay valid; (2) SIMD table-lookup scan of the codes
 *           (32 lookups per instruction), saturating uint16 accumulation (saturation only
 *           lowers the bound -> still valid); (3) exact distance for the k smallest bounds,
 *           then for every slot whose bound is below the current k-th distance.
 * Both return exact k-NN.  Times are measured in C per query (ns).
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return 1e9 * t.tv_sec + t.tv_nsec; }

static inline float l2(const float *a, const float *b, int d) {
    float s = 0.f;
    for (int j = 0; j < d; j++) { float t = a[j] - b[j]; s += t * t; }
    return s;
}

typedef struct { float d[64]; int64_t id[64]; int k, n; } topk_t;
static inline void tk_init(topk_t *t, int k) { t->k = k; t->n = 0; }
static inline float tk_worst(const topk_t *t) { return t->n < t->k ? INFINITY : t->d[t->n - 1]; }
static inline void tk_push(topk_t *t, float dist, int64_t id) {
    if (t->n == t->k && dist >= t->d[t->n - 1]) return;
    int i = t->n < t->k ? t->n++ : t->n - 1;
    while (i > 0 && t->d[i - 1] > dist) { t->d[i] = t->d[i - 1]; t->id[i] = t->id[i - 1]; i--; }
    t->d[i] = dist; t->id[i] = id;
}

void bf_batch(const float *Xs, int d, const int64_t *lstart,
              const float *Q, int nq, const int64_t *qlab, const int64_t *qoff, int k,
              int64_t *out_id, float *out_d, double *t_ns, int64_t *n_scanned) {
    for (int qi = 0; qi < nq; qi++) {
        const float *q = Q + (int64_t)qi * d;
        double t0 = now_ns();
        topk_t tk; tk_init(&tk, k);
        int64_t cnt = 0;
        for (int64_t a = qoff[qi]; a < qoff[qi + 1]; a++) {
            int64_t l = qlab[a];
            for (int64_t r = lstart[l]; r < lstart[l + 1]; r++) { tk_push(&tk, l2(Xs + r * d, q, d), r); cnt++; }
        }
        t_ns[qi] = now_ns() - t0;
        n_scanned[qi] = cnt;
        for (int i = 0; i < k; i++) { out_id[qi * k + i] = i < tk.n ? tk.id[i] : -1; out_d[qi * k + i] = i < tk.n ? tk.d[i] : INFINITY; }
    }
}

/* scan one block of 32 slots: returns 32 uint16 lower bounds (in table units) */
static inline void scan_block(const uint8_t *blk, const __m256i *tbl, int d, uint16_t *lb32) {
    __m256i acc_lo = _mm256_setzero_si256(), acc_hi = _mm256_setzero_si256(), z = _mm256_setzero_si256();
    for (int i = 0; i < d; i++) {
        __m256i c = _mm256_loadu_si256((const __m256i *)(blk + (int64_t)i * 32));
        __m256i v = _mm256_shuffle_epi8(tbl[i], c);
        acc_lo = _mm256_adds_epu16(acc_lo, _mm256_unpacklo_epi8(v, z));
        acc_hi = _mm256_adds_epu16(acc_hi, _mm256_unpackhi_epi8(v, z));
    }
    uint16_t lo[16], hi[16];
    _mm256_storeu_si256((__m256i *)lo, acc_lo);
    _mm256_storeu_si256((__m256i *)hi, acc_hi);
    /* unpacklo: bytes 0-7 and 16-23; unpackhi: bytes 8-15 and 24-31 */
    for (int j = 0; j < 8; j++) { lb32[j] = lo[j]; lb32[16 + j] = lo[8 + j]; lb32[8 + j] = hi[j]; lb32[24 + j] = hi[8 + j]; }
}

static int cmp_u32(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return (x > y) - (x < y); }

void va_batch(const float *Xs, int d, const uint8_t *codes, const int64_t *slot_row, const int64_t *lblock,
              const float *edges_lo, const float *edges_hi, /* [d][16] cell intervals (+-inf ok) */
              const float *Q, int nq, const int64_t *qlab, const int64_t *qoff, int k,
              int64_t *out_id, float *out_d, double *t_ns, int64_t *n_refined, int64_t *n_slots,
              uint16_t *lb_buf, int64_t *slot_buf /* scratch, size >= total slots */) {
    __m256i *tbl = aligned_alloc(32, sizeof(__m256i) * d);
    float *T = malloc(sizeof(float) * d * 16);
    for (int qi = 0; qi < nq; qi++) {
        const float *q = Q + (int64_t)qi * d;
        double t0 = now_ns();
        /* 1) table of squared gaps, quantized down to uint8 */
        float tmax = 0.f;
        for (int i = 0; i < d; i++) for (int c = 0; c < 16; c++) {
            float lo = edges_lo[i * 16 + c], hi = edges_hi[i * 16 + c], g = 0.f;
            if (q[i] < lo) g = lo - q[i]; else if (q[i] > hi) g = q[i] - hi;
            T[i * 16 + c] = g * g; if (g * g > tmax) tmax = g * g;
        }
        float scale = tmax > 0 ? tmax / 255.f : 1.f, inv = 1.f / scale;
        for (int i = 0; i < d; i++) {
            uint8_t row[16];
            for (int c = 0; c < 16; c++) { float v = floorf(T[i * 16 + c] * inv); row[c] = v > 255.f ? 255 : (uint8_t)v; }
            tbl[i] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)row));
        }
        /* 2) SIMD bound scan over the allowed labels' blocks */
        int64_t ns = 0;
        for (int64_t a = qoff[qi]; a < qoff[qi + 1]; a++) {
            int64_t l = qlab[a];
            for (int64_t b = lblock[l]; b < lblock[l + 1]; b++) {
                uint16_t lb32[32];
                scan_block(codes + b * (int64_t)d * 32, tbl, d, lb32);
                for (int j = 0; j < 32; j++) {
                    int64_t s = b * 32 + j;
                    if (slot_row[s] < 0) continue;
                    lb_buf[ns] = lb32[j]; slot_buf[ns] = slot_row[s]; ns++;
                }
            }
        }
        /* 3) refine: k smallest bounds first (bucket select on uint16), then all below threshold */
        topk_t tk; tk_init(&tk, k);
        int64_t refined = 0;
        if (ns > 0) {
            /* find the bound value of the k-th smallest via a coarse histogram */
            static int64_t hist[65536 >> 6];
            memset(hist, 0, sizeof(hist));
            for (int64_t s = 0; s < ns; s++) hist[lb_buf[s] >> 6]++;
            int64_t acc = 0; int cut = 0;
            for (; cut < (65536 >> 6); cut++) { acc += hist[cut]; if (acc >= k) break; }
            uint32_t lim = ((uint32_t)cut + 1) << 6;
            for (int64_t s = 0; s < ns; s++) if (lb_buf[s] < lim) { tk_push(&tk, l2(Xs + slot_buf[s] * d, q, d), slot_buf[s]); refined++; slot_buf[s] = -1; /* done */ }
            for (int64_t s = 0; s < ns; s++) {
                if (slot_buf[s] < 0) continue;
                if ((float)lb_buf[s] * scale >= tk_worst(&tk)) continue;
                tk_push(&tk, l2(Xs + slot_buf[s] * d, q, d), slot_buf[s]); refined++;
            }
        }
        t_ns[qi] = now_ns() - t0;
        n_refined[qi] = refined; n_slots[qi] = ns;
        for (int i = 0; i < k; i++) { out_id[qi * k + i] = i < tk.n ? tk.id[i] : -1; out_d[qi * k + i] = i < tk.n ? tk.d[i] : INFINITY; }
    }
    free(tbl); free(T);
}
