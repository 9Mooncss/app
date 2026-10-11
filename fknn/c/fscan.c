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
        /* 1) table of squared gaps (AVX-512), quantized down to uint8 */
        __m512 vmax = _mm512_setzero_ps();
        const __m512 zero = _mm512_setzero_ps();
        for (int i = 0; i < d; i++) {
            const __m512 vq = _mm512_set1_ps(q[i]);
            __m512 g = _mm512_max_ps(_mm512_max_ps(_mm512_sub_ps(_mm512_loadu_ps(edges_lo + i * 16), vq),
                                                   _mm512_sub_ps(vq, _mm512_loadu_ps(edges_hi + i * 16))), zero);
            g = _mm512_mul_ps(g, g);
            _mm512_storeu_ps(T + i * 16, g);
            vmax = _mm512_max_ps(vmax, g);
        }
        float tmax = _mm512_reduce_max_ps(vmax);
        float scale = tmax > 0 ? tmax / 255.f : 1.f, inv = 1.f / scale;
        const __m512 vinv = _mm512_set1_ps(inv), v255 = _mm512_set1_ps(255.f);
        for (int i = 0; i < d; i++) {
            __m512 v = _mm512_min_ps(_mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(T + i * 16), vinv), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC), v255);
            tbl[i] = _mm256_broadcastsi128_si256(_mm512_cvtusepi32_epi8(_mm512_cvttps_epu32(v)));
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

/* ---------------------------------------------------------------------------------
 * v2 kernel (AVX-512BW): blocks of 64 slots, table quantized DOWN to 6 bits (0..63)
 * so that 4 dimensions can be summed in uint8 without overflow (4*63 = 252) before
 * widening to uint16. Layout: codes2 [nb][d4][64] with d4 = d rounded up to a
 * multiple of 4 (padding dims have code 0 and an all-zero table row).
 * Bounds stay valid lower bounds (floor quantization, no saturation for d4*63 < 65536).
 * ------------------------------------------------------------------------------- */
static inline void scan_block64(const uint8_t *blk, const __m512i *tbl, int d4, uint16_t *lb64) {
    __m512i acc_lo = _mm512_setzero_si512(), acc_hi = _mm512_setzero_si512(), z = _mm512_setzero_si512();
    for (int i = 0; i < d4; i += 4) {
        __m512i s = _mm512_shuffle_epi8(tbl[i], _mm512_loadu_si512(blk + (int64_t)i * 64));
        s = _mm512_add_epi8(s, _mm512_shuffle_epi8(tbl[i + 1], _mm512_loadu_si512(blk + (int64_t)(i + 1) * 64)));
        s = _mm512_add_epi8(s, _mm512_shuffle_epi8(tbl[i + 2], _mm512_loadu_si512(blk + (int64_t)(i + 2) * 64)));
        s = _mm512_add_epi8(s, _mm512_shuffle_epi8(tbl[i + 3], _mm512_loadu_si512(blk + (int64_t)(i + 3) * 64)));
        acc_lo = _mm512_add_epi16(acc_lo, _mm512_unpacklo_epi8(s, z));
        acc_hi = _mm512_add_epi16(acc_hi, _mm512_unpackhi_epi8(s, z));
    }
    uint16_t lo[32], hi[32];
    _mm512_storeu_si512(lo, acc_lo);
    _mm512_storeu_si512(hi, acc_hi);
    /* per 128-bit lane L (0..3): unpacklo gives bytes 16L..16L+7, unpackhi bytes 16L+8..16L+15 */
    for (int L = 0; L < 4; L++)
        for (int j = 0; j < 8; j++) { lb64[16 * L + j] = lo[8 * L + j]; lb64[16 * L + 8 + j] = hi[8 * L + j]; }
}

void va2_batch(const float *Xs, int d, int d4, const uint8_t *codes, const int64_t *slot_row, const int64_t *lblock,
               const float *edges_lo, const float *edges_hi,
               const float *Q, int nq, const int64_t *qlab, const int64_t *qoff, int k,
               int64_t *out_id, float *out_d, double *t_ns, int64_t *n_refined, int64_t *n_slots,
               uint16_t *lb_buf, int64_t *slot_buf) {
    __m512i *tbl = aligned_alloc(64, sizeof(__m512i) * d4);
    float *T = malloc(sizeof(float) * d * 16);
    for (int qi = 0; qi < nq; qi++) {
        const float *q = Q + (int64_t)qi * d;
        double t0 = now_ns();
        float tmax = 0.f;
        for (int i = 0; i < d; i++) for (int c = 0; c < 16; c++) {
            float lo = edges_lo[i * 16 + c], hi = edges_hi[i * 16 + c], g = 0.f;
            if (q[i] < lo) g = lo - q[i]; else if (q[i] > hi) g = q[i] - hi;
            T[i * 16 + c] = g * g; if (g * g > tmax) tmax = g * g;
        }
        float scale = tmax > 0 ? tmax / 63.f : 1.f, inv = 1.f / scale;
        for (int i = 0; i < d4; i++) {
            uint8_t row[16] = {0};
            if (i < d) for (int c = 0; c < 16; c++) { float v = floorf(T[i * 16 + c] * inv); row[c] = v > 63.f ? 63 : (uint8_t)v; }
            tbl[i] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)row));
        }
        int64_t ns = 0;
        for (int64_t a = qoff[qi]; a < qoff[qi + 1]; a++) {
            int64_t l = qlab[a];
            for (int64_t b = lblock[l]; b < lblock[l + 1]; b++) {
                uint16_t lb64[64];
                scan_block64(codes + b * (int64_t)d4 * 64, tbl, d4, lb64);
                for (int j = 0; j < 64; j++) {
                    int64_t s = b * 64 + j;
                    if (slot_row[s] < 0) continue;
                    lb_buf[ns] = lb64[j]; slot_buf[ns] = slot_row[s]; ns++;
                }
            }
        }
        topk_t tk; tk_init(&tk, k);
        int64_t refined = 0;
        if (ns > 0) {
            static int64_t hist[65536 >> 4];
            memset(hist, 0, sizeof(hist));
            for (int64_t s = 0; s < ns; s++) hist[lb_buf[s] >> 4]++;
            int64_t acc = 0; int cut = 0;
            for (; cut < (65536 >> 4); cut++) { acc += hist[cut]; if (acc >= k) break; }
            uint32_t lim = ((uint32_t)cut + 1) << 4;
            for (int64_t s = 0; s < ns; s++) if (lb_buf[s] < lim) { tk_push(&tk, l2(Xs + slot_buf[s] * d, q, d), slot_buf[s]); refined++; slot_buf[s] = -1; }
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

/* ---------------------------------------------------------------------------------
 * v3 kernel (AVX-512 VBMI): b-bit codes (b = 5..7), 64-slot blocks, one-byte codes.
 * b <= 6: vpermb  (_mm512_permutexvar_epi8)  = 64 lookups into a 64-entry table;
 * b == 7: vpermi2b (_mm512_permutex2var_epi8) = 64 lookups into a 128-entry table.
 * Table rounded DOWN to uint8 (255 levels), saturating uint16 accumulation:
 * both only lower the bound -> exactness is kept.
 * edges_lo/hi: [d][2^b] cell intervals.  codes3: [nb][d][64].
 * ------------------------------------------------------------------------------- */
static inline void scan_block64_v3(const uint8_t *blk, const __m512i *tA, const __m512i *tB, int d, int b7, uint16_t *lb64) {
    __m512i acc_lo = _mm512_setzero_si512(), acc_hi = _mm512_setzero_si512(), z = _mm512_setzero_si512();
    for (int i = 0; i < d; i++) {
        __m512i c = _mm512_loadu_si512(blk + (int64_t)i * 64);
        __m512i v = b7 ? _mm512_permutex2var_epi8(tA[i], c, tB[i]) : _mm512_permutexvar_epi8(c, tA[i]);
        acc_lo = _mm512_adds_epu16(acc_lo, _mm512_unpacklo_epi8(v, z));
        acc_hi = _mm512_adds_epu16(acc_hi, _mm512_unpackhi_epi8(v, z));
    }
    uint16_t lo[32], hi[32];
    _mm512_storeu_si512(lo, acc_lo);
    _mm512_storeu_si512(hi, acc_hi);
    for (int L = 0; L < 4; L++)
        for (int j = 0; j < 8; j++) { lb64[16 * L + j] = lo[8 * L + j]; lb64[16 * L + 8 + j] = hi[8 * L + j]; }
}

void va3_batch(const float *Xs, int d, int bits, const uint8_t *codes, const int64_t *slot_row, const int64_t *lblock,
               const float *edges_lo, const float *edges_hi,
               const float *Q, int nq, const int64_t *qlab, const int64_t *qoff, int k,
               int64_t *out_id, float *out_d, double *t_ns, int64_t *n_refined, int64_t *n_slots,
               uint16_t *lb_buf, int64_t *slot_buf) {
    const int C = 1 << bits, b7 = bits == 7;
    __m512i *tA = aligned_alloc(64, sizeof(__m512i) * d), *tB = aligned_alloc(64, sizeof(__m512i) * d);
    float *T = malloc(sizeof(float) * d * C);
    for (int qi = 0; qi < nq; qi++) {
        const float *q = Q + (int64_t)qi * d;
        double t0 = now_ns();
        /* table build fully in AVX-512 (scalar fmaxf/floorf cost ~2-10 ns per entry) */
        __m512 vmax = _mm512_setzero_ps();
        const __m512 zero = _mm512_setzero_ps();
        for (int i = 0; i < d; i++) {
            const __m512 vq = _mm512_set1_ps(q[i]);
            for (int c = 0; c < C; c += 16) {   /* C = 2^bits, a multiple of 16 */
                __m512 g = _mm512_max_ps(_mm512_max_ps(_mm512_sub_ps(_mm512_loadu_ps(edges_lo + (int64_t)i * C + c), vq),
                                                       _mm512_sub_ps(vq, _mm512_loadu_ps(edges_hi + (int64_t)i * C + c))), zero);
                g = _mm512_mul_ps(g, g);
                _mm512_storeu_ps(T + (int64_t)i * C + c, g);
                vmax = _mm512_max_ps(vmax, g);
            }
        }
        float tmax = _mm512_reduce_max_ps(vmax);
        float scale = tmax > 0 ? tmax / 255.f : 1.f, inv = 1.f / scale;
        const __m512 vinv = _mm512_set1_ps(inv), v255 = _mm512_set1_ps(255.f);
        for (int i = 0; i < d; i++) {
            uint8_t row[128] __attribute__((aligned(64))) = {0};
            const float *Ti = T + (int64_t)i * C;
            for (int c = 0; c < C; c += 16) {
                __m512 v = _mm512_min_ps(_mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(Ti + c), vinv), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC), v255);
                _mm_storeu_si128((__m128i *)(row + c), _mm512_cvtusepi32_epi8(_mm512_cvttps_epu32(v)));
            }
            if (C < 64) for (int c = C; c < 64; c++) row[c] = row[c % C];
            tA[i] = _mm512_load_si512(row);
            tB[i] = _mm512_load_si512(row + 64);
        }
        int64_t ns = 0;
        for (int64_t a = qoff[qi]; a < qoff[qi + 1]; a++) {
            int64_t l = qlab[a];
            for (int64_t b = lblock[l]; b < lblock[l + 1]; b++) {
                uint16_t lb64[64];
                scan_block64_v3(codes + b * (int64_t)d * 64, tA, tB, d, b7, lb64);
                for (int j = 0; j < 64; j++) {
                    int64_t s = b * 64 + j;
                    if (slot_row[s] < 0) continue;
                    lb_buf[ns] = lb64[j]; slot_buf[ns] = slot_row[s]; ns++;
                }
            }
        }
        topk_t tk; tk_init(&tk, k);
        int64_t refined = 0;
        if (ns > 0) {
            static int64_t hist[65536 >> 4];
            memset(hist, 0, sizeof(hist));
            for (int64_t s = 0; s < ns; s++) hist[lb_buf[s] >> 4]++;
            int64_t acc = 0; int cut = 0;
            for (; cut < (65536 >> 4); cut++) { acc += hist[cut]; if (acc >= k) break; }
            uint32_t lim = ((uint32_t)cut + 1) << 4;
            for (int64_t s = 0; s < ns; s++) if (lb_buf[s] < lim) { tk_push(&tk, l2(Xs + slot_buf[s] * d, q, d), slot_buf[s]); refined++; slot_buf[s] = -1; }
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
    free(tA); free(tB); free(T);
}

/* ---------------------------------------------------------------------------------
 * v4 = v3 + streaming refinement + block early-abandon.
 * Dimensions are pre-permuted by decreasing variance (codes4/edges4 in that order;
 * Qp = queries permuted the same way, used only to build the table; exact distances
 * use the original Xs and Q). Blocks are processed in order; after each block, lanes
 * whose bound is below the current k-th exact distance t are refined immediately, so t
 * tightens early. Every CHK dims the block's partial bounds are checked: if all 64
 * lanes already have partial bound >= t, the block is abandoned (partial <= full bound,
 * so this is exact).
 * ------------------------------------------------------------------------------- */
#define CHK 32
void va4_batch(const float *Xs, int d, int bits, const uint8_t *codes, const int64_t *slot_row, const int64_t *lblock,
               const float *edges_lo, const float *edges_hi,
               const float *Q, const float *Qp, int nq, const int64_t *qlab, const int64_t *qoff, int k,
               int64_t *out_id, float *out_d, double *t_ns, int64_t *n_refined, int64_t *n_dims_scanned) {
    const int C = 1 << bits, b7 = bits == 7;
    __m512i *tA = aligned_alloc(64, sizeof(__m512i) * d), *tB = aligned_alloc(64, sizeof(__m512i) * d);
    float *T = malloc(sizeof(float) * d * C);
    for (int qi = 0; qi < nq; qi++) {
        const float *q = Q + (int64_t)qi * d, *qp = Qp + (int64_t)qi * d;
        double t0 = now_ns();
        __m512 vmax = _mm512_setzero_ps();
        const __m512 zero = _mm512_setzero_ps();
        for (int i = 0; i < d; i++) {
            const __m512 vq = _mm512_set1_ps(qp[i]);
            for (int c = 0; c < C; c += 16) {
                __m512 g = _mm512_max_ps(_mm512_max_ps(_mm512_sub_ps(_mm512_loadu_ps(edges_lo + (int64_t)i * C + c), vq),
                                                       _mm512_sub_ps(vq, _mm512_loadu_ps(edges_hi + (int64_t)i * C + c))), zero);
                g = _mm512_mul_ps(g, g);
                _mm512_storeu_ps(T + (int64_t)i * C + c, g);
                vmax = _mm512_max_ps(vmax, g);
            }
        }
        float tmax = _mm512_reduce_max_ps(vmax);
        float scale = tmax > 0 ? tmax / 255.f : 1.f, inv = 1.f / scale;
        const __m512 vinv = _mm512_set1_ps(inv), v255 = _mm512_set1_ps(255.f);
        for (int i = 0; i < d; i++) {
            uint8_t row[128] __attribute__((aligned(64))) = {0};
            const float *Ti = T + (int64_t)i * C;
            for (int c = 0; c < C; c += 16) {
                __m512 v = _mm512_min_ps(_mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(Ti + c), vinv), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC), v255);
                _mm_storeu_si128((__m128i *)(row + c), _mm512_cvtusepi32_epi8(_mm512_cvttps_epu32(v)));
            }
            if (C < 64) for (int c = C; c < 64; c++) row[c] = row[c % C];
            tA[i] = _mm512_load_si512(row);
            tB[i] = _mm512_load_si512(row + 64);
        }
        topk_t tk; tk_init(&tk, k);
        int64_t refined = 0, dims = 0;
        const __m512i z = _mm512_setzero_si512();
        for (int64_t a = qoff[qi]; a < qoff[qi + 1]; a++) {
            int64_t l = qlab[a];
            for (int64_t b = lblock[l]; b < lblock[l + 1]; b++) {
                const uint8_t *blk = codes + b * (int64_t)d * 64;
                __m512i acc_lo = z, acc_hi = z;
                float t = tk_worst(&tk);
                /* threshold in table units (bound*scale >= t  <=>  bound >= t/scale) */
                uint32_t tq = t == INFINITY ? 0xFFFFFFFFu : (uint32_t)fminf(t * inv, 65535.f);
                int abandoned = 0, i = 0;
                for (; i < d; i++) {
                    __m512i c = _mm512_loadu_si512(blk + (int64_t)i * 64);
                    __m512i v = b7 ? _mm512_permutex2var_epi8(tA[i], c, tB[i]) : _mm512_permutexvar_epi8(c, tA[i]);
                    acc_lo = _mm512_adds_epu16(acc_lo, _mm512_unpacklo_epi8(v, z));
                    acc_hi = _mm512_adds_epu16(acc_hi, _mm512_unpackhi_epi8(v, z));
                    if (((i + 1) % CHK) == 0 && tq != 0xFFFFFFFFu && i + 1 < d) {
                        __m512i mn = _mm512_min_epu16(acc_lo, acc_hi);
                        /* all lanes >= tq ? (padding lanes are fine: they are skipped anyway) */
                        if (_mm512_cmplt_epu16_mask(mn, _mm512_set1_epi16((short)tq)) == 0) { abandoned = 1; i++; break; }
                    }
                }
                dims += abandoned ? i : d;
                if (abandoned) continue;
                uint16_t lo[32], hi[32], lb64[64];
                _mm512_storeu_si512(lo, acc_lo);
                _mm512_storeu_si512(hi, acc_hi);
                for (int L = 0; L < 4; L++)
                    for (int j = 0; j < 8; j++) { lb64[16 * L + j] = lo[8 * L + j]; lb64[16 * L + 8 + j] = hi[8 * L + j]; }
                for (int j = 0; j < 64; j++) {
                    int64_t s = b * 64 + j, r = slot_row[s];
                    if (r < 0) continue;
                    if ((float)lb64[j] * scale >= tk_worst(&tk)) continue;
                    tk_push(&tk, l2(Xs + r * d, q, d), r); refined++;
                }
            }
        }
        t_ns[qi] = now_ns() - t0;
        n_refined[qi] = refined; n_dims_scanned[qi] = dims;
        for (int i = 0; i < k; i++) { out_id[qi * k + i] = i < tk.n ? tk.id[i] : -1; out_d[qi * k + i] = i < tk.n ? tk.d[i] : INFINITY; }
    }
    free(tA); free(tB); free(T);
}
