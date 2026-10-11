/* Micro-benchmark: cost of one "edge scan" (read a neighbor id + test a bit in the
 * allowed-bitmap) vs one L2 distance to a randomly located vector, single thread.
 * Used to convert S2H's edge counts into distance-equivalents.
 * Build: gcc -O3 -march=native edge_vs_dist.c -o edge_vs_dist
 * Run:   ./edge_vs_dist <n> <dim>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 88172645463325252ULL;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 200000;
    int d = argc > 2 ? atoi(argv[2]) : 100;
    const int M0 = 32, R = 2000000;
    float *X = malloc(sizeof(float) * n * d), *q = malloc(sizeof(float) * d);
    uint32_t *adj = malloc(sizeof(uint32_t) * n * M0);
    uint8_t *bits = calloc((n + 7) / 8, 1);
    uint32_t *ids = malloc(sizeof(uint32_t) * R);
    for (long i = 0; i < n * d; i++) X[i] = (float)(xr() % 1000) / 1000.f;
    for (int j = 0; j < d; j++) q[j] = 0.5f;
    for (long i = 0; i < n * M0; i++) adj[i] = xr() % n;
    for (long i = 0; i < n; i++) if (xr() % 20 == 0) bits[i >> 3] |= 1 << (i & 7);
    for (int r = 0; r < R; r++) ids[r] = xr() % n;

    double t0 = now(); float acc = 0;
    for (int r = 0; r < R; r++) {
        const float *x = X + (long)ids[r] * d; float s = 0;
        for (int j = 0; j < d; j++) { float t = x[j] - q[j]; s += t * t; }
        acc += s;
    }
    double t_dist = (now() - t0) / R;

    t0 = now(); long cnt = 0;
    for (int r = 0; r < R / M0 * 4; r++) {      /* expand random nodes: read list, test bits */
        const uint32_t *nb = adj + (long)ids[r] * M0;
        for (int j = 0; j < M0; j++) cnt += (bits[nb[j] >> 3] >> (nb[j] & 7)) & 1;
    }
    double t_edge = (now() - t0) / (R / M0 * 4 * (double)M0);
    printf("n=%ld d=%d  dist %.1f ns  edge %.2f ns  edge/dist = %.3f  (chk %g %ld)\n",
           n, d, t_dist * 1e9, t_edge * 1e9, t_edge / t_dist, acc, cnt);
    return 0;
}
