// Benchmark harness for the official ACORN implementation (stanford-futuredata/ACORN,
// a FAISS fork). Builds IndexACORNFlat, then searches each query alone with its own
// filter bitmap, single thread, timing every query in C++, for several efSearch values.
// Input files (raw little-endian): base f32[n][d], queries f32[nq][d], filters u8[nq][n],
// exact i64[nq][k], labels i32[n].  Output CSV: ef,query,time_us,recall
#include <faiss/IndexACORN.h>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

template <class T> std::vector<T> load(const char* path, size_t count) {
    std::vector<T> v(count);
    FILE* f = fopen(path, "rb");
    if (!f || fread(v.data(), sizeof(T), count, f) != count) { fprintf(stderr, "read %s failed\n", path); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 15) { fprintf(stderr, "usage: base q filters exact labels n d nq k M gamma Mbeta out ef1 [ef2 ...]\n"); return 1; }
    size_t n = atol(argv[6]), d = atol(argv[7]), nq = atol(argv[8]), k = atol(argv[9]);
    int M = atoi(argv[10]), gamma = atoi(argv[11]), Mb = atoi(argv[12]);
    auto X = load<float>(argv[1], n * d);
    auto Q = load<float>(argv[2], nq * d);
    auto F = load<char>(argv[3], nq * n);
    auto E = load<int64_t>(argv[4], nq * k);
    auto L = load<int32_t>(argv[5], n);
    std::vector<int> meta(L.begin(), L.end());
    omp_set_num_threads(4);
    auto t0 = std::chrono::steady_clock::now();
    faiss::IndexACORNFlat index(d, M, gamma, meta, Mb);
    index.add(n, X.data());
    double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "build %.1fs\n", build);
    omp_set_num_threads(1);
    std::ofstream out(argv[13]);
    out << "ef,query,time_us,recall\n";
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);
    for (int a = 14; a < argc; a++) {
        int ef = atoi(argv[a]);
        index.acorn.efSearch = ef;
        for (size_t qi = 0; qi < nq; qi++) {
            double best = 1e30;
            for (int rep = 0; rep < 2; rep++) {
                auto s = std::chrono::steady_clock::now();
                index.search(1, Q.data() + qi * d, k, D.data(), I.data(), F.data() + qi * n);
                double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s).count();
                if (us < best) best = us;
            }
            std::set<int64_t> ex(E.begin() + qi * k, E.begin() + (qi + 1) * k);
            int hit = 0;
            for (size_t j = 0; j < k; j++) hit += ex.count(I[j]);
            out << ef << "," << qi << "," << best << "," << (double)hit / k << "\n";
        }
        fprintf(stderr, "ef %d done\n", ef);
    }
    return 0;
}
