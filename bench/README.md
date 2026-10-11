# Native benchmarks

## `edge_vs_dist.c`
Cost of one adjacency scan vs one L2 distance (used to weight graph-search edge reads).
```bash
gcc -O3 -march=native bench/edge_vs_dist.c -o edge_vs_dist && ./edge_vs_dist 198000 100
```

## `acorn_bench.cpp` — official ACORN (SIGMOD'24) harness
```bash
git clone --depth 1 https://github.com/stanford-futuredata/ACORN.git acorn_src
# CMake export fix needed with recent CMake (nlohmann_json include path):
sed -i 's|target_include_directories(faiss PUBLIC ${nlohmann_json_SOURCE_DIR}/include)|target_include_directories(faiss PUBLIC $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/include>)|' acorn_src/CMakeLists.txt
cmake -S acorn_src -B acorn_src/build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=generic -DCMAKE_CXX_FLAGS="-O3 -march=native" \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so.3 -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so.3
make -C acorn_src/build -j4 faiss
g++ -O3 -march=native -fopenmp -I acorn_src -I acorn_src/build/_deps/nlohmann_json-src/include bench/acorn_bench.cpp \
  -L acorn_src/build/faiss -lfaiss -Wl,-rpath,$PWD/acorn_src/build/faiss -o acorn_bench
ACORN_BENCH=$PWD/acorn_bench python3 experiments/exp7_vs_acorn.py
```
Sanity check (no filter, 20k clustered points, M=32): recall@10 at ef 16/64/256 is
0.29/0.67/0.98 for ACORN-1 and 0.77/0.93/0.996 for ACORN-gamma (gamma=12) — ACORN-1
builds without neighbor pruning by design, so it needs a larger ef.
