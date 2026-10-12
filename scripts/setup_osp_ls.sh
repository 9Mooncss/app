#!/usr/bin/env bash
# Fetch the OSP benchmark + reference solver (github.com/iolab-uniud/osp-ls) into external/osp-ls,
# apply three small fixes needed to build it, and compile it (needs cmake, g++, libboost-program-options-dev).
#   1) disable the missing local-search-damodaran subdirectory
#   2) fix a syntax error in main.cc (stray "||")
#   3) re-enable printing of the solution (used to cross-check our evaluator)
set -euo pipefail
DEST="${1:-external/osp-ls}"
if [ ! -d "$DEST" ]; then git clone -q --depth 1 https://github.com/iolab-uniud/osp-ls.git "$DEST"; fi
cd "$DEST"
sed -i 's|^include_directories(${PROJECT_SOURCE_DIR}/local-search-damodaran)|# &|; s|^add_subdirectory(local-search-damodaran)|# &|' CMakeLists.txt
sed -i 's/method == std::string("LAHC_all") ||)/method == std::string("LAHC_all"))/' local-search/main.cc
sed -i 's|                // << "{\\"solution\\": {" << out <<  "}, "|                << "{\\"solution\\": {" << out <<  "}}\\n"|' local-search/main.cc
if [ ! -x bin/osp ]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
  make -C build -j"$(nproc)" > /dev/null
fi
ls -la bin/osp
