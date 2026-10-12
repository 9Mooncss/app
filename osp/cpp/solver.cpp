// Fast simulated annealing for the Oven Scheduling Problem (OSP).
// Cost semantics identical to github.com/iolab-uniud/osp-ls (verified via osp/instance.py).
//
// Key idea: O(changed suffix) incremental evaluation with early re-synchronisation.
// Each machine keeps start/end times and prefix costs per batch position. A move is described,
// per affected machine, as an "edit" (p0, head[], shift): new positions p0..p0+|head|-1 are given
// by head[], and every later new position k maps to the unchanged old batch k - shift. The
// schedule is re-simulated from p0 and the simulation stops as soon as the state entering an
// unchanged batch equals the old state there (same previous end time, attribute, scheduled flag):
// the remaining cost is then read from the stored prefix sums. Nothing is mutated unless accepted.
//
// Input: text file written by osp/fast.py.  Output: solution (machine, batches) + stats.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace std;
typedef long long ll;

struct Inst {
    int H, A, M, S, N;
    vector<int> sc, stt, cap, init, as, ae, est, lst, mint, maxt, sz, attr;
    vector<vector<int>> elig_list;   // job -> eligible machines
    vector<vector<char>> elig;       // job x machine
    ll wr, wt, wsc, ub;
} I;

struct Batch {
    int attr = 0, size = 0, dur = 0, maxt = 1 << 30, est = 0;
    vector<int> jobs;      // job ids
    vector<int> lsts;      // sorted latest-end of jobs
    void recompute() {
        size = 0; dur = 0; maxt = 1 << 30; est = 0; lsts.clear();
        for (int j : jobs) {
            size += I.sz[j]; dur = max(dur, I.mint[j]); maxt = min(maxt, I.maxt[j]); est = max(est, I.est[j]);
            lsts.push_back(I.lst[j]);
        }
        sort(lsts.begin(), lsts.end());
        if (!jobs.empty()) attr = I.attr[jobs[0]];
    }
    int tardy(int end) const { return int(lower_bound(lsts.begin(), lsts.end(), end) - lsts.begin()); }
};

struct Machine {
    vector<Batch> b;
    vector<int> st, en;       // start / end per position
    vector<ll> pc;            // prefix cost including position p
    ll cost = 0;
};

static inline int batch_start(int m, int earliest, int setup, int dur, int prev_end) {
    int e = max(earliest, prev_end + setup);
    if (e + dur > I.H) return I.H + 1;
    const int *as = &I.as[m * I.S], *ae = &I.ae[m * I.S];
    for (int s = 0; s < I.S; s++) {
        int es = max(e, as[s] + setup);
        if (es - setup >= as[s] && es + dur <= ae[s]) return es;
    }
    return I.H + 1;
}

// cost of one batch given the state entering it; updates state
static inline ll step(int m, const Batch &B, int &prev_attr, int &prev_end, int &prev_start, int &start_out, int &end_out) {
    int st = I.stt[prev_attr * I.A + B.attr];
    int s;
    if (prev_start > I.H) s = I.H + 1;
    else s = batch_start(m, B.est, st, B.dur, prev_end);
    int e = s + B.dur;
    ll c;
    if (s <= I.H) c = I.wsc * I.sc[prev_attr * I.A + B.attr] + I.wr * B.dur + I.wt * B.tardy(e);
    else c = 2 * I.ub * (ll)B.jobs.size();
    prev_attr = B.attr; prev_end = e; prev_start = s; start_out = s; end_out = e;
    return c;
}

vector<Machine> MS;

static void full_recompute(int m) {
    Machine &X = MS[m];
    int n = X.b.size();
    X.st.assign(n, 0); X.en.assign(n, 0); X.pc.assign(n, 0);
    int pa = I.init[m], pe = 0, ps = 0; ll acc = 0;
    for (int p = 0; p < n; p++) { acc += step(m, X.b[p], pa, pe, ps, X.st[p], X.en[p]); X.pc[p] = acc; }
    X.cost = acc;
}

// Evaluate machine m with edit (p0, head, shift): returns new machine cost.
static ll eval_edit(int m, int p0, const vector<const Batch *> &head, int shift) {
    const Machine &X = MS[m];
    int oldn = X.b.size();
    int newn = p0 + (int)head.size() + max(0, oldn - (p0 + (int)head.size() - shift) );
    // number of tail elements: old indices from (p0+|head|-shift) to oldn-1
    int tail_from_old = p0 + (int)head.size() - shift;
    if (tail_from_old > oldn) tail_from_old = oldn;
    newn = p0 + (int)head.size() + (oldn - tail_from_old);
    int pa, pe, ps; ll acc;
    if (p0 == 0) { pa = I.init[m]; pe = 0; ps = 0; acc = 0; }
    else { pa = X.b[p0 - 1].attr; pe = X.en[p0 - 1]; ps = X.st[p0 - 1]; acc = X.pc[p0 - 1]; }
    int s, e;
    for (int k = p0; k < newn; k++) {
        int hk = k - p0;
        if (hk < (int)head.size()) { acc += step(m, *head[hk], pa, pe, ps, s, e); continue; }
        int ko = k - shift;   // old index of this unchanged batch
        // re-synchronisation: state entering old batch ko equals current state?
        int opa, ope, ops;
        if (ko == 0) { opa = I.init[m]; ope = 0; ops = 0; }
        else { opa = X.b[ko - 1].attr; ope = X.en[ko - 1]; ops = X.st[ko - 1]; }
        bool osched = ops <= I.H, nsched = ps <= I.H;
        if (opa == pa && ope == pe && osched == nsched) {
            return acc + (X.cost - (ko == 0 ? 0 : X.pc[ko - 1]));
        }
        acc += step(m, X.b[ko], pa, pe, ps, s, e);
    }
    return acc;
}

// ---------------------------------------------------------------- moves
mt19937_64 rng(1);
static inline int rnd(int n) { return int(rng() % (unsigned long long)n); }
static inline double urand() { return (rng() >> 11) * (1.0 / 9007199254740992.0); }

vector<pair<int, int>> where;   // job -> (machine, position)  (kept exact after each accepted move)
Batch T1, T2;                    // scratch batches

static void rebuild_where(int m) { for (int p = 0; p < (int)MS[m].b.size(); p++) for (int j : MS[m].b[p].jobs) where[j] = {m, p}; }

static bool compatible_add(const Batch &B, int j, int m) {
    if (B.attr != I.attr[j] || !I.elig[j][m]) return false;
    if (B.size + I.sz[j] > I.cap[m]) return false;
    int d = max(B.dur, I.mint[j]), mx = min(B.maxt, I.maxt[j]);
    return d <= mx;
}

static void make_without(const Batch &B, int j, Batch &out) {
    out.jobs.clear();
    for (int x : B.jobs) if (x != j) out.jobs.push_back(x);
    out.recompute();
}
static void make_with(const Batch &B, int j, Batch &out) { out.jobs = B.jobs; out.jobs.push_back(j); out.recompute(); }

ll total_cost() { ll c = 0; for (auto &X : MS) c += X.cost; return c; }

struct Edit { int m, p0, shift; vector<const Batch *> head; };
vector<Edit> edits;   // edits of the current candidate move

// apply: rebuild affected machines from edits (materialise new batch vectors)
static void apply_edits() {
    // materialise every head batch first: heads may point into machines that are rebuilt below
    vector<vector<Batch>> heads(edits.size());
    for (size_t i = 0; i < edits.size(); i++) for (auto *h : edits[i].head) heads[i].push_back(*h);
    for (size_t ei = 0; ei < edits.size(); ei++) {
        auto &E = edits[ei];
        Machine &X = MS[E.m];
        int oldn = X.b.size();
        int tail_from_old = min(oldn, E.p0 + (int)E.head.size() - E.shift);
        vector<Batch> nb;
        nb.reserve(E.p0 + E.head.size() + oldn - tail_from_old);
        for (int k = 0; k < E.p0; k++) nb.push_back(move(X.b[k]));
        for (auto &h : heads[ei]) nb.push_back(move(h));
        for (int k = tail_from_old; k < oldn; k++) nb.push_back(move(X.b[k]));
        X.b.swap(nb);
    }
    for (auto &E : edits) { full_recompute(E.m); rebuild_where(E.m); }
}

// candidate generators fill `edits` and return true if a valid candidate was produced
// pointers in heads may refer to MS batches (unchanged) or scratch T1/T2.
static vector<const Batch *> H1, H2;

static bool mv_job_to_existing() {
    int j = rnd(I.N);
    auto [m1, p1] = where[j];
    const auto &el = I.elig_list[j];
    int m2 = el[rnd(el.size())];
    int nb2 = MS[m2].b.size();
    if (nb2 == 0) return false;
    int p2 = rnd(nb2);
    if (m1 == m2 && p1 == p2) return false;
    const Batch &B2 = MS[m2].b[p2];
    if (!compatible_add(B2, j, m2)) return false;
    const Batch &B1 = MS[m1].b[p1];
    make_with(B2, j, T2);
    edits.clear();
    bool empty1 = B1.jobs.size() == 1;
    if (!empty1) make_without(B1, j, T1);
    if (m1 != m2) {
        Edit e1{m1, p1, empty1 ? -1 : 0, {}};
        if (!empty1) e1.head.push_back(&T1);
        Edit e2{m2, p2, 0, {&T2}};
        edits.push_back(e1); edits.push_back(e2);
    } else {
        int lo = min(p1, p2), hi = max(p1, p2);
        Edit e{m1, lo, 0, {}};
        for (int k = lo; k <= hi; k++) {
            if (k == p1) { if (!empty1) e.head.push_back(&T1); }
            else if (k == p2) e.head.push_back(&T2);
            else e.head.push_back(&MS[m1].b[k]);
        }
        e.shift = empty1 ? -1 : 0;
        edits.push_back(e);
    }
    return true;
}

static bool mv_job_to_new() {
    int j = rnd(I.N);
    auto [m1, p1] = where[j];
    const Batch &B1 = MS[m1].b[p1];
    if (B1.jobs.size() == 1 && rnd(2)) return false;   // moving a singleton = batch move; allowed half the time
    const auto &el = I.elig_list[j];
    int m2 = el[rnd(el.size())];
    T2.jobs.assign(1, j); T2.recompute();
    bool empty1 = B1.jobs.size() == 1;
    if (!empty1) make_without(B1, j, T1);
    edits.clear();
    if (m1 != m2) {
        int q = rnd(MS[m2].b.size() + 1);    // insert before old position q
        Edit e1{m1, p1, empty1 ? -1 : 0, {}};
        if (!empty1) e1.head.push_back(&T1);
        Edit e2{m2, q, 1, {&T2}};
        edits.push_back(e1); edits.push_back(e2);
    } else {
        int n = MS[m1].b.size();
        int q = rnd(n + 1);                  // insert before old position q (old indexing)
        int lo = min(p1, q), hi = max(p1, q == n ? n - 1 : q);
        Edit e{m1, lo, 0, {}};
        for (int k = lo; k <= hi; k++) {
            if (k == q) e.head.push_back(&T2);
            if (k == p1) { if (!empty1) e.head.push_back(&T1); }
            else e.head.push_back(&MS[m1].b[k]);
        }
        if (q == n) e.head.push_back(&T2);
        int newlen_block = e.head.size(), oldlen_block = hi - lo + 1;
        e.shift = newlen_block - oldlen_block;
        edits.push_back(e);
    }
    return true;
}

static bool mv_swap_batches() {
    int m = rnd(I.M), n = MS[m].b.size();
    if (n < 2) return false;
    int p1 = rnd(n), p2 = rnd(n);
    if (p1 == p2) return false;
    if (p1 > p2) swap(p1, p2);
    if (p2 - p1 > 1 && rnd(2)) p2 = p1 + 1;   // favour adjacent swaps
    Edit e{m, p1, 0, {}};
    for (int k = p1; k <= p2; k++) e.head.push_back(&MS[m].b[k == p1 ? p2 : (k == p2 ? p1 : k)]);
    edits.assign(1, e);
    return true;
}

static bool mv_batch_insert() {
    int m = rnd(I.M), n = MS[m].b.size();
    if (n < 2) return false;
    int p = rnd(n), q = rnd(n);
    if (p == q) return false;
    int lo = min(p, q), hi = max(p, q);
    Edit e{m, lo, 0, {}};
    if (p < q) { for (int k = p + 1; k <= q; k++) e.head.push_back(&MS[m].b[k]); e.head.push_back(&MS[m].b[p]); }
    else { e.head.push_back(&MS[m].b[p]); for (int k = q; k < p; k++) e.head.push_back(&MS[m].b[k]); }
    edits.assign(1, e);
    return true;
}

static bool mv_batch_to_machine() {
    int m1 = rnd(I.M), n1 = MS[m1].b.size();
    if (!n1) return false;
    int p1 = rnd(n1);
    const Batch &B = MS[m1].b[p1];
    int m2 = rnd(I.M);
    if (m2 == m1 || B.size > I.cap[m2]) return false;
    for (int j : B.jobs) if (!I.elig[j][m2]) return false;
    int q = rnd(MS[m2].b.size() + 1);
    Edit e1{m1, p1, -1, {}};
    Edit e2{m2, q, 1, {&B}};
    edits.clear(); edits.push_back(e1); edits.push_back(e2);
    return true;
}

static bool mv_swap_jobs() {
    int j1 = rnd(I.N), j2 = rnd(I.N);
    if (j1 == j2 || I.attr[j1] != I.attr[j2]) return false;
    auto [m1, p1] = where[j1]; auto [m2, p2] = where[j2];
    if (m1 == m2 && p1 == p2) return false;
    const Batch &B1 = MS[m1].b[p1], &B2 = MS[m2].b[p2];
    T1.jobs = B1.jobs; for (int &x : T1.jobs) if (x == j1) x = j2; T1.recompute();
    T2.jobs = B2.jobs; for (int &x : T2.jobs) if (x == j2) x = j1; T2.recompute();
    if (!I.elig[j2][m1] || !I.elig[j1][m2]) return false;
    if (T1.size > I.cap[m1] || T2.size > I.cap[m2] || T1.dur > T1.maxt || T2.dur > T2.maxt) return false;
    edits.clear();
    if (m1 != m2) { edits.push_back({m1, p1, 0, {&T1}}); edits.push_back({m2, p2, 0, {&T2}}); }
    else {
        int lo = min(p1, p2), hi = max(p1, p2);
        Edit e{m1, lo, 0, {}};
        for (int k = lo; k <= hi; k++) e.head.push_back(k == p1 ? &T1 : (k == p2 ? &T2 : &MS[m1].b[k]));
        edits.push_back(e);
    }
    return true;
}

// ---------------------------------------------------------------- initial solution
static void initial_solution() {
    MS.assign(I.M, Machine());
    vector<int> order(I.N);
    for (int j = 0; j < I.N; j++) order[j] = j;
    sort(order.begin(), order.end(), [](int a, int b) { return I.lst[a] < I.lst[b] || (I.lst[a] == I.lst[b] && I.est[a] < I.est[b]); });
    where.assign(I.N, {0, 0});
    for (int j : order) {
        // try to join the last compatible batch on an eligible machine; else open a new batch at the end
        ll best = LLONG_MAX; int bm = -1, bp = -1;
        for (int m : I.elig_list[j]) {
            Machine &X = MS[m];
            ll before = X.cost;
            for (int p = max(0, (int)X.b.size() - 3); p < (int)X.b.size(); p++) {
                if (!compatible_add(X.b[p], j, m)) continue;
                make_with(X.b[p], j, T2);
                vector<const Batch *> h{&T2};
                ll c = eval_edit(m, p, h, 0) - before;
                if (c < best) { best = c; bm = m; bp = p; }
            }
            T2.jobs.assign(1, j); T2.recompute();
            vector<const Batch *> h{&T2};
            ll c = eval_edit(m, X.b.size(), h, 1) - before;
            if (c < best) { best = c; bm = m; bp = -1; }
        }
        Machine &X = MS[bm];
        if (bp >= 0) { X.b[bp].jobs.push_back(j); X.b[bp].recompute(); }
        else { Batch nb; nb.jobs.push_back(j); nb.recompute(); X.b.push_back(nb); }
        full_recompute(bm);
    }
    for (int m = 0; m < I.M; m++) rebuild_where(m);
}

// ---------------------------------------------------------------- I/O
static void read_instance(const char *path) {
    FILE *f = fopen(path, "r");
    auto rd = [&]() { int x; if (fscanf(f, "%d", &x) != 1) { fprintf(stderr, "bad input\n"); exit(1); } return x; };
    auto rdl = [&]() { ll x; if (fscanf(f, "%lld", &x) != 1) { fprintf(stderr, "bad input\n"); exit(1); } return x; };
    I.H = rd(); I.A = rd(); I.M = rd(); I.S = rd(); I.N = rd();
    I.sc.resize(I.A * I.A); for (auto &x : I.sc) x = rd();
    I.stt.resize(I.A * I.A); for (auto &x : I.stt) x = rd();
    I.cap.resize(I.M); for (auto &x : I.cap) x = rd();
    I.init.resize(I.M); for (auto &x : I.init) x = rd();
    I.as.resize(I.M * I.S); for (auto &x : I.as) x = rd();
    I.ae.resize(I.M * I.S); for (auto &x : I.ae) x = rd();
    for (auto *v : {&I.est, &I.lst, &I.mint, &I.maxt, &I.sz, &I.attr}) { v->resize(I.N); for (auto &x : *v) x = rd(); }
    I.elig.assign(I.N, vector<char>(I.M, 0)); I.elig_list.assign(I.N, {});
    for (int j = 0; j < I.N; j++) { int k = rd(); for (int t = 0; t < k; t++) { int m = rd(); I.elig[j][m] = 1; I.elig_list[j].push_back(m); } }
    I.wr = rdl(); I.wt = rdl(); I.wsc = rdl(); I.ub = rdl();
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: solver instance.txt time_limit_s seed [T0_factor Tend_factor rates...]\n"); return 1; }
    read_instance(argv[1]);
    double tlim = atof(argv[2]);
    rng.seed(atoll(argv[3]));
    double t0f = argc > 4 ? atof(argv[4]) : 0.02, tef = argc > 5 ? atof(argv[5]) : 0.00002;
    double rates[6] = {0.30, 0.15, 0.15, 0.10, 0.10, 0.20};   // JE, JN, swapB, insertB, B->machine, swapJ
    for (int i = 0; i < 6 && 6 + i < argc; i++) rates[i] = atof(argv[6 + i]);
    double cum[6]; double s = 0; for (int i = 0; i < 6; i++) { s += rates[i]; cum[i] = s; }

    auto tstart = chrono::steady_clock::now();
    auto elapsed = [&]() { return chrono::duration<double>(chrono::steady_clock::now() - tstart).count(); };
    initial_solution();
    ll cur = total_cost(), best = cur;
    double t_init = elapsed();
    vector<Machine> bestMS; bool best_saved = false;
    // temperature relative to the instance's upper bound (costs are integers up to ~UB)
    double T0 = t0f * I.ub, Tend = tef * I.ub;
    ll it = 0, acc = 0, valid = 0;
    double T = T0, frac = 0;
    while (true) {
        if ((it & 1023) == 0) {
            frac = elapsed() / tlim;
            if (frac >= 1.0) break;
            T = T0 * pow(Tend / T0, frac);
        }
        it++;
        double r = urand() * s; bool ok;
        if (r < cum[0]) ok = mv_job_to_existing();
        else if (r < cum[1]) ok = mv_job_to_new();
        else if (r < cum[2]) ok = mv_swap_batches();
        else if (r < cum[3]) ok = mv_batch_insert();
        else if (r < cum[4]) ok = mv_batch_to_machine();
        else ok = mv_swap_jobs();
        if (!ok) continue;
        valid++;
        ll delta = 0;
        for (auto &E : edits) delta += eval_edit(E.m, E.p0, E.head, E.shift) - MS[E.m].cost;
        if (delta <= 0 || urand() < exp(-double(delta) / T)) {
            if (delta > 0 && cur == best && !best_saved) { bestMS = MS; best_saved = true; }   // leaving a best state
            apply_edits();
            cur += delta; acc++;
            if (cur < best) { best = cur; best_saved = false; }
        }
    }
    if (best_saved && cur != best) MS = bestMS;
    ll check = 0; for (int m = 0; m < I.M; m++) { full_recompute(m); check += MS[m].cost; }
    printf("{\"cost\": %lld, \"normalized\": %.9f, \"iterations\": %lld, \"valid_moves\": %lld, \"accepted\": %lld, \"time\": %.3f, \"init_time\": %.4f, \"check\": %lld}\n",
           best, double(best) / I.ub, it, valid, acc, elapsed(), t_init, check);
    for (int m = 0; m < I.M; m++) {
        printf("M %d", m);
        for (auto &B : MS[m].b) { printf(" |"); for (int j : B.jobs) printf(" %d", j); }
        printf("\n");
    }
    return 0;
}
