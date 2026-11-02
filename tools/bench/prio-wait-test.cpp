// Lab test of PrioWait::Blocks (src/local/prio-wait.h): synthetic priority-operation queues, a readback window,
// with and without range awareness. Invariant: an operation that may write the window and whose tick is <= the
// readback's is always waited for (no reordering of an overlapping write); the range-aware wait never blocks
// longer than the old one.
#include "../../src/local/prio-wait.h"
#include <cstdio>
#include <deque>
#include <random>
using namespace PrioWait;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
int main() {
	const uint64_t W0 = 0x220800000, W1 = W0 + 512 * 1024;
	std::deque<Op> q;
	Op none {};
	q = {{5, None}, {6, None}, {9, None}};                       // flips / interrupts only
	CHECK(Blocks(q, false, none, 8, W0, W1, false));
	CHECK(!Blocks(q, false, none, 8, W0, W1, true));
	q = {{5, None}, {7, Range, W0 + 0x40, W0 + 0x44}, {9, None}}; // an EOP-style label write inside the window
	CHECK(Blocks(q, false, none, 8, W0, W1, false));
	CHECK(Blocks(q, false, none, 8, W0, W1, true));
	q = {{7, Range, W1 + 0x1000, W1 + 0x1004}};                   // outside the window
	CHECK(!Blocks(q, false, none, 8, W0, W1, true));
	q = {{7, Range, W1, W1 + 4}};                                 // edges
	CHECK(!Blocks(q, false, none, 8, W0, W1, true));
	q = {{7, Range, W1 - 4, W1}};
	CHECK(Blocks(q, false, none, 8, W0, W1, true));
	q = {{7, Range, W0 - 4, W0 + 1}};
	CHECK(Blocks(q, false, none, 8, W0, W1, true));
	q = {{3, Unknown}};                                           // undeclared: conservative
	CHECK(Blocks(q, false, none, 8, W0, W1, true));
	q = {{9, Unknown}, {10, Range, W0, W1}};                      // later ticks never block
	CHECK(!Blocks(q, false, none, 8, W0, W1, false));
	CHECK(!Blocks(q, false, none, 8, W0, W1, true));
	q = {};                                                       // the active op
	CHECK(Blocks(q, true, Op {7, Range, W0, W0 + 16}, 8, W0, W1, true));
	CHECK(!Blocks(q, true, Op {7, None}, 8, W0, W1, true));
	CHECK(Blocks(q, true, Op {7, None}, 8, W0, W1, false));
	CHECK(Blocks(q, true, Op {7, Unknown}, 8, W0, W1, true));
	std::mt19937_64 rng(42);
	long rounds = 0;
	for (int round = 0; round < 500000; ++round, ++rounds) {
		q.clear();
		uint64_t t = 1;
		const int n = static_cast<int>(rng() % 10);
		for (int i = 0; i < n; ++i) {
			t += rng() % 3;
			Op op {t, static_cast<uint8_t>(rng() % 3)};
			if (op.cls == Range) {
				op.begin = W0 - 0x10000 + (rng() % 0xa0000);
				op.end   = op.begin + 1 + rng() % 0x20000;
			}
			q.push_back(op);
		}
		const bool active = rng() % 2;
		Op         aop {1 + rng() % 12, static_cast<uint8_t>(rng() % 3), W0 - 0x8000 + (rng() % 0x90000), 0};
		aop.end           = aop.begin + 4;
		const uint64_t tick = rng() % 16;
		bool must = active && aop.tick <= tick && MayWrite(aop, W0, W1);
		for (const auto& op: q)
			if (op.tick <= tick && MayWrite(op, W0, W1)) must = true;
		const bool aware = Blocks(q, active, aop, tick, W0, W1, true);
		const bool old   = Blocks(q, active, aop, tick, W0, W1, false);
		CHECK(aware == must); // waits for every op that may write the window (and only those)
		CHECK(!aware || old); // never longer than before
		if (fails > 5) break;
	}
	std::printf(fails == 0 ? "prio-wait-test: all checks passed (%ld random queues)\n" : "prio-wait-test: FAILURES\n", rounds);
	return fails == 0 ? 0 : 1;
}
