#include "graphics/host_gpu/renderer/image/groupNotices.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::GroupNotices;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "GroupNoticesTests: failed: %s\n", message);
		std::abort();
	}
}

std::vector<std::pair<uint32_t, GroupNotices::Kind>> Read(const GroupNotices& notices, uint64_t from, uint64_t to, bool expected = true) {
	std::vector<std::pair<uint32_t, GroupNotices::Kind>> seen;
	const bool ok = notices.Since(from, to, [&](uint32_t image, GroupNotices::Kind kind) { seen.emplace_back(image, kind); });
	Check(ok == expected, "Since's answer");
	return seen;
}

} // namespace

int main() {
	using Kind = GroupNotices::Kind;
	constexpr uint64_t Capacity = uint64_t {1} << 16u;
	auto notices = std::make_unique<GroupNotices>();
	Check(notices->End() == 0u, "starts empty");
	Check(Read(*notices, 0u, 0u).empty(), "nothing to read");

	// Each reader has its own position, and sees each entry in order, with its kind.
	notices->Add(7u, Kind::Contents);
	notices->Add(9u, Kind::State);
	const auto middle = notices->End();
	Check(middle == 2u, "two entries");
	notices->Add(7u, Kind::State);
	const auto all = Read(*notices, 0u, notices->End());
	Check(all.size() == 3u && all[0] == std::pair {7u, Kind::Contents} && all[1] == std::pair {9u, Kind::State} && all[2] == std::pair {7u, Kind::State},
	      "in order");
	const auto rest = Read(*notices, middle, notices->End());
	Check(rest.size() == 1u && rest[0].first == 7u, "a later position sees only what came after it");

	// An entry added after `to` is left for the next read, which is how a walk's own changes reach the next walk.
	const auto to = notices->End();
	notices->Add(11u, Kind::Contents);
	Check(Read(*notices, middle, to).size() == 1u, "up to `to` only");
	Check(Read(*notices, to, notices->End()).size() == 1u, "the entry after it is still there");

	// A reader that fell behind by more than the list holds is told so, and walks everything.
	const auto old = notices->End();
	for (uint64_t i = 0; i < Capacity; ++i) notices->Add(static_cast<uint32_t>(i), Kind::Contents);
	const auto exact = Read(*notices, old, notices->End());
	Check(exact.size() == Capacity && exact.front().first == 0u && exact.back().first == Capacity - 1u, "exactly as many as it holds still reads");
	notices->Add(1u, Kind::State);
	Check(Read(*notices, old, notices->End(), false).empty(), "one more and the oldest is gone");
	Check(Read(*notices, old + 1u, notices->End()).size() == Capacity, "the next position still reads");

	// Page faults add from other threads: nothing is lost.
	auto shared = std::make_unique<GroupNotices>();
	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < 4u; ++t) {
		threads.emplace_back([&shared, t] {
			for (uint32_t i = 0; i < 1000u; ++i) shared->Add(t, Kind::Contents);
		});
	}
	for (auto& thread: threads) thread.join();
	Check(shared->End() == 4000u, "every add counted");
	std::array<uint32_t, 4> counts {};
	Check(shared->Since(0u, shared->End(), [&](uint32_t image, Kind) { counts[image]++; }), "reads");
	for (const auto count: counts) Check(count == 1000u, "each thread's entries");

	std::puts("GroupNoticesTests: ok");
	return 0;
}
