#include "common/packetStream.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PacketStreamTests: failed: %s\n", message);
		std::abort();
	}
}

// What the consumer saw: each packet's number, and whether its payload held the bytes the producer wrote.
struct Seen {
	std::vector<uint32_t> order;
	uint64_t              bad_payloads = 0;
	std::atomic<bool>     gate {true}; // a packet with number 0xffffffff waits for it
	uint32_t              slow_us = 0;
	uint64_t              mismatches = 0;
};

// Payload: number, size, then size - 8 bytes of (number + i) & 0xff.
void Run(const std::byte* data, void* context) {
	auto&    seen = *static_cast<Seen*>(context);
	uint32_t number, size;
	std::memcpy(&number, data, 4);
	std::memcpy(&size, data + 4, 4);
	if (number == 0xffffffffu) {
		while (!seen.gate.load()) std::this_thread::yield();
		return;
	}
	for (uint32_t i = 8; i < size; i++) {
		if (static_cast<uint8_t>(data[i]) != static_cast<uint8_t>(number + i)) {
			seen.bad_payloads++;
			break;
		}
	}
	seen.order.push_back(number);
	if (seen.slow_us != 0) std::this_thread::sleep_for(std::chrono::microseconds(seen.slow_us));
}

std::byte* Put(Common::PacketStream& stream, uint32_t number, uint32_t size) {
	auto* data = stream.Append(&Run, size);
	std::memcpy(data, &number, 4);
	std::memcpy(data + 4, &size, 4);
	for (uint32_t i = 8; i < size; i++) data[i] = static_cast<std::byte>(number + i);
	return data;
}

// Sizes from tiny to past the spill limit, so that packets cross blocks and spill.
uint32_t SizeOf(uint32_t number) {
	static const uint32_t sizes[] = {8, 24, 100, 1000, 4096, 16 * 1024, 16 * 1024 + 8, 40000, 300000, 64};
	return sizes[number % (sizeof(sizes) / sizeof(sizes[0]))];
}

void TestOrderAndContents() {
	Seen seen;
	{
		Common::PacketStream stream(&seen, true, 8);
		stream.Start();
		constexpr uint32_t count = 5000;
		for (uint32_t i = 0; i < count; i++) {
			Put(stream, i, SizeOf(i));
			if (i % 37 == 0) stream.Publish();
		}
		stream.Drain();
		Check(stream.Executed() == count, "drain runs every packet");
		Check(seen.order.size() == count, "every packet ran once");
		for (uint32_t i = 0; i < count; i++) Check(seen.order[i] == i, "packets run in order");
		Check(seen.bad_payloads == 0, "payloads arrive as written");
		Check(stream.Mismatches() == 0, "no verify mismatch");
	}
}

void TestBackpressure() {
	Seen seen;
	seen.slow_us = 20;
	Common::PacketStream stream(&seen, false, 2);
	stream.Start();
	constexpr uint32_t count = 3000;
	for (uint32_t i = 0; i < count; i++) Put(stream, i, 1000); // ~3 MB through two blocks of 64 KB
	stream.Stop();
	Check(seen.order.size() == count, "stop runs everything queued");
	for (uint32_t i = 0; i < count; i++) Check(seen.order[i] == i, "order under backpressure");
	Check(stream.BackpressureWaits() > 0, "the producer waited for blocks");
}

void TestVerifyCatchesChange() {
	Seen seen;
	seen.gate.store(false);
	Common::PacketStream stream(
	    &seen, true, 8, [](void* context, Common::PacketStream::Run, size_t) { static_cast<Seen*>(context)->mismatches++; });
	stream.Start();
	Put(stream, 0xffffffffu, 8); // holds the consumer
	auto* data = Put(stream, 7, 64);
	Put(stream, 8, 64);
	stream.Publish();
	data[20] = std::byte {0x5a}; // changed after it was queued and published, before it runs
	seen.gate.store(true);
	stream.Drain();
	Check(stream.Mismatches() == 1 && seen.mismatches == 1, "verify finds the changed packet");
	Check(seen.bad_payloads == 1, "the change reached the run (the test changed what it checks)");
}

void TestNothingRunsUnpublished() {
	Seen seen;
	Common::PacketStream stream(&seen, false, 8);
	stream.Start();
	Put(stream, 1, 16);
	Put(stream, 2, 16);
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	Check(stream.Executed() == 0, "nothing runs before it is published");
	stream.Publish();
	stream.WaitExecuted(2);
	Check(seen.order.size() == 2, "published packets run");
}

} // namespace

int main() {
	TestOrderAndContents();
	TestBackpressure();
	TestVerifyCatchesChange();
	TestNothingRunsUnpublished();
	std::printf("PacketStreamTests: ok\n");
	return 0;
}
