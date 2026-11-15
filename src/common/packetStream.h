#ifndef EMULATOR_SRC_COMMON_PACKETSTREAM_H_
#define EMULATOR_SRC_COMMON_PACKETSTREAM_H_

// An ordered stream of packets: one producer thread appends them, one consumer thread runs them in the order
// they were appended (the Vulkan recording thread's queue, record-thread design H1). A packet is a run function
// and a payload of plain bytes that the producer writes once and the consumer only reads. Packets go in blocks of
// BlockBytes; the consumer recycles a block once it has run every packet in it. A payload over SpillBytes lives in
// an allocation of its own, which the consumer frees after running it.
//
// Nothing runs before the producer publishes it (Publish, a full block, Drain). With verify on, the producer
// hashes each payload once it is written, and the consumer hashes it again just before running it: a difference
// means the bytes changed between the two (a payload pointing at memory the producer reuses, a recycled block).

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Common {

class PacketStream {
public:
	using Run      = void (*)(const std::byte* payload, void* context);
	using Mismatch = void (*)(void* context, Run run, size_t bytes);

	static constexpr size_t BlockBytes = size_t {64} << 10u;
	static constexpr size_t SpillBytes = size_t {16} << 10u;
	static constexpr size_t Align      = 16;

	// context goes to every run function; max_blocks bounds the blocks appended but not yet run (backpressure).
	PacketStream(void* context, bool verify, size_t max_blocks, Mismatch on_mismatch = nullptr)
	    : m_context(context), m_verify(verify), m_max_blocks(std::max<size_t>(max_blocks, 2)),
	      m_on_mismatch(on_mismatch) {
		m_tail = m_head = new Block;
		m_blocks        = 1;
	}
	~PacketStream() {
		Stop();
		for (auto* block = m_head; block != nullptr;) {
			auto* next = block->next.load(std::memory_order_relaxed);
			delete block;
			block = next;
		}
		for (auto* block: m_free) delete block;
	}
	PacketStream(const PacketStream&)            = delete;
	PacketStream& operator=(const PacketStream&) = delete;

	void Start() {
		m_thread = std::jthread([this](std::stop_token stop) { Consume(stop); });
	}
	// Runs everything appended, then ends the consumer thread.
	void Stop() {
		if (!m_thread.joinable()) return;
		Drain();
		m_thread.request_stop();
		Wake();
		m_thread.join();
	}

	// Producer: a payload of `bytes` (16-aligned, uninitialized) that `run` gets. It is the producer's to fill until
	// the next Append, Publish or Drain.
	[[nodiscard]] std::byte* Append(Run run, size_t bytes) {
		CheckProducer();
		Seal();
		const bool spill   = bytes > SpillBytes;
		const auto payload = spill ? sizeof(Spill) : bytes;
		const auto size    = AlignUp(sizeof(Header) + payload);
		if (m_used + size > BlockBytes) NextBlock();
		auto* header  = reinterpret_cast<Header*>(m_tail->data + m_used);
		header->run   = run;
		header->size  = static_cast<uint32_t>(size);
		header->spill = spill ? 1u : 0u;
		header->check = 0;
		m_used += size;
		m_appended++;
		m_open = header;
		auto* data = reinterpret_cast<std::byte*>(header + 1);
		if (!spill) return data;
		auto* own = new (std::align_val_t {Align}) std::byte[AlignUp(bytes)];
		new (data) Spill {own, bytes};
		return own;
	}
	// Producer: lets the consumer run everything appended so far.
	void Publish() {
		CheckProducer();
		Seal();
		m_tail->committed.store(m_used, std::memory_order_release);
		if (m_published.load(std::memory_order_relaxed) == m_appended) return;
		m_published.store(m_appended, std::memory_order_seq_cst);
		if (m_sleeping.load(std::memory_order_seq_cst)) Wake();
	}
	// Producer: publishes, then waits until the consumer has run everything appended.
	void Drain() {
		Publish();
		WaitExecuted(m_appended);
	}
	// Producer: waits until the consumer has run the first `count` packets (count <= Published()).
	void WaitExecuted(uint64_t count) {
		if (m_executed.load(std::memory_order_acquire) >= count) return;
		m_waiters.fetch_add(1, std::memory_order_seq_cst);
		for (auto seen = m_executed.load(std::memory_order_seq_cst); seen < count;
		     seen      = m_executed.load(std::memory_order_seq_cst)) {
			m_executed.wait(seen, std::memory_order_seq_cst);
		}
		m_waiters.fetch_sub(1, std::memory_order_seq_cst);
	}

	[[nodiscard]] uint64_t Appended() const noexcept { return m_appended; }
	[[nodiscard]] uint64_t Executed() const noexcept { return m_executed.load(std::memory_order_acquire); }
	[[nodiscard]] uint64_t Mismatches() const noexcept { return m_mismatches.load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t BackpressureWaits() const noexcept { return m_backpressure_waits; }
	[[nodiscard]] bool     OnConsumer() const noexcept { return std::this_thread::get_id() == m_consumer_id; }
	// Appends and publishes from another thread than the producer (verify only): a race on the producer side.
	[[nodiscard]] uint64_t ForeignAppends() const noexcept { return m_foreign; }
	// The calling thread is the producer from now on (an owner that hands over, at start-up and at shutdown).
	void Adopt() { m_producer_id = std::this_thread::get_id(); }

private:
	struct Header {
		Run      run;
		uint32_t size;  // bytes to the next header
		uint32_t spill; // 1: the payload is a Spill
		uint64_t check; // the payload's hash, with verify on
		uint64_t pad;
	};
	static_assert(sizeof(Header) % Align == 0);
	struct Spill {
		std::byte* data;
		size_t     bytes;
	};
	struct Block {
		std::atomic<size_t>          committed {0}; // bytes the consumer may run
		std::atomic<Block*>          next {nullptr};
		alignas(Align) std::byte     data[BlockBytes];
	};

	static constexpr size_t AlignUp(size_t value) { return (value + Align - 1u) & ~(Align - 1u); }

	static uint64_t Hash(const std::byte* data, size_t bytes) {
		uint64_t hash = 0x9e3779b97f4a7c15ull ^ bytes;
		size_t   i    = 0;
		for (; i + 8 <= bytes; i += 8) {
			uint64_t word;
			std::memcpy(&word, data + i, 8);
			hash = (hash ^ word) * 0x100000001b3ull;
			hash ^= hash >> 29u;
		}
		for (; i < bytes; i++) hash = (hash ^ static_cast<uint8_t>(data[i])) * 0x100000001b3ull;
		return hash;
	}
	static std::pair<const std::byte*, size_t> Payload(const Header* header) {
		const auto* data = reinterpret_cast<const std::byte*>(header + 1);
		if (header->spill == 0u) return {data, header->size - sizeof(Header)};
		const auto* spill = reinterpret_cast<const Spill*>(data);
		return {spill->data, spill->bytes};
	}
	// The consumer sleeps on m_signal: std::atomic::wait returns only once the value differs, so every wake changes it.
	void Wake() {
		m_signal.fetch_add(1, std::memory_order_seq_cst);
		m_signal.notify_one();
	}
	void CheckProducer() {
		if (!m_verify) return;
		const auto id = std::this_thread::get_id();
		if (m_producer_id == std::thread::id {}) m_producer_id = id;
		if (id != m_producer_id) m_foreign++;
	}
	// The open packet is written: hash it.
	void Seal() {
		if (m_open == nullptr) return;
		if (m_verify) {
			const auto [data, bytes] = Payload(m_open);
			m_open->check            = Hash(data, bytes);
		}
		m_open = nullptr;
	}
	void NextBlock() {
		// The consumer recycles the blocks it finished; past the bound the producer waits for one.
		Block* block = nullptr;
		for (;;) {
			{
				std::lock_guard lock(m_free_mutex);
				if (!m_free.empty()) {
					block = m_free.back();
					m_free.pop_back();
					break;
				}
			}
			if (m_blocks < m_max_blocks) {
				block = new Block;
				m_blocks++;
				break;
			}
			m_backpressure_waits++;
			Publish();
			// The consumer may have run everything (and freed a block) since the free list was read: then waiting for one
			// more packet would never return.
			if (const auto executed = m_executed.load(std::memory_order_acquire); executed < m_appended) {
				WaitExecuted(executed + 1u);
			}
		}
		block->committed.store(0, std::memory_order_relaxed);
		block->next.store(nullptr, std::memory_order_relaxed);
		m_tail->committed.store(m_used, std::memory_order_release);
		m_tail->next.store(block, std::memory_order_release);
		m_tail = block;
		m_used = 0;
	}

	void Consume(std::stop_token stop) {
		m_consumer_id   = std::this_thread::get_id();
		uint64_t ran    = 0;
		Block*   block  = m_head;
		size_t   offset = 0;
		for (;;) {
			const auto published = m_published.load(std::memory_order_acquire);
			while (ran < published) {
				const auto committed = block->committed.load(std::memory_order_acquire);
				if (offset == committed) {
					// Everything in this block ran, and more was published: the producer moved on.
					Block* next = block->next.load(std::memory_order_acquire);
					if (next == nullptr || offset != block->committed.load(std::memory_order_acquire)) continue;
					{
						std::lock_guard lock(m_free_mutex);
						m_free.push_back(block);
					}
					block  = next;
					offset = 0;
					continue;
				}
				const auto* header       = reinterpret_cast<const Header*>(block->data + offset);
				const auto [data, bytes] = Payload(header);
				if (m_verify && Hash(data, bytes) != header->check) {
					m_mismatches.fetch_add(1, std::memory_order_relaxed);
					if (m_on_mismatch != nullptr) m_on_mismatch(m_context, header->run, bytes);
				}
				header->run(data, m_context);
				if (header->spill != 0u) ::operator delete[](const_cast<std::byte*>(data), std::align_val_t {Align});
				offset += header->size;
				m_executed.store(++ran, std::memory_order_seq_cst);
				if (m_waiters.load(std::memory_order_seq_cst) != 0) m_executed.notify_all();
			}
			m_head = block;
			// Idle: spin briefly (a producer mid-frame publishes again within microseconds), then sleep.
			bool more = false;
			for (int spin = 0; spin < 256 && !more; spin++) {
#if defined(_M_X64) || defined(__x86_64__)
				_mm_pause();
#endif
				more = m_published.load(std::memory_order_acquire) != ran;
			}
			if (more) continue;
			// Sleeping, then the signal, then the work: a Publish that this misses sees m_sleeping and moves the signal.
			m_sleeping.store(true, std::memory_order_seq_cst);
			const auto signal = m_signal.load(std::memory_order_seq_cst);
			if (m_published.load(std::memory_order_seq_cst) == ran) {
				if (stop.stop_requested()) {
					m_sleeping.store(false, std::memory_order_seq_cst);
					return;
				}
				m_signal.wait(signal, std::memory_order_seq_cst);
			}
			m_sleeping.store(false, std::memory_order_seq_cst);
		}
	}

	// Producer
	void*                 m_context;
	bool                  m_verify;
	size_t                m_max_blocks;
	Mismatch              m_on_mismatch;
	Block*                m_tail     = nullptr;
	size_t                m_used     = 0;
	uint64_t              m_appended = 0;
	Header*               m_open     = nullptr;
	size_t                m_blocks   = 0;
	uint64_t              m_backpressure_waits = 0;
	std::thread::id       m_producer_id;
	uint64_t              m_foreign = 0;
	// Shared
	std::atomic<uint64_t> m_published {0};
	std::atomic<uint64_t> m_executed {0};
	std::atomic<uint32_t> m_waiters {0};
	std::atomic<bool>     m_sleeping {false};
	std::atomic<uint32_t> m_signal {0};
	std::atomic<uint64_t> m_mismatches {0};
	std::mutex            m_free_mutex;
	std::vector<Block*>   m_free;
	// Consumer
	Block*                m_head = nullptr;
	std::thread::id       m_consumer_id;
	std::jthread          m_thread;
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_PACKETSTREAM_H_
