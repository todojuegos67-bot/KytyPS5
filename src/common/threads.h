#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"

#include <memory>
#include <string>

namespace Common {

void InitializeThreads();

using thread_func_t    = void (*)(void*);
using wait_poll_func_t = void (*)();

struct ThreadPrivate;
struct MutexPrivate;
struct CondVarPrivate;

class Thread {
public:
	Thread(thread_func_t func, void* arg);
	~Thread();

	void Join();
	void Detach();

	// Once a thread has finished, the id may be reused by another thread.
	[[nodiscard]] std::string GetId() const;

	// The id is unique and can't be reused by another thread.
	[[nodiscard]] int GetUniqueId() const;

	static void Sleep(uint32_t millis);
	static void SleepMicro(uint32_t micros);
	// Idle workers must yield CPU time even for a short delay.
	static void SleepMicroWithoutSpinning(uint32_t micros);
	static void SleepNano(uint64_t nanos);
	static bool IsMainThread();

	// Get current thread id
	// Once a thread has finished, the id may be reused by another thread.
	static std::string GetThreadId();

	// Get current thread id
	// The id is unique and can't be reused by another thread.
	static int GetThreadIdUnique();

	KYTY_CLASS_NO_COPY(Thread);

private:
	std::unique_ptr<ThreadPrivate> m_thread;
};

class Mutex {
public:
	Mutex();
	~Mutex();

	void Lock();
	void Unlock();
	bool TryLock();

	friend class CondVar;

	KYTY_CLASS_NO_COPY(Mutex);

private:
	std::unique_ptr<MutexPrivate> m_mutex;
};

class CondVar {
public:
	CondVar();
	~CondVar();

	void Wait(Mutex* mutex);
	bool WaitFor(Mutex* mutex, uint32_t micros);
	void Signal();
	void SignalAll();

	static void SignalThread(int thread_id);
	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
	std::unique_ptr<CondVarPrivate> m_cond_var;
};

// A thread's own T on the heap, made at the thread's first Get() and deleted when the thread exits: for a large
// per-thread cache. A `thread_local T` is part of the static TLS block that the loader allocates and fills for every
// thread of the process when it starts (the driver's and the system's threads too), whether the thread ever uses it
// or not; this way the threads that never use it pay a pointer. Get() is a thread-local load and a test.
template <typename T>
class ThreadHeap {
public:
	static T& Get() {
		if (T* value = t_value; value != nullptr) [[likely]] {
			return *value;
		}
		return Make();
	}

private:
	struct Owner {
		T* value = nullptr;
		~Owner() {
			t_value = nullptr;
			delete value;
		}
	};
	static T& Make() {
		thread_local Owner owner;
		if (owner.value == nullptr) {
			owner.value = new T();
		}
		t_value = owner.value;
		return *owner.value;
	}
	static inline thread_local T* t_value = nullptr;
};

class LockGuard {
public:
	using mutex_type = Mutex;

	// NOLINTNEXTLINE(google-runtime-references)
	explicit LockGuard(mutex_type& m): m_mutex(m) { m_mutex.Lock(); }

	~LockGuard() { m_mutex.Unlock(); }

	KYTY_CLASS_NO_COPY(LockGuard);

private:
	mutex_type& m_mutex;
};

} // namespace Common

#endif /* KYTY_COMMON_THREADS_H_ */
