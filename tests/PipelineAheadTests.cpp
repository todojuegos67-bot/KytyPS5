#include "graphics/host_gpu/renderer/pipeline/pipelineAhead.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::AheadTranslation;
using Libs::Graphics::AheadWorkerCount;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "PipelineAheadTests: failed: %s\n", message);
    std::abort();
  }
}

void TestWorkerCount() {
  // Small machines keep the old pool, min(threads / 2, 8); larger ones keep four threads free, at most 16 workers.
  Check(AheadWorkerCount(0) == 1, "no thread count still gives one worker");
  Check(AheadWorkerCount(1) == 1 && AheadWorkerCount(2) == 1, "one or two threads give one worker");
  Check(AheadWorkerCount(4) == 2, "4 threads give 2 workers, as before");
  Check(AheadWorkerCount(6) == 3, "6 threads give 3 workers, as before");
  Check(AheadWorkerCount(8) == 4, "8 threads give 4 workers, as before");
  Check(AheadWorkerCount(12) == 8, "12 threads give 8 workers");
  Check(AheadWorkerCount(16) == 12, "16 threads give 12 workers");
  Check(AheadWorkerCount(20) == 16, "20 threads give 16 workers");
  Check(AheadWorkerCount(32) == 16 && AheadWorkerCount(128) == 16, "at most 16 workers");
  for (uint32_t threads = 1; threads <= 256; threads++) {
    const auto workers = AheadWorkerCount(threads);
    Check(workers >= std::min(threads / 2u, 8u), "fewer workers than before");
    Check(workers <= 16u, "more than 16 workers");
    Check(threads <= 4 || workers <= threads - 4 || workers == std::min(threads / 2u, 8u),
          "fewer than four threads left");
  }
}

void TestTranslationSteps() {
  using Step = AheadTranslation<int>::Step;
  AheadTranslation<int> program;
  Check(program.Peek() == Step::Waiting, "an untranslated program is not ready for the scanner");
  Check(program.Take(1) == Step::Translate, "the first item does not translate");
  Check(program.Peek() == Step::Waiting, "a program being translated is ready for the scanner");
  Check(program.Take(2) == Step::Waiting && program.Take(3) == Step::Waiting, "items do not wait on the translation");
  const auto waiting = program.Finish(true);
  Check(waiting.size() == 2 && waiting[0] == 2 && waiting[1] == 3, "the waiting items are not returned in order");
  Check(program.Ready() && program.Take(4) == Step::Ready && program.Peek() == Step::Ready, "a translated program is not ready");
  Check(program.Finish(true).empty(), "items are returned twice");

  AheadTranslation<int> failed;
  Check(failed.Take(1) == Step::Translate && failed.Take(2) == Step::Waiting, "the failing program is not translated once");
  const auto released = failed.Finish(false);
  Check(released.size() == 1 && released[0] == 2, "the items waiting on a failed translation are kept");
  Check(failed.Unsupported() && failed.Take(3) == Step::Unsupported && failed.Peek() == Step::Unsupported,
        "a failed translation is not unsupported");

  // An exit found while it was translated (a materialization the registers break) wins over the translation.
  AheadTranslation<int> broken;
  Check(broken.Take(1) == Step::Translate, "the broken program is not translated");
  broken.SetUnsupported();
  (void)broken.Finish(true);
  Check(broken.Take(2) == Step::Unsupported && !broken.Ready(), "unsupported is lost by a later Finish");
}

// The workers' protocol of PipelineCache::Ahead::TakeProgram: one queue under one lock, items that wait put back at
// the front once the translation is done. Every item is prepared once, and every program translated once.
void TestTranslationWorkers() {
  constexpr uint32_t Programs = 5;
  constexpr uint32_t Items    = 400;
  constexpr uint32_t Workers  = 8;
  struct Item {
    uint32_t program = 0;
    uint32_t id      = 0;
  };
  std::mutex                                     mutex;
  std::condition_variable                        changed;
  std::deque<Item>                               jobs;
  std::array<AheadTranslation<Item>, Programs>   programs;
  std::array<uint32_t, Programs>                 translations {};
  std::vector<uint32_t>                          prepared(Items, 0);
  uint32_t                                       left = Items;
  for (uint32_t id = 0; id < Items; id++) jobs.push_back({id % Programs, id});

  const auto work = [&] {
    for (;;) {
      Item item;
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return left == 0 || !jobs.empty(); });
        if (left == 0) return;
        item = jobs.front();
        jobs.pop_front();
      }
      auto &program = programs[item.program];
      using Step    = AheadTranslation<Item>::Step;
      Step step;
      {
        std::lock_guard lock(mutex);
        step = program.Take(item);
      }
      if (step == Step::Waiting) continue;
      if (step == Step::Translate) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::lock_guard lock(mutex);
        translations[item.program]++;
        auto waiting = program.Finish(true);
        jobs.insert(jobs.begin(), waiting.begin(), waiting.end());
      }
      {
        std::lock_guard lock(mutex);
        prepared[item.id]++;
        left--;
      }
      changed.notify_all();
    }
  };
  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < Workers; i++) threads.emplace_back(work);
  for (auto &thread : threads) thread.join();

  for (uint32_t i = 0; i < Programs; i++) Check(translations[i] == 1, "a program was translated more than once");
  for (uint32_t id = 0; id < Items; id++) Check(prepared[id] == 1, "an item was lost or prepared twice");
  Check(jobs.empty(), "items are left in the queue");
}

} // namespace

int main() {
  TestWorkerCount();
  TestTranslationSteps();
  TestTranslationWorkers();
  std::printf("PipelineAheadTests: ok\n");
  return 0;
}
