// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "FrameInterpolationSwapchainVK_Helpers.h"

#ifdef _WIN32
#include <dwmapi.h>
#endif  // #ifdef _WIN32

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <unordered_map>

// Kyty: no submit waits for a timeline value whose signal no submit has sent yet. The interpolation
// work goes on the game queue and waits for a composition the presenter thread submits later; while
// that wait sat in the queue, the presenter's vkQueuePresentKHR blocked in the kernel
// (NtDxgkSubmitPresentToHwQueue) and nothing moved again (NVIDIA, Windows: frames stopped at places
// 44, 90 and 94). The sparse binds in commandScheduler.cpp follow the same rule.
// shortcut: keyed by handle, so a semaphore recreated under an old handle may pass early (the old
// behaviour); key by creation if the log ever shows the guard missing a wait.
namespace
{
std::mutex                                   g_sentMutex;
std::condition_variable                      g_sentChanged;
std::unordered_map<VkSemaphore, uint64_t>    g_sent;  // highest value each timeline semaphore was sent to signal
unsigned long long                           g_guardWaits    = 0;
unsigned long long                           g_guardTimeouts = 0;

void waitUntilSent(const SubmissionSemaphores& waits)
{
    std::unique_lock<std::mutex> lock(g_sentMutex);
    for (uint32_t i = 0; i < waits.count; ++i)
    {
        const uint64_t value = waits.values[i];  // 0 for a binary semaphore
        const auto     sent  = [&] { return g_sent[waits.semaphores[i]] >= value; };
        if (value == 0 || sent())
            continue;
        ++g_guardWaits;
        // The signal comes from another thread that needs nothing from this one; the timeout only keeps
        // a lost signal from hanging the caller, which then submits as before.
        if (!g_sentChanged.wait_for(lock, std::chrono::seconds(1), sent))
            ++g_guardTimeouts;
    }
}

void recordSent(const SubmissionSemaphores& signals)
{
    {
        std::lock_guard<std::mutex> lock(g_sentMutex);
        for (uint32_t i = 0; i < signals.count; ++i)
        {
            uint64_t& sent = g_sent[signals.semaphores[i]];
            sent           = (std::max)(sent, signals.values[i]);
        }
    }
    g_sentChanged.notify_all();
}
}  // namespace

extern "C" void kyty_fi_GetSendGuardStats(unsigned long long* waits, unsigned long long* timeouts)
{
    std::lock_guard<std::mutex> lock(g_sentMutex);
    *waits    = g_guardWaits;
    *timeouts = g_guardTimeouts;
}

void waitForPerformanceCount(const int64_t targetCount)
{
    int64_t currentCount = 0;
    do
    {
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&currentCount));
    } while (currentCount < targetCount);
}

VkResult VulkanQueue::submit(VkCommandBuffer commandBuffer, SubmissionSemaphores& semaphoresToWait, SubmissionSemaphores& semaphoresToSignal, VkFence fence)
{
    VkSubmitInfo submitInfo         = {};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = semaphoresToWait.count;
    submitInfo.pWaitSemaphores      = semaphoresToWait.semaphores;
    submitInfo.pWaitDstStageMask    = semaphoresToWait.waitStages;
    submitInfo.signalSemaphoreCount = semaphoresToSignal.count;
    submitInfo.pSignalSemaphores    = semaphoresToSignal.semaphores;

    if (commandBuffer == VK_NULL_HANDLE)
    {
        submitInfo.commandBufferCount = 0;
        submitInfo.pCommandBuffers    = nullptr;
    }
    else
    {
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffer;
    }

    VkTimelineSemaphoreSubmitInfo timelineSemaphoreSubmitInfo = {};
    if (semaphoresToSignal.count == 0 && semaphoresToWait.count == 0)
    {
        submitInfo.pNext = nullptr;
    }
    else
    {
        timelineSemaphoreSubmitInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineSemaphoreSubmitInfo.pNext                     = nullptr;
        timelineSemaphoreSubmitInfo.waitSemaphoreValueCount   = semaphoresToWait.count;
        timelineSemaphoreSubmitInfo.pWaitSemaphoreValues      = semaphoresToWait.values;
        timelineSemaphoreSubmitInfo.signalSemaphoreValueCount = semaphoresToSignal.count;
        timelineSemaphoreSubmitInfo.pSignalSemaphoreValues    = semaphoresToSignal.values;

        submitInfo.pNext = &timelineSemaphoreSubmitInfo;
    }

    waitUntilSent(semaphoresToWait);

    VkResult res = VK_SUCCESS;
    if (submitFunc != nullptr)
        res = submitFunc(1, &submitInfo, fence);
    else
        res = vkQueueSubmit(queue, 1, &submitInfo, fence);

    if (res == VK_SUCCESS)
        recordSent(semaphoresToSignal);

    semaphoresToWait.reset();
    semaphoresToSignal.reset();

    return res;
}

VkResult VulkanQueue::submit(VkCommandBuffer commandBuffer, VkSemaphore timelineSemaphore, uint64_t signalValue)
{
    SubmissionSemaphores semaphoresToWait;
    SubmissionSemaphores semaphoresToSignal;

    semaphoresToSignal.add(timelineSemaphore, signalValue);

    return submit(commandBuffer, semaphoresToWait, semaphoresToSignal);
}

VkResult CreateShaderModule(VkDevice device, size_t codeSize, const uint32_t* pCode, VkShaderModule* pModule, const VkAllocationCallbacks* pAllocator)
{
    VkShaderModuleCreateInfo info = {};
    info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.pNext                    = nullptr;
    info.flags                    = 0;
    info.codeSize                 = codeSize;
    info.pCode                    = pCode;

    return vkCreateShaderModule(device, &info, pAllocator, pModule);
}
