// protbench: what a write-protection re-arm costs on this machine, the way the emulator does it (mprotect of a few
// pages of a shared memfd mapping, then a write fault that lifts it) against userfaultfd write-protection
// (UFFDIO_WRITEPROTECT, synchronous; and asynchronous with PAGEMAP_SCAN), with N threads of the process busy on other
// CPUs (TLB shootdowns reach them) and M extra mappings (VMA tree size).
//   protbench [busy threads 7] [extra mappings 15000] [pages per range 4] [rounds 2000]
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/userfaultfd.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define SIZE (256ul << 20)
static uint8_t*        g_map;
static atomic_int      g_stop;
static long            g_page;
static volatile size_t g_sink;

static double Now() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static void* Busy(void* arg) {
	(void)arg;
	size_t sum = 0, i = 0;
	while (!atomic_load_explicit(&g_stop, memory_order_relaxed)) {
		sum += g_map[(i * 4096 * 7 + 64) % SIZE];
		i++;
	}
	g_sink = sum;
	return NULL;
}

static void Handler(int sig, siginfo_t* info, void* context) {
	(void)sig;
	(void)context;
	uint8_t* page = (uint8_t*)((uintptr_t)info->si_addr & ~(uintptr_t)(g_page - 1));
	mprotect(page, g_page, PROT_READ | PROT_WRITE);
}

static int        g_sigbus_fd = -1;
static atomic_long g_bus_faults;

static void BusHandler(int sig, siginfo_t* info, void* context) {
	(void)sig;
	(void)context;
	const uintptr_t            page = (uintptr_t)info->si_addr & ~(uintptr_t)(g_page - 1);
	struct uffdio_writeprotect w    = {.range = {page, (uint64_t)g_page}, .mode = 0};
	ioctl(g_sigbus_fd, UFFDIO_WRITEPROTECT, &w);
	atomic_fetch_add_explicit(&g_bus_faults, 1, memory_order_relaxed);
}

int main(int argc, char** argv) {
	const int busy = argc > 1 ? atoi(argv[1]) : 7, extra = argc > 2 ? atoi(argv[2]) : 15000;
	const int pages = argc > 3 ? atoi(argv[3]) : 4, rounds = argc > 4 ? atoi(argv[4]) : 2000;
	g_page = sysconf(_SC_PAGESIZE);
	const int fd = memfd_create("protbench", 0);
	if (fd < 0 || ftruncate(fd, SIZE) != 0) return 1;
	g_map = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	memset(g_map, 1, SIZE);
	// Extra small mappings of the same file elsewhere (the guest's scattered pieces).
	for (int i = 0; i < extra; i++) {
		void* p = mmap(NULL, 2 * g_page, (i & 1) ? PROT_READ : PROT_READ | PROT_WRITE, MAP_SHARED, fd,
		               (off_t)((i * 2 * g_page) % SIZE));
		if (p == MAP_FAILED) break;
	}
	struct sigaction sa = {0};
	sa.sa_sigaction    = Handler;
	sa.sa_flags        = SA_SIGINFO;
	sigaction(SIGSEGV, &sa, NULL);
	pthread_t threads[64];
	for (int i = 0; i < busy; i++) pthread_create(&threads[i], NULL, Busy, NULL);
	usleep(100000);
	const size_t range = (size_t)pages * g_page;
	// 1. mprotect re-arm (read-only) + a write fault that lifts it (handler: mprotect of its page).
	double protect = 0, fault = 0;
	for (int r = 0; r < rounds; r++) {
		uint8_t*     at = g_map + ((size_t)(r * 7919) * range) % (SIZE - range);
		const double t0 = Now();
		mprotect(at, range, PROT_READ);
		const double t1 = Now();
		for (int p = 0; p < pages; p++) at[p * g_page] = 2;
		const double t2 = Now();
		protect += t1 - t0;
		fault += t2 - t1;
	}
	printf("mprotect: re-arm %.2f us per call (%d pages), write faults %.2f us per page\n", protect / rounds, pages,
	       fault / rounds / pages);
	// 2. userfaultfd write-protection (asynchronous: the kernel lifts it on a write and keeps the written bit).
	const int uffd = (int)syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
	if (uffd < 0) {
		printf("userfaultfd: %s\n", strerror(errno));
	} else {
		struct uffdio_api api = {.api = UFFD_API,
		                         .features = UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_HUGETLBFS_SHMEM |
		                                     UFFD_FEATURE_WP_UNPOPULATED};
		if (ioctl(uffd, UFFDIO_API, &api) != 0) {
			printf("UFFDIO_API (async wp): %s\n", strerror(errno));
		} else {
			struct uffdio_register reg = {.range = {(uintptr_t)g_map, SIZE}, .mode = UFFDIO_REGISTER_MODE_WP};
			if (ioctl(uffd, UFFDIO_REGISTER, &reg) != 0) {
				printf("UFFDIO_REGISTER wp: %s\n", strerror(errno));
			} else {
				double wp = 0, wfault = 0, scan = 0;
				long   written = 0;
				for (int r = 0; r < rounds; r++) {
					uint8_t*                at = g_map + ((size_t)(r * 7919) * range) % (SIZE - range);
					struct uffdio_writeprotect w = {.range = {(uintptr_t)at, range}, .mode = UFFDIO_WRITEPROTECT_MODE_WP};
					const double            t0 = Now();
					ioctl(uffd, UFFDIO_WRITEPROTECT, &w);
					const double t1 = Now();
					for (int p = 0; p < pages; p++) at[p * g_page] = 3;
					const double t2 = Now();
#ifdef PAGEMAP_SCAN
					// The written pages of the range, write-protected again in the same call.
					struct page_region regions[16];
					struct pm_scan_arg arg = {.size = sizeof(arg), .flags = PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC,
					                          .start = (uintptr_t)at, .end = (uintptr_t)at + range,
					                          .vec = (uintptr_t)regions, .vec_len = 16, .category_mask = PAGE_IS_WRITTEN,
					                          .return_mask = PAGE_IS_WRITTEN};
					const int    pagemap = open("/proc/self/pagemap", O_RDONLY);
					const double t3      = Now();
					const long   n       = ioctl(pagemap, PAGEMAP_SCAN, &arg);
					scan += Now() - t3;
					if (n > 0) written += n;
					close(pagemap);
#endif
					wp += t1 - t0;
					wfault += t2 - t1;
				}
				printf("uffd async wp: re-arm %.2f us per call, write faults %.2f us per page, PAGEMAP_SCAN get+re-arm "
				       "%.2f us per call (%ld written regions seen)\n",
				       wp / rounds, wfault / rounds / pages, scan / rounds, written);
			}
		}
	}
	// 3. userfaultfd write-protection with UFFD_FEATURE_SIGBUS (synchronous): a write raises SIGBUS in the writing thread
	// (as the emulator's SIGSEGV now), whose handler lifts the protection of its page (UFFDIO_WRITEPROTECT, mode 0).
	const int sigbus_fd = (int)syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
	if (sigbus_fd >= 0) {
		struct uffdio_api api = {.api = UFFD_API,
		                         .features = UFFD_FEATURE_SIGBUS | UFFD_FEATURE_WP_HUGETLBFS_SHMEM |
		                                     UFFD_FEATURE_WP_UNPOPULATED};
		g_sigbus_fd = sigbus_fd;
		struct sigaction bus = {0};
		bus.sa_sigaction     = BusHandler;
		bus.sa_flags         = SA_SIGINFO;
		sigaction(SIGBUS, &bus, NULL);
		// A mapping of its own (the async registration above owns g_map's range).
		uint8_t* map = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		struct uffdio_register reg = {.range = {(uintptr_t)map, SIZE}, .mode = UFFDIO_REGISTER_MODE_WP};
		if (ioctl(sigbus_fd, UFFDIO_API, &api) != 0 || ioctl(sigbus_fd, UFFDIO_REGISTER, &reg) != 0) {
			printf("uffd sigbus: %s\n", strerror(errno));
		} else {
			double wp = 0, wfault = 0;
			for (int r = 0; r < rounds; r++) {
				uint8_t*                   at = map + ((size_t)(r * 7919) * range) % (SIZE - range);
				struct uffdio_writeprotect w  = {.range = {(uintptr_t)at, range}, .mode = UFFDIO_WRITEPROTECT_MODE_WP};
				const double               t0 = Now();
				ioctl(sigbus_fd, UFFDIO_WRITEPROTECT, &w);
				const double t1 = Now();
				for (int p = 0; p < pages; p++) at[p * g_page] = 4;
				const double t2 = Now();
				wp += t1 - t0;
				wfault += t2 - t1;
			}
			printf("uffd sigbus wp: re-arm %.2f us per call, write faults (SIGBUS + lift in the handler) %.2f us per "
			       "page, %ld faults\n",
			       wp / rounds, wfault / rounds / pages, (long)atomic_load(&g_bus_faults));
		}
	}
	atomic_store(&g_stop, 1);
	for (int i = 0; i < busy; i++) pthread_join(threads[i], NULL);
	return 0;
}
