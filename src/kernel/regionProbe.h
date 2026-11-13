#ifndef KYTY_KERNEL_REGIONPROBE_H_
#define KYTY_KERNEL_REGIONPROBE_H_

// (blk-1) A read-only start-up probe for one host address (0xff60000000 by default: the ~1 GiB private allocation the
// "Private breakdown:" line lists first, right above the guest backing store's view). Each Check() prints, with
// std::printf, what VirtualQuery says is at that address (free, or its allocation: base, reserved and committed size,
// type, protection) and the checkpoint's name; the first call also starts a low-cost thread that polls the address every
// millisecond until something is allocated there, prints that moment with the last checkpoint passed, then prints the
// allocation every time its commit changes by 32 MiB or more (for 10 minutes). The allocation's first bytes are read
// with ReadProcessMemory (no fault possible) to tell an NT heap segment (0xffeeffee) or a Segment Heap (0xddeeddee) from
// the rest. Nothing is hooked, allocated at that address, protected or freed.
// KYTY_REGION_PROBE=0: off. KYTY_REGION_PROBE_ADDR=<hex>: another address.

namespace RegionProbe {

// checkpoint: a string literal (kept by pointer for the polling thread).
void Check(const char* checkpoint);

} // namespace RegionProbe

#endif /* KYTY_KERNEL_REGIONPROBE_H_ */
