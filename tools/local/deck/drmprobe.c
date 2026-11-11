// drmprobe: libdrm's view of the DRM devices (drmGetDevices2), as RADV enumerates them; for checking Mesa's drm-shim.
// Build: cc -o drmprobe drmprobe.c $(pkg-config --cflags --libs libdrm)
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <xf86drm.h>

int main(void) {
	drmDevicePtr devices[8];
	const int    count = drmGetDevices2(0, devices, 8);
	printf("drmGetDevices2: %d\n", count);
	for (int i = 0; i < count; i++) {
		printf("  available nodes 0x%x bustype %d", devices[i]->available_nodes, devices[i]->bustype);
		if (devices[i]->bustype == DRM_BUS_PCI)
			printf(" pci %04x:%04x", devices[i]->deviceinfo.pci->vendor_id, devices[i]->deviceinfo.pci->device_id);
		if (devices[i]->available_nodes & (1 << DRM_NODE_RENDER)) printf(" render %s", devices[i]->nodes[DRM_NODE_RENDER]);
		printf("\n");
	}
	const int fd = open("/dev/dri/renderD128", O_RDWR);
	printf("open renderD128: %d\n", fd);
	if (fd >= 0) {
		drmVersionPtr version = drmGetVersion(fd);
		if (version != NULL) printf("  driver %s %d.%d\n", version->name, version->version_major, version->version_minor);
	}
	return 0;
}
