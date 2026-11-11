// drmver: an LD_PRELOAD for Mesa's amdgpu drm-shim (libamdgpu_noop_drm_shim.so) under a newer driver than the shim:
// drmGetVersion reports the amdgpu kernel interface at least 3.54 (what the Steam Deck's RADV requires), the rest as
// the shim gives it. Build: cc -shared -fPIC -o drmver.so drmver.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

typedef struct {
	int   version_major;
	int   version_minor;
	int   version_patchlevel;
	int   name_len;
	char* name;
	int   date_len;
	char* date;
	int   desc_len;
	char* desc;
} drmVersion;

drmVersion* drmGetVersion(int fd) {
	static drmVersion* (*real)(int) = NULL;
	if (real == NULL) real = (drmVersion * (*)(int)) dlsym(RTLD_NEXT, "drmGetVersion");
	drmVersion* version = real != NULL ? real(fd) : NULL;
	if (version != NULL && version->version_major == 3 && version->version_minor < 54) version->version_minor = 54;
	return version;
}
