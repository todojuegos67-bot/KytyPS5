// vkprobe: the Vulkan devices this environment gives and what pipeline caches and binaries are keyed by (name, ids,
// driver version, pipelineCacheUUID, VK_KHR_pipeline_binary's global key): compare the Steam Deck with RADV's null
// device for its GPU (RADV_FORCE_FAMILY=vangogh) where the Deck's own driver library runs on another PC.
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

static void hex(const uint8_t* bytes, uint32_t count) {
	for (uint32_t i = 0; i < count; i++) printf("%02x", bytes[i]);
}

int main(void) {
	VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion        = VK_API_VERSION_1_3;
	VkInstanceCreateInfo info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	info.pApplicationInfo     = &app;
	VkInstance instance;
	if (vkCreateInstance(&info, NULL, &instance) != VK_SUCCESS) {
		printf("no instance\n");
		return 1;
	}
	uint32_t         count = 8;
	VkPhysicalDevice devices[8];
	vkEnumeratePhysicalDevices(instance, &count, devices);
	for (uint32_t d = 0; d < count; d++) {
		VkPhysicalDeviceSubgroupSizeControlProperties subgroup = {
		    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
		VkPhysicalDeviceDriverProperties driver = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &subgroup};
		VkPhysicalDeviceProperties2      props  = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
		vkGetPhysicalDeviceProperties2(devices[d], &props);
		const VkPhysicalDeviceProperties* p = &props.properties;
		printf("device %u: %s vendor %04x device %04x driver %u.%u.%u (%s %s) type %d subgroup %u-%u\n  cache uuid ", d,
		       p->deviceName, p->vendorID, p->deviceID, VK_VERSION_MAJOR(p->driverVersion),
		       VK_VERSION_MINOR(p->driverVersion), VK_VERSION_PATCH(p->driverVersion), driver.driverName,
		       driver.driverInfo, p->deviceType, subgroup.minSubgroupSize, subgroup.maxSubgroupSize);
		hex(p->pipelineCacheUUID, VK_UUID_SIZE);
		printf("\n");
		// A device with VK_KHR_pipeline_binary: its global key.
		float                   priority = 1.0f;
		VkDeviceQueueCreateInfo queue    = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		queue.queueCount                 = 1;
		queue.pQueuePriorities           = &priority;
		const char*                                 extension = VK_KHR_PIPELINE_BINARY_EXTENSION_NAME;
		VkPhysicalDevicePipelineBinaryFeaturesKHR   binary    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR};
		binary.pipelineBinaries                               = VK_TRUE;
		VkDeviceCreateInfo create                             = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &binary};
		create.queueCreateInfoCount                           = 1;
		create.pQueueCreateInfos                              = &queue;
		create.enabledExtensionCount                          = 1;
		create.ppEnabledExtensionNames                        = &extension;
		VkDevice device;
		if (vkCreateDevice(devices[d], &create, NULL, &device) != VK_SUCCESS) {
			printf("  no device with VK_KHR_pipeline_binary\n");
			continue;
		}
		PFN_vkGetPipelineKeyKHR get_key = (PFN_vkGetPipelineKeyKHR)vkGetDeviceProcAddr(device, "vkGetPipelineKeyKHR");
		VkPipelineBinaryKeyKHR  key     = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
		if (get_key != NULL && get_key(device, NULL, &key) == VK_SUCCESS) {
			printf("  binary global key ");
			hex(key.key, key.keySize);
			printf("\n");
		}
		vkDestroyDevice(device, NULL);
	}
	vkDestroyInstance(instance, NULL);
	return 0;
}
