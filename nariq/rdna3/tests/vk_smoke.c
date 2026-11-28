#include "ggml-vulkan.h"
#include "ggml-backend.h"
#include <stdio.h>
int main(void) {
    int n = ggml_backend_vk_get_device_count();
    printf("vulkan devices: %d\n", n);
    if (n < 1) return 1;
    ggml_backend_t b = ggml_backend_vk_init(0);
    if (!b) { puts("init failed"); return 1; }
    printf("backend: %s\n", ggml_backend_name(b));
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    printf("device: %s\n", ggml_backend_dev_description(dev));
    size_t free_b, total_b;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    printf("memory: %.1f / %.1f MB\n", free_b/1048576.0, total_b/1048576.0);
    ggml_backend_free(b);
    puts("VK_SMOKE_OK");
    return 0;
}
