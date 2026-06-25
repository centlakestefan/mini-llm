/* gpu_check.cu — G1: verify the CUDA toolchain and inspect the GPU.
 *
 * This does no model work. It just confirms that nvcc + MSVC build and link,
 * that the runtime sees your GTX 1080 Ti, and that we can allocate device
 * memory. If this runs and prints "compute capability: 6.1", the toolchain is
 * ready for the real port (G2).
 *
 * Build:  nvcc -arch=sm_61 gpu_check.cu -o gpu_check.exe
 *   (if nvcc complains your MSVC is unsupported, add:  -allow-unsupported-compiler)
 * Run:    .\gpu_check.exe
 */

#include <cstdio>
#include <cuda_runtime.h>

/* tiny helper: print and abort on any CUDA error */
static void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        printf("CUDA error during %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

int main(void) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    printf("CUDA devices found: %d\n", count);
    if (count == 0) { printf("no CUDA device visible — check driver/toolkit\n"); return 1; }

    for (int i = 0; i < count; i++) {
        cudaDeviceProp p;
        check(cudaGetDeviceProperties(&p, i), "cudaGetDeviceProperties");
        printf("\nDevice %d: %s\n", i, p.name);
        printf("  compute capability : %d.%d   (expect 6.1 for the 1080 Ti)\n", p.major, p.minor);
        printf("  global memory      : %.1f GB\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  multiprocessors    : %d   (1080 Ti = 28 SMs, 3584 cores)\n", p.multiProcessorCount);
        printf("  max threads/block  : %d\n", p.maxThreadsPerBlock);
        printf("  warp size          : %d\n", p.warpSize);
        printf("  memory bus width   : %d-bit\n", p.memoryBusWidth);
        printf("  GPU clock          : %.0f MHz\n", p.clockRate / 1000.0);
    }

    /* sanity: allocate, set, and free 16 MB on the device */
    const size_t bytes = 16u * 1024u * 1024u;
    void *d = nullptr;
    check(cudaMalloc(&d, bytes), "cudaMalloc");
    check(cudaMemset(d, 0, bytes), "cudaMemset");
    check(cudaFree(d), "cudaFree");
    printf("\ncudaMalloc/Memset/Free of 16 MB: ok\n");
    printf("toolchain looks good — ready for G2 (matmul kernel).\n");
    return 0;
}
