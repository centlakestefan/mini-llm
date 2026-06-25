/* gpu_matmul.cu — G2: the GPU matmul kernel, verified against the CPU.
 *
 * This is the single kernel that matters: in run.c essentially all compute is
 * matmul. Here we prove the GPU version is numerically equivalent to the CPU
 * version before we trust it inside the model (G3).
 *
 * Build (CUDA 12.9 + 14.44 toolset, Pascal sm_61):
 *   cmd /c 'call "...\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 ^
 *          && nvcc -arch=sm_61 gpu_matmul.cu -o gpu_matmul.exe'
 * Run:  .\gpu_matmul.exe
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t e_ = (call);                                             \
    if (e_ != cudaSuccess) {                                            \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_),       \
               __FILE__, __LINE__); exit(1);                            \
    } } while (0)

/* ---- CPU reference: identical to matmul() in run.c ----
 * W is (d, n) row-major, x is (n,), out is (d,):  out[i] = sum_j W[i,j]*x[j] */
static void matmul_cpu(float *out, const float *x, const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[(long long)i * n + j] * x[j];
        out[i] = val;
    }
}

/* ---- GPU kernel: one thread computes one output element (one row of W . x) ----
 * This is the direct parallelization of the CPU outer loop. */
__global__ void matmul_kernel(float *out, const float *x, const float *w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[(long long)i * n + j] * x[j];
        out[i] = val;
    }
}

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; } /* [-1,1] */

static void run_test(int n, int d) {
    size_t wsz = (size_t)d * n;
    float *x = (float *)malloc((size_t)n * sizeof(float));
    float *w = (float *)malloc(wsz * sizeof(float));
    float *out_cpu = (float *)malloc((size_t)d * sizeof(float));
    float *out_gpu = (float *)malloc((size_t)d * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = frand();
    for (size_t i = 0; i < wsz; i++) w[i] = frand();

    /* CPU */
    matmul_cpu(out_cpu, x, w, n, d);

    /* GPU: upload, launch, download */
    float *dx, *dw, *dout;
    CUDA_CHECK(cudaMalloc(&dx, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, wsz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, (size_t)d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, w, wsz * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256, blocks = (d + threads - 1) / threads;

    /* warm up, then time the kernel alone with CUDA events */
    matmul_kernel<<<blocks, threads>>>(dout, dx, dw, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matmul_kernel<<<blocks, threads>>>(dout, dx, dw, n, d);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);

    CUDA_CHECK(cudaMemcpy(out_gpu, dout, (size_t)d * sizeof(float), cudaMemcpyDeviceToHost));

    /* compare */
    float max_abs = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = fabsf(out_cpu[i] - out_gpu[i]);
        if (diff > max_abs) max_abs = diff;
    }
    printf("matmul n=%-5d d=%-6d : max|cpu-gpu| = %.3e   kernel %.3f ms   %s\n",
           n, d, max_abs, ms, max_abs < 1e-3f ? "MATCH" : "MISMATCH!");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
    free(x); free(w); free(out_cpu); free(out_gpu);
}

int main(void) {
    srand(1234);
    printf("verifying GPU matmul kernel against the CPU reference:\n\n");
    run_test(288, 288);    /* an attention/FFN-sized projection */
    run_test(288, 768);    /* FFN up-projection width            */
    run_test(288, 32000);  /* the classifier: the biggest matmul */
    printf("\n(tiny nonzero diffs are float summation-order noise, not errors.)\n");
    return 0;
}
