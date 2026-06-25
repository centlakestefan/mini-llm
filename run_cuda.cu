/* run_cuda.cu — G3: run.c's forward pass on the GPU (NVIDIA, FP32, sm_61).
 *
 * Weights + the whole RunState (incl. KV cache) live in device memory and never
 * leave it during generation. The host loops over layers launching kernels, and
 * copies back only the final logits to sample on the CPU. Tokenizer + sampler
 * stay on the host (trivial cost). Every kernel mirrors a piece of run.c.
 *
 * Build (CUDA 12.9 + 14.44 toolset, Pascal sm_61):
 *   cmd /c 'call "...\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 ^
 *          && nvcc -arch=sm_61 -Wno-deprecated-gpu-targets run_cuda.cu -o run_cuda.exe'
 * Run:  .\run_cuda.exe "Once upon a time" 1.0
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n",                 \
        cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

/* ============================ model structs (host) ============================ */
typedef struct { int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len; } Config;

typedef struct {
    float *token_embedding_table, *rms_att_weight, *wq, *wk, *wv, *wo,
          *rms_ffn_weight, *w1, *w2, *w3, *rms_final_weight, *wcls;
} TransformerWeights;

/* device-resident scratch (same layout as run.c's RunState) */
typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *logits, *key_cache, *value_cache;
} DeviceState;

/* pointer arithmetic only — works for host OR device base pointers */
static void map_weights(TransformerWeights *w, Config *c, float *p, int shared) {
    int hs = c->dim / c->n_heads; long long n = c->n_layers;
    w->token_embedding_table = p; p += (long long)c->vocab_size * c->dim;
    w->rms_att_weight = p;        p += n * c->dim;
    w->wq = p;                    p += n * c->dim * (c->n_heads * hs);
    w->wk = p;                    p += n * c->dim * (c->n_kv_heads * hs);
    w->wv = p;                    p += n * c->dim * (c->n_kv_heads * hs);
    w->wo = p;                    p += n * (c->n_heads * hs) * c->dim;
    w->rms_ffn_weight = p;        p += n * c->dim;
    w->w1 = p;                    p += n * c->dim * c->hidden_dim;
    w->w2 = p;                    p += n * c->hidden_dim * c->dim;
    w->w3 = p;                    p += n * c->dim * c->hidden_dim;
    w->rms_final_weight = p;      p += c->dim;
    p += c->seq_len * hs / 2; p += c->seq_len * hs / 2;   /* skip legacy RoPE tables */
    w->wcls = shared ? w->token_embedding_table : p;
}

/* ================================ GPU kernels ================================ */

/* out = W·x ; W is (d,n) row-major. One thread per output row. */
__global__ void matmul_k(float *out, const float *x, const float *w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        float v = 0.0f;
        for (int j = 0; j < n; j++) v += w[(long long)i * n + j] * x[j];
        out[i] = v;
    }
}

/* o = w * x / rms(x). Single block, shared-memory reduction. */
__global__ void rmsnorm_k(float *o, const float *x, const float *w, int size) {
    __shared__ float red[256];
    __shared__ float scale;
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int j = tid; j < size; j += blockDim.x) ss += x[j] * x[j];
    red[tid] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    if (tid == 0) scale = rsqrtf(red[0] / size + 1e-5f);
    __syncthreads();
    for (int j = tid; j < size; j += blockDim.x) o[j] = w[j] * (scale * x[j]);
}

/* rotate consecutive pairs of q (and k within kv_dim) by angle ~ pos */
__global__ void rope_k(float *q, float *k, int pos, int dim, int kv_dim, int head_size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= dim) return;
    int head_dim = idx % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq, fcr = cosf(val), fci = sinf(val);
    float q0 = q[idx], q1 = q[idx + 1];
    q[idx] = q0 * fcr - q1 * fci; q[idx + 1] = q0 * fci + q1 * fcr;
    if (idx < kv_dim) {
        float k0 = k[idx], k1 = k[idx + 1];
        k[idx] = k0 * fcr - k1 * fci; k[idx + 1] = k0 * fci + k1 * fcr;
    }
}

/* one block per head: score vs all cached keys, softmax, blend the values */
__global__ void attention_k(float *xb, const float *q, const float *kbase,
                            const float *vbase, int pos, int head_size,
                            int kv_dim, int kv_mul) {
    __shared__ float s_att[256];   /* seq_len */
    __shared__ float red[256];
    int tid = threadIdx.x, h = blockIdx.x;
    const float *qh = q + h * head_size;
    int kvh = h / kv_mul;
    float scale = rsqrtf((float)head_size);

    for (int t = tid; t <= pos; t += blockDim.x) {
        const float *k = kbase + (long long)t * kv_dim + kvh * head_size;
        float sc = 0.0f;
        for (int i = 0; i < head_size; i++) sc += qh[i] * k[i];
        s_att[t] = sc * scale;
    }
    __syncthreads();

    /* softmax max */
    float m = -1e30f;
    for (int t = tid; t <= pos; t += blockDim.x) m = fmaxf(m, s_att[t]);
    red[tid] = m; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float maxv = red[0]; __syncthreads();

    /* exp + sum */
    float sum = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) { float e = expf(s_att[t] - maxv); s_att[t] = e; sum += e; }
    red[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float total = red[0]; __syncthreads();

    /* weighted sum of values */
    float *xbh = xb + h * head_size;
    for (int i = tid; i < head_size; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++)
            acc += s_att[t] * vbase[(long long)t * kv_dim + kvh * head_size + i];
        xbh[i] = acc / total;
    }
}

__global__ void add_k(float *x, const float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += y[i];
}
__global__ void swiglu_k(float *hb, const float *hb2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float z = hb[i]; z *= 1.0f / (1.0f + expf(-z)); hb[i] = z * hb2[i]; }
}

/* ============================== forward (host) ============================== */
#define BLK 256
static inline int grid(int n) { return (n + BLK - 1) / BLK; }

static void forward_cuda(Config *c, TransformerWeights *w, DeviceState *s,
                         int token, int pos, float *logits_host) {
    int dim = c->dim, hidden = c->hidden_dim, hs = dim / c->n_heads;
    int kv_dim = (dim * c->n_kv_heads) / c->n_heads;
    int kv_mul = c->n_heads / c->n_kv_heads;

    /* embedding: x <- row `token` (device-to-device copy) */
    CUDA_CHECK(cudaMemcpy(s->x, w->token_embedding_table + (long long)token * dim,
                          dim * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int l = 0; l < c->n_layers; l++) {
        rmsnorm_k<<<1, BLK>>>(s->xb, s->x, w->rms_att_weight + (long long)l * dim, dim);

        long long loff = (long long)l * c->seq_len * kv_dim;
        float *k = s->key_cache + loff + (long long)pos * kv_dim;
        float *v = s->value_cache + loff + (long long)pos * kv_dim;

        matmul_k<<<grid(dim), BLK>>>(s->q, s->xb, w->wq + (long long)l * dim * dim, dim, dim);
        matmul_k<<<grid(kv_dim), BLK>>>(k, s->xb, w->wk + (long long)l * dim * kv_dim, dim, kv_dim);
        matmul_k<<<grid(kv_dim), BLK>>>(v, s->xb, w->wv + (long long)l * dim * kv_dim, dim, kv_dim);

        rope_k<<<grid(dim / 2), BLK>>>(s->q, k, pos, dim, kv_dim, hs);

        attention_k<<<c->n_heads, BLK>>>(s->xb, s->q, s->key_cache + loff,
                                         s->value_cache + loff, pos, hs, kv_dim, kv_mul);

        matmul_k<<<grid(dim), BLK>>>(s->xb2, s->xb, w->wo + (long long)l * dim * dim, dim, dim);
        add_k<<<grid(dim), BLK>>>(s->x, s->xb2, dim);

        rmsnorm_k<<<1, BLK>>>(s->xb, s->x, w->rms_ffn_weight + (long long)l * dim, dim);
        matmul_k<<<grid(hidden), BLK>>>(s->hb, s->xb, w->w1 + (long long)l * dim * hidden, dim, hidden);
        matmul_k<<<grid(hidden), BLK>>>(s->hb2, s->xb, w->w3 + (long long)l * dim * hidden, dim, hidden);
        swiglu_k<<<grid(hidden), BLK>>>(s->hb, s->hb2, hidden);
        matmul_k<<<grid(dim), BLK>>>(s->xb, s->hb, w->w2 + (long long)l * dim * hidden, hidden, dim);
        add_k<<<grid(dim), BLK>>>(s->x, s->xb, dim);
    }

    rmsnorm_k<<<1, BLK>>>(s->x, s->x, w->rms_final_weight, dim);
    matmul_k<<<grid(c->vocab_size), BLK>>>(s->logits, s->x, w->wcls, dim, c->vocab_size);

    /* the only transfer in the loop: logits back to the host for sampling */
    CUDA_CHECK(cudaMemcpy(logits_host, s->logits, c->vocab_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

/* ============================ tokenizer (host) ============================ */
typedef struct { char *str; int id; } TokenIndex;
typedef struct {
    char **vocab; float *vocab_scores; TokenIndex *sorted;
    int vocab_size; unsigned int max_token_length; char byte_pieces[512];
} Tokenizer;

static int cmp_tok(const void *a, const void *b) { return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str); }
static int str_lookup(char *s, TokenIndex *sorted, int n) {
    TokenIndex key; key.str = s;
    TokenIndex *r = (TokenIndex*)bsearch(&key, sorted, n, sizeof(TokenIndex), cmp_tok);
    return r ? r->id : -1;
}
static void build_tokenizer(Tokenizer *t, const char *path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted = NULL;
    for (int i = 0; i < 256; i++) { t->byte_pieces[i*2] = (char)i; t->byte_pieces[i*2+1] = '\0'; }
    FILE *f = fopen(path, "rb");
    if (!f) { printf("can't open %s\n", path); exit(1); }
    if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) { printf("bad tokenizer\n"); exit(1); }
    for (int i = 0; i < vocab_size; i++) {
        int len; fread(t->vocab_scores + i, sizeof(float), 1, f); fread(&len, sizeof(int), 1, f);
        t->vocab[i] = (char*)malloc(len + 1); fread(t->vocab[i], 1, len, f); t->vocab[i][len] = '\0';
    }
    fclose(f);
}
static char *decode(Tokenizer *t, int prev, int token) {
    char *piece = t->vocab[token];
    if (prev == 1 && piece[0] == ' ') piece++;
    unsigned char bv;
    if (sscanf(piece, "<0x%02hhX>", &bv) == 1) piece = (char*)t->byte_pieces + bv * 2;
    return piece;
}
static void encode(Tokenizer *t, char *text, int bos, int eos, int *tokens, int *n_tokens) {
    if (t->sorted == NULL) {
        t->sorted = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) { t->sorted[i].str = t->vocab[i]; t->sorted[i].id = i; }
        qsort(t->sorted, t->vocab_size, sizeof(TokenIndex), cmp_tok);
    }
    char *buf = (char*)malloc(t->max_token_length * 2 + 3);
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') tokens[(*n_tokens)++] = str_lookup((char*)" ", t->sorted, t->vocab_size);
    size_t len = 0;
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) len = 0;
        buf[len++] = *c; buf[len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && len < 4) continue;
        int id = str_lookup(buf, t->sorted, t->vocab_size);
        if (id != -1) tokens[(*n_tokens)++] = id;
        else for (size_t i = 0; i < len; i++) tokens[(*n_tokens)++] = (unsigned char)buf[i] + 3;
        len = 0;
    }
    while (1) {
        float best = -1e10f; int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(buf, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(buf, t->sorted, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best) { best = t->vocab_scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(buf);
}

/* ============================== sampler (host) ============================== */
typedef struct { float prob; int index; } ProbIndex;
typedef struct { int vocab_size; ProbIndex *probindex; float temperature, topp; unsigned long long rng; } Sampler;

static void softmax(float *x, int n) {
    float m = x[0]; for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}
static unsigned int random_u32(unsigned long long *st) {
    *st ^= *st >> 12; *st ^= *st << 25; *st ^= *st >> 27; return (*st * 0x2545F4914F6CDD1DULL) >> 32;
}
static float random_f32(unsigned long long *st) { return (random_u32(st) >> 8) / 16777216.0f; }
static int sample_argmax(float *p, int n) { int b = 0; for (int i = 1; i < n; i++) if (p[i] > p[b]) b = i; return b; }
static int sample_mult(float *p, int n, float coin) { float cdf = 0; for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; } return n - 1; }
static int cmp_pi(const void *a, const void *b) { float x = ((ProbIndex*)a)->prob, y = ((ProbIndex*)b)->prob; return (x < y) - (x > y); }
static int sample_topp(float *p, int n, float topp, ProbIndex *pi, float coin) {
    int n0 = 0; float cut = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) if (p[i] >= cut) { pi[n0].index = i; pi[n0].prob = p[i]; n0++; }
    qsort(pi, n0, sizeof(ProbIndex), cmp_pi);
    float cum = 0; int last = n0 - 1;
    for (int i = 0; i < n0; i++) { cum += pi[i].prob; if (cum > topp) { last = i; break; } }
    float r = coin * cum, cdf = 0;
    for (int i = 0; i <= last; i++) { cdf += pi[i].prob; if (r < cdf) return pi[i].index; }
    return pi[last].index;
}
static int sample(Sampler *s, float *logits) {
    if (s->temperature == 0.0f) return sample_argmax(logits, s->vocab_size);
    for (int i = 0; i < s->vocab_size; i++) logits[i] /= s->temperature;
    softmax(logits, s->vocab_size);
    float coin = random_f32(&s->rng);
    if (s->topp <= 0 || s->topp >= 1) return sample_mult(logits, s->vocab_size, coin);
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}

/* ================================== main ================================== */
int main(int argc, char **argv) {
    const char *checkpoint = "stories15M.bin";
    char *prompt = (argc >= 2) ? argv[1] : (char*)"Once upon a time";
    float temperature = (argc >= 3) ? (float)atof(argv[2]) : 1.0f;
    float topp = 0.9f; unsigned long long seed = 1234ULL;

    /* --- load checkpoint on the host --- */
    FILE *f = fopen(checkpoint, "rb");
    if (!f) { printf("can't open %s\n", checkpoint); return 1; }
    fseek(f, 0, SEEK_END); long long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    char *data = (char*)malloc(fsz); fread(data, 1, fsz, f); fclose(f);
    Config c = *(Config*)data;
    int shared = c.vocab_size > 0 ? 1 : 0; if (c.vocab_size < 0) c.vocab_size = -c.vocab_size;

    /* --- upload weights to the device, map device pointers --- */
    char *d_data; CUDA_CHECK(cudaMalloc(&d_data, fsz));
    CUDA_CHECK(cudaMemcpy(d_data, data, fsz, cudaMemcpyHostToDevice));
    TransformerWeights dw;
    map_weights(&dw, &c, (float*)(d_data + sizeof(Config)), shared);

    /* --- allocate device run state --- */
    int kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;
    DeviceState s;
    CUDA_CHECK(cudaMalloc(&s.x,   c.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.xb,  c.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.xb2, c.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.hb,  c.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.hb2, c.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.q,   c.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.logits, c.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.key_cache,   (long long)c.n_layers * c.seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.value_cache, (long long)c.n_layers * c.seq_len * kv_dim * sizeof(float)));

    Tokenizer tok; build_tokenizer(&tok, "tokenizer.bin", c.vocab_size);
    Sampler smp; smp.vocab_size = c.vocab_size; smp.temperature = temperature;
    smp.topp = topp; smp.rng = seed; smp.probindex = (ProbIndex*)malloc(c.vocab_size * sizeof(ProbIndex));

    float *logits_host = (float*)malloc(c.vocab_size * sizeof(float));
    int *prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    int n_prompt = 0;
    encode(&tok, prompt, 1, 0, prompt_tokens, &n_prompt);

    printf("GPU: prompt \"%s\" (temp=%.2f, top-p=%.2f)\n", prompt, temperature, topp);
    printf("--------------------------------------------------\n");

    int token = prompt_tokens[0], pos = 0, steps = c.seq_len, generated = 0;
    clock_t start = 0;
    while (pos < steps) {
        forward_cuda(&c, &dw, &s, token, pos, logits_host);
        int next;
        if (pos < n_prompt - 1) next = prompt_tokens[pos + 1];
        else { next = sample(&smp, logits_host); generated++; }
        pos++;
        if (next == 1) break;
        printf("%s", decode(&tok, token, next)); fflush(stdout);
        token = next;
        if (start == 0) start = clock();
    }
    printf("\n");
    if (generated > 1) {
        double secs = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("\n[%d tokens, %.1f tok/s on GPU]\n", pos, pos / secs);
    }

    /* (skipping frees — process exit reclaims everything) */
    return 0;
}
