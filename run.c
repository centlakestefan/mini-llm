/* run.c — a tiny llama.cpp-style inference engine, written from scratch in C.
 *
 * Loads a Llama-architecture model and generates text on the CPU: tokenize a
 * prompt, run the transformer forward one token at a time, sample the next
 * token from the logits, decode, repeat.
 *
 * Two checkpoint formats are auto-detected by a magic number:
 *   - legacy llama2.c (e.g. TinyStories stories15M): interleaved RoPE, no biases.
 *   - v2 "qwn2" (e.g. Qwen2.5): rotate-half RoPE, q/k/v biases, configurable
 *     rope_theta, grouped-query attention, tied classifier.
 *
 * Usage:  run [-v] [-m model.bin] [prompt] [temperature]
 *   -v            print a per-token trace (prefill/decode markers + confidence)
 *   -m FILE       checkpoint to load (default: stories15M.bin)
 *   -probe "ids"  feed comma-separated token ids, greedy-decode, print ids
 *   -n N          number of steps for -probe (default 32)
 *   prompt        text to continue (default: "Once upon a time")
 *   temperature   0 = greedy (default, deterministic); >0 samples, e.g. 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>

#define QWN2_MAGIC 0x716E3277u   /* "qwn2", little-endian, marks the v2 format */

/* 64-bit file offsets: plain fseek/ftell use a 32-bit long on Windows and
 * overflow past 2 GB, and a single fread of multiple GB is not portable. */
#ifdef _WIN32
  #define FSEEK64 _fseeki64
  #define FTELL64 _ftelli64
#else
  #define FSEEK64 fseeko
  #define FTELL64 ftello
#endif

/* ---- architecture hyperparameters ---- */
typedef struct {
    int dim;        /* transformer width: size of the activation vector x      */
    int hidden_dim; /* inner dimension of the feed-forward network             */
    int n_layers;   /* number of transformer blocks                            */
    int n_heads;    /* number of query heads                                   */
    int n_kv_heads; /* number of key/value heads (<= n_heads => grouped-query) */
    int vocab_size; /* number of distinct tokens                               */
    int seq_len;    /* maximum context length                                  */
    /* --- fields below are set by the loader, not read verbatim from disk --- */
    int head_dim;     /* size of one attention head (may differ from dim/heads) */
    float rope_theta; /* RoPE base frequency (10000 legacy, 1e6 for Qwen)       */
    float rms_eps;    /* RMSNorm epsilon (1e-5 legacy, 1e-6 for Qwen)           */
    int rope_neox;    /* 1 = HF rotate-half RoPE; 0 = llama2.c interleaved      */
    int has_qkv_bias; /* 1 = add per-head q/k/v bias (Qwen); 0 = none           */
    int v2;           /* 1 = v2 "qwn2" chat model; 0 = legacy story model       */
} Config;

/* ---- pointers into the big weight buffer, one per tensor ----
 * Shapes are written as comments. Per-layer tensors are stored contiguously
 * for all layers, so e.g. wq holds n_layers blocks of (dim x att_dim) back to
 * back, where att_dim = n_heads*head_dim and kv_dim = n_kv_heads*head_dim. */
typedef struct {
    float *token_embedding_table; /* (vocab_size, dim)   token id -> vector     */
    float *rms_att_weight;        /* (n_layers, dim)     RMSNorm before attn    */
    float *wq;                    /* (n_layers, att_dim, dim)                   */
    float *wk;                    /* (n_layers, kv_dim, dim)                    */
    float *wv;                    /* (n_layers, kv_dim, dim)                    */
    float *wo;                    /* (n_layers, dim, att_dim)                   */
    float *rms_ffn_weight;        /* (n_layers, dim)     RMSNorm before FFN     */
    float *w1;                    /* (n_layers, hidden_dim, dim)  FFN gate      */
    float *w2;                    /* (n_layers, dim, hidden_dim)  FFN down      */
    float *w3;                    /* (n_layers, hidden_dim, dim)  FFN up        */
    float *rms_final_weight;      /* (dim,)              final RMSNorm          */
    float *bq;                    /* (n_layers, att_dim) q bias  (NULL if none) */
    float *bk;                    /* (n_layers, kv_dim)  k bias  (NULL if none) */
    float *bv;                    /* (n_layers, kv_dim)  v bias  (NULL if none) */
    float *wcls;                  /* (vocab_size, dim)   classifier (may alias  */
                                  /*                     token_embedding_table) */
} TransformerWeights;

/* ---- scratch buffers, overwritten every token ---- */
typedef struct {
    float *x;   /* (dim,)        current activation                            */
    float *xb;  /* (dim,)        activation inside a residual branch           */
    float *xb2; /* (dim,)        spare buffer                                  */
    float *hb;  /* (hidden_dim,) FFN hidden buffer                             */
    float *hb2; /* (hidden_dim,) FFN hidden buffer                             */
    float *q;   /* (att_dim,)    query for the current token                   */
    float *att; /* (n_heads, seq_len)  attention scores                        */
    float *logits; /* (vocab_size,)    output scores over the vocabulary       */
    float *key_cache;   /* (n_layers, seq_len, kv_dim)  cached keys            */
    float *value_cache; /* (n_layers, seq_len, kv_dim)  cached values          */
} RunState;

/* Everything about a loaded model: config, weight pointers, scratch, and the
 * raw file buffer the weight pointers point into. */
typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    float *data;            /* the whole checkpoint file in memory            */
    long long file_size;    /* its size in bytes                              */
} Transformer;

/* Point each weight pointer at the right place in the flat float buffer `p`,
 * advancing `p` past each tensor. The order MUST match how the file was written. */
static void memory_map_weights(TransformerWeights *w, Config *c, float *p, int shared_weights) {
    long long head_dim = c->head_dim;
    long long att_dim  = (long long)c->n_heads * head_dim;
    long long kv_dim   = (long long)c->n_kv_heads * head_dim;
    long long n_layers = c->n_layers;

    w->token_embedding_table = p; p += (long long)c->vocab_size * c->dim;
    w->rms_att_weight = p;        p += n_layers * c->dim;
    w->wq = p;                    p += n_layers * c->dim * att_dim;
    w->wk = p;                    p += n_layers * c->dim * kv_dim;
    w->wv = p;                    p += n_layers * c->dim * kv_dim;
    w->wo = p;                    p += n_layers * att_dim * c->dim;
    w->rms_ffn_weight = p;        p += n_layers * c->dim;
    w->w1 = p;                    p += n_layers * c->dim * c->hidden_dim;
    w->w2 = p;                    p += n_layers * c->hidden_dim * c->dim;
    w->w3 = p;                    p += n_layers * c->dim * c->hidden_dim;
    w->rms_final_weight = p;      p += c->dim;

    if (c->has_qkv_bias) {
        w->bq = p; p += n_layers * att_dim;
        w->bk = p; p += n_layers * kv_dim;
        w->bv = p; p += n_layers * kv_dim;
    } else {
        w->bq = w->bk = w->bv = NULL;
        p += (long long)c->seq_len * head_dim / 2; /* skip legacy freq_cis_real */
        p += (long long)c->seq_len * head_dim / 2; /* skip legacy freq_cis_imag */
    }
    w->wcls = shared_weights ? w->token_embedding_table : p;
}

/* Read the whole checkpoint into memory and wire up the weight pointers. */
static void read_checkpoint(const char *path, Transformer *t) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "could not open '%s'\n", path); exit(1); }

    FSEEK64(f, 0, SEEK_END);
    t->file_size = FTELL64(f);
    FSEEK64(f, 0, SEEK_SET);

    t->data = malloc(t->file_size);
    if (!t->data) { fprintf(stderr, "out of memory (%lld bytes)\n", t->file_size); exit(1); }
    /* read in 256 MB chunks; a single multi-GB fread is not portable */
    long long got = 0;
    const size_t CHUNK = 256ULL * 1024 * 1024;
    while (got < t->file_size) {
        size_t want = (size_t)((t->file_size - got) < (long long)CHUNK ? (t->file_size - got) : CHUNK);
        size_t r = fread((char *)t->data + got, 1, want, f);
        if (r != want) { fprintf(stderr, "short read on checkpoint at %lld\n", got); exit(1); }
        got += r;
    }
    fclose(f);

    Config *c = &t->config;
    unsigned int magic = *(unsigned int *)t->data;
    int shared_weights;
    long long header_bytes;
    float *weights_ptr;

    if (magic == QWN2_MAGIC) {
        /* v2 header: 11 int32 + 1 float32 (48 bytes). h[0]=magic, h[1]=version. */
        int *h = (int *)t->data;
        c->dim = h[2]; c->hidden_dim = h[3]; c->n_layers = h[4]; c->n_heads = h[5];
        c->n_kv_heads = h[6]; c->vocab_size = h[7]; c->seq_len = h[8]; c->head_dim = h[9];
        shared_weights = h[10];
        memcpy(&c->rope_theta, &h[11], sizeof(float));
        c->rms_eps = 1e-6f; c->rope_neox = 1; c->has_qkv_bias = 1; c->v2 = 1;
        header_bytes = 48;
    } else {
        /* legacy llama2.c header: 7 int32 (28 bytes); sign of vocab = tied. */
        int *h = (int *)t->data;
        c->dim = h[0]; c->hidden_dim = h[1]; c->n_layers = h[2]; c->n_heads = h[3];
        c->n_kv_heads = h[4]; c->vocab_size = h[5]; c->seq_len = h[6];
        shared_weights = c->vocab_size > 0 ? 1 : 0;
        if (c->vocab_size < 0) c->vocab_size = -c->vocab_size;
        c->head_dim = c->dim / c->n_heads;
        c->rope_theta = 10000.0f; c->rms_eps = 1e-5f; c->rope_neox = 0; c->has_qkv_bias = 0;
        c->v2 = 0;
        header_bytes = 28;
    }

    weights_ptr = (float *)((char *)t->data + header_bytes);
    memory_map_weights(&t->weights, c, weights_ptr, shared_weights);

    /* --- verify our layout against the real file size --- */
    long long head_dim = c->head_dim;
    long long att_dim = (long long)c->n_heads * head_dim;
    long long kv_dim  = (long long)c->n_kv_heads * head_dim;
    long long n = c->n_layers;
    long long floats =
        (long long)c->vocab_size * c->dim +   /* token_embedding */
        n * c->dim +                          /* rms_att         */
        n * c->dim * att_dim +                /* wq              */
        n * c->dim * kv_dim +                 /* wk              */
        n * c->dim * kv_dim +                 /* wv              */
        n * att_dim * c->dim +                /* wo              */
        n * c->dim +                          /* rms_ffn         */
        n * c->dim * c->hidden_dim +          /* w1              */
        n * c->hidden_dim * c->dim +          /* w2              */
        n * c->dim * c->hidden_dim +          /* w3              */
        c->dim;                               /* rms_final       */
    if (c->has_qkv_bias) floats += n * att_dim + 2 * n * kv_dim;       /* bq,bk,bv */
    else                 floats += (long long)c->seq_len * head_dim;   /* freq tables */
    if (!shared_weights) floats += (long long)c->vocab_size * c->dim;  /* wcls    */
    long long expected = header_bytes + floats * (long long)sizeof(float);

    printf("checkpoint layout check: expected %lld bytes, file is %lld bytes -> %s\n",
           expected, t->file_size, expected == t->file_size ? "MATCH" : "MISMATCH!");
    if (expected != t->file_size) {
        fprintf(stderr, "layout mismatch: our tensor map does not fit the file\n");
        exit(1);
    }
}

/* Allocate the scratch buffers. kv_dim is the width of one token's K (or V)
 * across all kv heads — smaller than att_dim when grouped-query attention is used. */
static void malloc_run_state(RunState *s, Config *c) {
    int att_dim = c->n_heads * c->head_dim;
    int kv_dim  = c->n_kv_heads * c->head_dim;
    s->x   = calloc(c->dim, sizeof(float));
    s->xb  = calloc(att_dim > c->dim ? att_dim : c->dim, sizeof(float));
    s->xb2 = calloc(c->dim, sizeof(float));
    s->hb  = calloc(c->hidden_dim, sizeof(float));
    s->hb2 = calloc(c->hidden_dim, sizeof(float));
    s->q   = calloc(att_dim, sizeof(float));
    s->att = calloc((long long)c->n_heads * c->seq_len, sizeof(float));
    s->logits = calloc(c->vocab_size, sizeof(float));
    s->key_cache   = calloc((long long)c->n_layers * c->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc((long long)c->n_layers * c->seq_len * kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
        !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "run state allocation failed\n");
        exit(1);
    }

    long long kv_floats = 2LL * c->n_layers * c->seq_len * kv_dim;
    printf("run state: kv_dim=%d, KV cache = %lld floats (%.2f MB)\n",
           kv_dim, kv_floats, kv_floats * sizeof(float) / (1024.0 * 1024.0));
}

static void free_run_state(RunState *s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->q); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
}

static void free_transformer(Transformer *t) {
    free_run_state(&t->state);
    free(t->data);
}

/* ---------------------------- arithmetic primitives ---------------------------- */

/* RMSNorm: o[j] = weight[j] * x[j] / sqrt(mean(x^2) + eps)
 * Rescales the vector to a stable magnitude, then applies a learned per-channel
 * gain. eps avoids division by zero (1e-5 for stories, 1e-6 for Qwen). */
static void rmsnorm(float *o, float *x, float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = ss / size + eps;
    ss = 1.0f / sqrtf(ss);              /* the reciprocal RMS, a single scalar */
    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

/* matmul: xout = W @ x, where W is (d, n) row-major, x is (n,), xout is (d,).
 * i.e. xout[i] = sum over j of W[i*n + j] * x[j].
 * This single function performs every learned projection in the network and is
 * where essentially all the compute lives. */
static void matmul(float *xout, float *x, float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++)
            val += w[(long long)i * n + j] * x[j];
        xout[i] = val;
    }
}

/* softmax in place: x <- exp(x - max) / sum(...). The max subtraction keeps
 * exp() from overflowing; it doesn't change the result. */
static void softmax(float *x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

/* Apply rotary position embeddings to one head's vector in the HF "rotate-half"
 * (GPT-NeoX) convention: element i pairs with element i+half, both rotated by
 * the same angle pos * theta^(-2i/head_dim). */
static void rope_neox_head(float *vec, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
        float angle = pos * freq;
        float ca = cosf(angle), sa = sinf(angle);
        float x0 = vec[i], x1 = vec[i + half];
        vec[i]        = x0 * ca - x1 * sa;
        vec[i + half] = x1 * ca + x0 * sa;
    }
}

/* The full forward pass for one token at position `pos`: embedding lookup, then
 * each layer (RMSNorm -> attention with RoPE + KV cache -> residual, then
 * RMSNorm -> SwiGLU feed-forward -> residual), and finally a RMSNorm + classifier
 * that leaves the next-token logits in s->logits. */
static void forward(Transformer *t, int token, int pos) {
    Config *c = &t->config;
    TransformerWeights *w = &t->weights;
    RunState *s = &t->state;
    int dim = c->dim;
    int head_dim = c->head_dim;
    int att_dim = c->n_heads * head_dim;
    int kv_dim = c->n_kv_heads * head_dim;
    int kv_mul = c->n_heads / c->n_kv_heads; /* query heads sharing one kv head */

    /* embedding lookup: x <- row `token` of the embedding table */
    memcpy(s->x, w->token_embedding_table + (long long)token * dim, dim * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        /* --- attention RMSNorm --- */
        rmsnorm(s->xb, s->x, w->rms_att_weight + (long long)l * dim, dim, c->rms_eps);

        /* this layer's K/V slots in the cache for the current position */
        long long loff = (long long)l * c->seq_len * kv_dim;
        float *k = s->key_cache   + loff + (long long)pos * kv_dim;
        float *v = s->value_cache + loff + (long long)pos * kv_dim;

        /* --- Q, K, V projections (+ optional bias) --- */
        matmul(s->q, s->xb, w->wq + (long long)l * dim * att_dim, dim, att_dim);
        matmul(k,    s->xb, w->wk + (long long)l * dim * kv_dim,  dim, kv_dim);
        matmul(v,    s->xb, w->wv + (long long)l * dim * kv_dim,  dim, kv_dim);
        if (c->has_qkv_bias) {
            float *bq = w->bq + (long long)l * att_dim;
            float *bk = w->bk + (long long)l * kv_dim;
            float *bv = w->bv + (long long)l * kv_dim;
            for (int i = 0; i < att_dim; i++) s->q[i] += bq[i];
            for (int i = 0; i < kv_dim; i++) { k[i] += bk[i]; v[i] += bv[i]; }
        }

        /* --- RoPE: rotate q and k by an angle ~ pos --- */
        if (c->rope_neox) {
            for (int h = 0; h < c->n_heads; h++)
                rope_neox_head(s->q + h * head_dim, head_dim, pos, c->rope_theta);
            for (int h = 0; h < c->n_kv_heads; h++)
                rope_neox_head(k + h * head_dim, head_dim, pos, c->rope_theta);
        } else {
            /* llama2.c interleaved convention: rotate pairs (i, i+1) */
            for (int i = 0; i < att_dim; i += 2) {
                int hd = i % head_dim;
                float freq = 1.0f / powf(c->rope_theta, hd / (float)head_dim);
                float angle = pos * freq;
                float fcr = cosf(angle), fci = sinf(angle);
                int rotn = (i < kv_dim) ? 2 : 1;       /* rotate q (and k if in range) */
                for (int vv = 0; vv < rotn; vv++) {
                    float *vec = (vv == 0) ? s->q : k;
                    float v0 = vec[i], v1 = vec[i + 1];
                    vec[i]     = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
        }

        /* --- multi-head self-attention --- */
        for (int h = 0; h < c->n_heads; h++) {
            float *q   = s->q   + h * head_dim;              /* this head's query  */
            float *att = s->att + (long long)h * c->seq_len; /* this head's scores */

            /* score query against every key from position 0..pos */
            for (int t2 = 0; t2 <= pos; t2++) {
                float *kk = s->key_cache + loff + (long long)t2 * kv_dim
                          + (h / kv_mul) * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) score += q[i] * kk[i];
                att[t2] = score / sqrtf((float)head_dim);
            }
            softmax(att, pos + 1);                           /* scores -> weights */

            /* weighted sum of the values -> this head's slice of xb */
            float *xb = s->xb + h * head_dim;
            for (int i = 0; i < head_dim; i++) xb[i] = 0.0f;
            for (int t2 = 0; t2 <= pos; t2++) {
                float *vv2 = s->value_cache + loff + (long long)t2 * kv_dim
                           + (h / kv_mul) * head_dim;
                float a = att[t2];
                for (int i = 0; i < head_dim; i++) xb[i] += a * vv2[i];
            }
        }

        /* --- output projection back into the residual stream --- */
        matmul(s->xb2, s->xb, w->wo + (long long)l * att_dim * dim, att_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        /* --- FFN sub-layer: RMSNorm -> SwiGLU -> residual --- */
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + (long long)l * dim, dim, c->rms_eps);

        int hidden_dim = c->hidden_dim;
        matmul(s->hb,  s->xb, w->w1 + (long long)l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + (long long)l * dim * hidden_dim, dim, hidden_dim);

        /* SwiGLU: hb <- silu(w1·x) * (w3·x), where silu(z) = z * sigmoid(z) */
        for (int i = 0; i < hidden_dim; i++) {
            float z = s->hb[i];
            z *= 1.0f / (1.0f + expf(-z));   /* SiLU activation */
            s->hb[i] = z * s->hb2[i];        /* gated by the w3 projection */
        }

        matmul(s->xb, s->hb, w->w2 + (long long)l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    /* --- final RMSNorm, then classifier: x -> logits over the vocabulary --- */
    rmsnorm(s->x, s->x, w->rms_final_weight, dim, c->rms_eps);
    matmul(s->logits, s->x, w->wcls, dim, c->vocab_size);
}

/* ================================ tokenizer ================================ */
/* NOTE: this is the SentencePiece-style tokenizer for the legacy stories models.
 * The Qwen byte-level BPE tokenizer is added in a later step. */

typedef struct {
    char *str; /* the vocab string */
    int id;    /* its token id     */
} TokenIndex;

typedef struct {
    char **vocab;          /* vocab[id] -> string piece                     */
    float *vocab_scores;   /* merge priority for each piece                 */
    TokenIndex *sorted;    /* vocab sorted by string, for bsearch on encode */
    int vocab_size;
    unsigned int max_token_length;
    char byte_pieces[512]; /* 256 single-byte strings, for <0xXX> tokens    */
} Tokenizer;

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

static int str_lookup(char *str, TokenIndex *sorted, int n) {
    TokenIndex key; key.str = str;
    TokenIndex *res = bsearch(&key, sorted, n, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

static void build_tokenizer(Tokenizer *t, const char *path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->vocab_scores = malloc(vocab_size * sizeof(float));
    t->sorted = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "could not open tokenizer '%s'\n", path); exit(1); }
    if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) { fprintf(stderr, "bad tokenizer\n"); exit(1); }
    for (int i = 0; i < vocab_size; i++) {
        int len;
        fread(t->vocab_scores + i, sizeof(float), 1, f);
        fread(&len, sizeof(int), 1, f);
        t->vocab[i] = malloc(len + 1);
        fread(t->vocab[i], 1, len, f);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

static void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted);
}

static char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;     /* strip space after BOS */
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)     /* a raw-byte token */
        piece = (char *)t->byte_pieces + byte_val * 2;
    return piece;
}

static void encode(Tokenizer *t, char *text, int bos, int eos, int *tokens, int *n_tokens) {
    if (t->sorted == NULL) {
        t->sorted = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted[i].str = t->vocab[i];
            t->sorted[i].id = i;
        }
        qsort(t->sorted, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char *buf = malloc(t->max_token_length * 2 + 3);
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;                  /* BOS */

    if (text[0] != '\0') {
        int sp = str_lookup(" ", t->sorted, t->vocab_size);
        tokens[(*n_tokens)++] = sp;
    }

    size_t len = 0;
    for (char *cc = text; *cc != '\0'; cc++) {
        if ((*cc & 0xC0) != 0x80) len = 0;
        buf[len++] = *cc;
        buf[len] = '\0';
        if ((*(cc + 1) & 0xC0) == 0x80 && len < 4) continue;
        int id = str_lookup(buf, t->sorted, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < len; i++)
                tokens[(*n_tokens)++] = (unsigned char)buf[i] + 3; /* byte fallback */
        }
        len = 0;
    }

    while (1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(buf, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(buf, t->sorted, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;                  /* EOS */
    free(buf);
}

/* ===================== Qwen byte-level BPE tokenizer ===================== */
/* GPT-2 style: map every raw byte to a printable "byte-unicode" char, split
 * text with Qwen's pre-tokenizer regex, then merge pieces by learned rank.
 * Loaded from qwen_tokenizer.bin (see export_tokenizer.py). */

/* --- a tiny open-addressing string->int hash map (keys are not owned) --- */
typedef struct { const char *key; int val; } HEntry;
typedef struct { HEntry *e; int cap; } HMap;

static unsigned long long fnv1a(const char *s, int len) {
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < len; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}
static void hmap_init(HMap *m, int cap) { m->cap = cap; m->e = calloc(cap, sizeof(HEntry)); }
static void hmap_put(HMap *m, const char *key, int val) {
    int i = (int)(fnv1a(key, (int)strlen(key)) % (unsigned)m->cap);
    while (m->e[i].key) i = (i + 1) % m->cap;
    m->e[i].key = key; m->e[i].val = val;
}
/* look up a (possibly non-terminated) key of given length */
static int hmap_get(HMap *m, const char *key, int len) {
    int i = (int)(fnv1a(key, len) % (unsigned)m->cap);
    while (m->e[i].key) {
        if ((int)strlen(m->e[i].key) == len && memcmp(m->e[i].key, key, len) == 0)
            return m->e[i].val;
        i = (i + 1) % m->cap;
    }
    return -1;
}

typedef struct {
    char **vocab;          /* id -> byte-unicode string ("" for unused ids) */
    int vocab_size;
    char **merge_str;      /* rank -> "left right" string                   */
    int n_merges;
    char **spec_str;       /* special token contents                        */
    int *spec_id;
    int *spec_len;
    int n_special;
    HMap vocab2id;         /* byte-unicode piece -> id                      */
    HMap merge2rank;       /* "left right" -> rank                          */
    int byte_to_cp[256];   /* GPT-2 byte -> unicode codepoint               */
    int cp_to_byte[1024];  /* and back (-1 if none)                         */
} QTokenizer;

static int qt_is_space(int cp){ return cp==' '||cp=='\t'||cp=='\n'||cp=='\r'||cp==0x0b||cp==0x0c; }
static int qt_is_digit(int cp){ return cp>='0' && cp<='9'; }
/* \p{L} approximation: ASCII letters plus any non-ASCII codepoint */
static int qt_is_letter(int cp){ return (cp>='A'&&cp<='Z')||(cp>='a'&&cp<='z')||cp>=0x80; }
static int qt_is_punct(int cp){ return !qt_is_space(cp)&&!qt_is_letter(cp)&&!qt_is_digit(cp); }

/* decode one UTF-8 codepoint from s[p..L); set *adv to bytes consumed */
static int utf8_next(const unsigned char *s, int L, int p, int *adv) {
    unsigned char c = s[p];
    if (c < 0x80) { *adv = 1; return c; }
    if ((c >> 5) == 0x6 && p+1 < L) { *adv = 2; return ((c&0x1f)<<6)|(s[p+1]&0x3f); }
    if ((c >> 4) == 0xe && p+2 < L) { *adv = 3; return ((c&0x0f)<<12)|((s[p+1]&0x3f)<<6)|(s[p+2]&0x3f); }
    if ((c >> 3) == 0x1e && p+3 < L) { *adv = 4; return ((c&0x07)<<18)|((s[p+1]&0x3f)<<12)|((s[p+2]&0x3f)<<6)|(s[p+3]&0x3f); }
    *adv = 1; return c;
}
/* encode a codepoint (< 0x800 here) as UTF-8; return bytes written */
static int utf8_enc(int cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3f)); return 2;
}

static char *read_lenstr(FILE *f) {
    int len; if (fread(&len, 4, 1, f) != 1) { fprintf(stderr, "bad tokenizer\n"); exit(1); }
    char *s = malloc(len + 1);
    if (len) fread(s, 1, len, f);
    s[len] = '\0';
    return s;
}

static void qt_build(QTokenizer *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "could not open '%s' (run export_tokenizer.py)\n", path); exit(1); }
    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic, "QTK1", 4) != 0) { fprintf(stderr, "bad tokenizer magic\n"); exit(1); }
    fread(&t->vocab_size, 4, 1, f);
    fread(&t->n_merges, 4, 1, f);
    fread(&t->n_special, 4, 1, f);

    t->vocab = malloc(t->vocab_size * sizeof(char *));
    hmap_init(&t->vocab2id, 1 << 19);
    for (int i = 0; i < t->vocab_size; i++) {
        t->vocab[i] = read_lenstr(f);
        if (t->vocab[i][0] != '\0') hmap_put(&t->vocab2id, t->vocab[i], i);
    }
    t->merge_str = malloc(t->n_merges * sizeof(char *));
    hmap_init(&t->merge2rank, 1 << 19);
    for (int i = 0; i < t->n_merges; i++) {
        t->merge_str[i] = read_lenstr(f);
        hmap_put(&t->merge2rank, t->merge_str[i], i);
    }
    t->spec_str = malloc(t->n_special * sizeof(char *));
    t->spec_id  = malloc(t->n_special * sizeof(int));
    t->spec_len = malloc(t->n_special * sizeof(int));
    for (int i = 0; i < t->n_special; i++) {
        t->spec_str[i] = read_lenstr(f);
        fread(&t->spec_id[i], 4, 1, f);
        t->spec_len[i] = (int)strlen(t->spec_str[i]);
    }
    fclose(f);

    /* GPT-2 bytes<->unicode: printable bytes map to themselves, the rest to
     * 256, 257, ... so every byte becomes a single printable codepoint. */
    for (int i = 0; i < 1024; i++) t->cp_to_byte[i] = -1;
    for (int b = 0; b < 256; b++)
        t->byte_to_cp[b] = ((b>=0x21&&b<=0x7e)||(b>=0xa1&&b<=0xac)||(b>=0xae&&b<=0xff)) ? b : -1;
    int n = 0;
    for (int b = 0; b < 256; b++) if (t->byte_to_cp[b] == -1) t->byte_to_cp[b] = 256 + n++;
    for (int b = 0; b < 256; b++) t->cp_to_byte[t->byte_to_cp[b]] = b;
}

/* BPE-encode one pre-token (raw bytes), appending token ids to ids[]. */
static void bpe_piece(QTokenizer *t, const char *bytes, int blen, int *ids, int *n) {
    if (blen == 0) return;
    /* map raw bytes -> byte-unicode buffer u; each byte is one symbol */
    char *u = malloc(blen * 2 + 1);
    int *off = malloc(blen * sizeof(int));
    int *len = malloc(blen * sizeof(int));
    int ulen = 0, nsym = 0;
    for (int i = 0; i < blen; i++) {
        off[nsym] = ulen;
        ulen += utf8_enc(t->byte_to_cp[(unsigned char)bytes[i]], u + ulen);
        len[nsym] = ulen - off[nsym];
        nsym++;
    }
    char *key = malloc(blen * 2 + 2);   /* scratch for "left right" lookups */

    while (nsym > 1) {
        int best_rank = INT_MAX, best = -1;
        for (int i = 0; i < nsym - 1; i++) {
            int kl = 0;
            memcpy(key + kl, u + off[i], len[i]);   kl += len[i];
            key[kl++] = ' ';
            memcpy(key + kl, u + off[i+1], len[i+1]); kl += len[i+1];
            int rank = hmap_get(&t->merge2rank, key, kl);
            if (rank >= 0 && rank < best_rank) { best_rank = rank; best = i; }
        }
        if (best < 0) break;
        len[best] += len[best+1];                   /* symbols are contiguous in u */
        for (int j = best+1; j < nsym-1; j++) { off[j] = off[j+1]; len[j] = len[j+1]; }
        nsym--;
    }
    for (int i = 0; i < nsym; i++) {
        int id = hmap_get(&t->vocab2id, u + off[i], len[i]);
        if (id >= 0) ids[(*n)++] = id;
    }
    free(u); free(off); free(len); free(key);
}

/* Pre-tokenize a special-token-free segment (Qwen's regex), BPE each piece. */
static void qt_encode_segment(QTokenizer *t, const char *seg, int slen, int *ids, int *n) {
    int i = 0;
    while (i < slen) {
        int adv; int cp = utf8_next((const unsigned char *)seg, slen, i, &adv);
        int start = i;

        /* 's 't 're 've 'm 'll 'd  (case-insensitive) */
        if (cp == '\'' && i + 1 < slen) {
            char la = (char)tolower((unsigned char)seg[i+1]);
            char lb = (i + 2 < slen) ? (char)tolower((unsigned char)seg[i+2]) : 0;
            int take = 0;
            if ((la=='r'&&lb=='e')||(la=='v'&&lb=='e')||(la=='l'&&lb=='l')) take = 2;
            else if (la=='s'||la=='t'||la=='m'||la=='d') take = 1;
            if (take) { bpe_piece(t, seg+start, 1+take, ids, n); i += 1+take; continue; }
        }
        /* letters (\p{L}+) */
        if (qt_is_letter(cp)) {
            i += adv;
            while (i < slen) { int a; int c = utf8_next((const unsigned char *)seg, slen, i, &a);
                               if (qt_is_letter(c)) i += a; else break; }
            bpe_piece(t, seg+start, i-start, ids, n); continue;
        }
        /* one leading non-CRLF/non-letter/non-digit char then letters */
        if (cp != '\r' && cp != '\n' && !qt_is_digit(cp)) {
            int a; int j = i + adv;
            if (j < slen) { int c = utf8_next((const unsigned char *)seg, slen, j, &a);
                if (qt_is_letter(c)) {
                    int k = j;
                    while (k < slen) { int a2; int c2 = utf8_next((const unsigned char *)seg, slen, k, &a2);
                                       if (qt_is_letter(c2)) k += a2; else break; }
                    bpe_piece(t, seg+start, k-start, ids, n); i = k; continue;
                }
            }
        }
        /* single digit (\p{N}) */
        if (qt_is_digit(cp)) { bpe_piece(t, seg+start, adv, ids, n); i += adv; continue; }

        /* optional leading space then punctuation run then trailing newlines */
        {
            int k = i, ok = 0;
            if (cp == ' ' && i + adv < slen) {
                int a; int c = utf8_next((const unsigned char *)seg, slen, i+adv, &a);
                if (qt_is_punct(c)) { ok = 1; k = i + adv; }
            }
            if (!ok && qt_is_punct(cp)) { ok = 1; k = i; }
            if (ok) {
                int m = k;
                while (m < slen) { int a; int c = utf8_next((const unsigned char *)seg, slen, m, &a);
                                   if (qt_is_punct(c)) m += a; else break; }
                while (m < slen && (seg[m]=='\r' || seg[m]=='\n')) m++;
                bpe_piece(t, seg+start, m-start, ids, n); i = m; continue;
            }
        }
        /* whitespace run (\s+); hand the last space to the next token if it
         * is followed by a letter or punctuation (GPT-2 leading-space rule) */
        {
            int e = i;
            while (e < slen) { int a; int c = utf8_next((const unsigned char *)seg, slen, e, &a);
                               if (qt_is_space(c)) e += a; else break; }
            if (e < slen && seg[e-1] == ' ' && (e-1) > i) {
                int a; int c = utf8_next((const unsigned char *)seg, slen, e, &a);
                if (qt_is_letter(c) || qt_is_punct(c)) {
                    bpe_piece(t, seg+start, (e-1)-start, ids, n); i = e-1; continue;
                }
            }
            bpe_piece(t, seg+start, e-i, ids, n); i = e; continue;
        }
    }
}

/* Encode full text to token ids, splitting on special tokens first. */
static void qt_encode(QTokenizer *t, const char *text, int *ids, int *n) {
    int L = (int)strlen(text), p = 0;
    *n = 0;
    while (p < L) {
        int bs = -1, bi = -1, blen = 0;
        for (int s = p; s < L; s++) {
            for (int k = 0; k < t->n_special; k++)
                if (s + t->spec_len[k] <= L && memcmp(text+s, t->spec_str[k], t->spec_len[k]) == 0
                    && t->spec_len[k] > blen) { bs = s; bi = k; blen = t->spec_len[k]; }
            if (bs == s) break;       /* earliest match wins (longest at that spot) */
        }
        if (bs >= 0) {
            if (bs > p) qt_encode_segment(t, text+p, bs-p, ids, n);
            ids[(*n)++] = t->spec_id[bi];
            p = bs + blen;
        } else {
            qt_encode_segment(t, text+p, L-p, ids, n);
            break;
        }
    }
}

/* Decode one token id to raw bytes (appended to out); returns bytes written.
 * Maps byte-unicode codepoints back to the original bytes. */
static int qt_decode(QTokenizer *t, int id, char *out) {
    if (id < 0 || id >= t->vocab_size) return 0;
    const unsigned char *s = (const unsigned char *)t->vocab[id];
    int L = (int)strlen((const char *)s), p = 0, o = 0;
    while (p < L) {
        int adv; int cp = utf8_next(s, L, p, &adv); p += adv;
        int b = (cp >= 0 && cp < 1024) ? t->cp_to_byte[cp] : -1;
        if (b >= 0) out[o++] = (char)b;
    }
    return o;
}

static void qt_free(QTokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    for (int i = 0; i < t->n_merges; i++) free(t->merge_str[i]);
    for (int i = 0; i < t->n_special; i++) free(t->spec_str[i]);
    free(t->vocab); free(t->merge_str); free(t->spec_str);
    free(t->spec_id); free(t->spec_len);
    free(t->vocab2id.e); free(t->merge2rank.e);
}

/* ================================= sampler ================================= */

typedef struct { float prob; int index; } ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;        /* scratch for top-p sorting */
    float temperature;           /* 0 => greedy; <1 sharpen; >1 flatten */
    float topp;                  /* nucleus threshold (e.g. 0.9); <=0 or >=1 disables */
    unsigned long long rng_state;
} Sampler;

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1DULL) >> 32;
}
static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_argmax(float *p, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (p[i] > p[best]) best = i;
    return best;
}

static int sample_mult(float *p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}

static int compare_probindex(const void *a, const void *b) {
    float pa = ((ProbIndex *)a)->prob, pb = ((ProbIndex *)b)->prob;
    return (pa < pb) - (pa > pb);
}

static int sample_topp(float *p, int n, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++)
        if (p[i] >= cutoff) { probindex[n0].index = i; probindex[n0].prob = p[i]; n0++; }
    qsort(probindex, n0, sizeof(ProbIndex), compare_probindex);

    float cumulative = 0.0f;
    int last = n0 - 1;
    for (int i = 0; i < n0; i++) { cumulative += probindex[i].prob; if (cumulative > topp) { last = i; break; } }

    float r = coin * cumulative, cdf = 0.0f;
    for (int i = 0; i <= last; i++) { cdf += probindex[i].prob; if (r < cdf) return probindex[i].index; }
    return probindex[last].index;
}

static void build_sampler(Sampler *s, int vocab_size, float temperature, float topp, unsigned long long seed) {
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->rng_state = seed;
    s->probindex = malloc(vocab_size * sizeof(ProbIndex));
}
static void free_sampler(Sampler *s) { free(s->probindex); }

static int sample(Sampler *s, float *logits) {
    if (s->temperature == 0.0f) return sample_argmax(logits, s->vocab_size);

    for (int i = 0; i < s->vocab_size; i++) logits[i] /= s->temperature;
    softmax(logits, s->vocab_size);
    float coin = random_f32(&s->rng_state);
    if (s->topp <= 0.0f || s->topp >= 1.0f) return sample_mult(logits, s->vocab_size, coin);
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}

/* =============================== generation =============================== */

static void generate(Transformer *t, Tokenizer *tok, Sampler *sampler, char *prompt, int steps, int verbose) {
    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    int n_prompt = 0;
    encode(tok, prompt, /*bos=*/1, /*eos=*/0, prompt_tokens, &n_prompt);
    if (n_prompt < 1) { fprintf(stderr, "empty prompt\n"); exit(1); }

    int V = t->config.vocab_size;

    if (verbose) {
        printf("[encode] \"%s\" -> %d tokens\n", prompt, n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            int tk = prompt_tokens[i];
            const char *pc = (tk == 1) ? "<s>" : decode(tok, 0, tk);
            printf("  [prefill] pos=%2d  id=%5d  \"%s\"\n", i, tk, pc);
        }
        printf("  ----- prompt ingested (%d tokens); now generating -----\n", n_prompt);
    }

    char story[16384]; story[0] = '\0';

    clock_t start = 0;
    int token = prompt_tokens[0];
    int pos = 0, generated = 0;

    while (pos < steps) {
        forward(t, token, pos);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            float prob = 0.0f;
            if (verbose) {
                float *cp = malloc(V * sizeof(float));
                memcpy(cp, t->state.logits, V * sizeof(float));
                softmax(cp, V);
                next = sample(sampler, t->state.logits);
                prob = cp[next];
                free(cp);
            } else {
                next = sample(sampler, t->state.logits);
            }
            generated++;
            if (start == 0) start = clock();
            if (verbose && next != 1) {
                const char *pc = decode(tok, token, next);
                printf("  [decode]  pos=%2d  id=%5d  \"%s\"   p=%.1f%%\n",
                       pos + 1, next, pc, prob * 100.0f);
            }
        }
        pos++;

        if (next == 1) break;

        const char *piece = decode(tok, token, next);
        if (verbose) {
            strncat(story, piece, sizeof(story) - strlen(story) - 1);
        } else {
            printf("%s", piece);
            fflush(stdout);
        }
        token = next;
    }
    if (!verbose) printf("\n");

    if (verbose) {
        double secs = (start && generated) ? (double)(clock() - start) / CLOCKS_PER_SEC : 0.0;
        printf("  ----- done: %d generated, %d total tokens%s",
               generated, pos, secs > 0 ? "" : "\n");
        if (secs > 0) printf(", %.1f tok/s -----\n", generated / secs);
        printf("[output] %s\n", story);
    } else if (generated > 1) {
        double secs = (double)(clock() - start) / CLOCKS_PER_SEC;
        fprintf(stderr, "\n[%d tokens, %.1f tok/s]\n", pos, pos / secs);
    }
    free(prompt_tokens);
}

/* -probe: feed raw token ids (no tokenizer), greedy-decode `steps` tokens, and
 * print the full id stream. Used to validate the forward pass against a known
 * reference before the model's own tokenizer exists. */
static void run_probe(Transformer *t, int *ids, int n_ids, int steps) {
    int eos_a = 151643, eos_b = 151645;   /* Qwen <|endoftext|> / <|im_end|> */
    int token = ids[0];
    int pos = 0;
    printf("[probe ids]");
    while (pos < steps) {
        forward(t, token, pos);
        int next;
        if (pos < n_ids - 1) next = ids[pos + 1];
        else                 next = sample_argmax(t->state.logits, t->config.vocab_size);
        printf(" %d", next);
        pos++;
        if (pos >= n_ids && (next == eos_a || next == eos_b)) break;
        token = next;
    }
    printf("\n");
}

/* =============================== chat (Qwen) =============================== */

#define QWEN_IM_START 151644
#define QWEN_IM_END   151645
#define QWEN_EOT      151643

/* Interactive ChatML loop. The KV cache persists across turns, so the model
 * sees the whole conversation; `pos` is the running position in that cache. */
static void chat(Transformer *t, QTokenizer *tok, Sampler *sampler) {
    int max_seq = t->config.seq_len;
    const char *SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    size_t cap = 1 << 16;
    char *line = malloc(cap);
    char *prompt = malloc(cap + 256);
    int *ids = malloc((cap + 64) * sizeof(int));
    char piece[256];   /* one token can decode to many raw bytes */
    int pos = 0, turn = 0;

    printf("\nQwen2.5-1.5B-Instruct chat. Type a message; /exit or Ctrl+Z to quit.\n");

    while (1) {
        printf("\n> ");
        fflush(stdout);
        if (!fgets(line, (int)cap, stdin)) break;
        int ln = (int)strlen(line);
        while (ln > 0 && (line[ln-1] == '\n' || line[ln-1] == '\r')) line[--ln] = '\0';
        if (ln == 0) continue;
        if (strcmp(line, "/exit") == 0 || strcmp(line, "/quit") == 0) break;

        /* wrap the user turn in ChatML; the first turn carries the system prompt,
         * later turns open with a newline to separate them in the cache */
        if (turn == 0)
            snprintf(prompt, cap + 256,
                "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                SYSTEM, line);
        else
            snprintf(prompt, cap + 256,
                "\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", line);
        turn++;

        int n = 0;
        qt_encode(tok, prompt, ids, &n);
        if (pos + n + 1 >= max_seq) { printf("[context window full, ending chat]\n"); break; }

        /* prefill: feed the prompt tokens, leaving logits that predict the reply */
        for (int i = 0; i < n; i++) { forward(t, ids[i], pos); pos++; }

        /* decode the assistant's reply until <|im_end|> / <|endoftext|> */
        while (pos < max_seq) {
            int next = sample(sampler, t->state.logits);
            forward(t, next, pos);   /* advance the cache with the chosen token */
            pos++;
            if (next == QWEN_IM_END || next == QWEN_EOT) break;
            int b = qt_decode(tok, next, piece);
            fwrite(piece, 1, b, stdout);
            fflush(stdout);
        }
        printf("\n");
    }
    free(line); free(prompt); free(ids);
    printf("\nbye.\n");
}

int main(int argc, char **argv) {
    const char *checkpoint = "stories15M.bin";
    char *prompt = "Once upon a time";
    char *probe_ids = NULL;
    char *encode_text = NULL;
    float temperature = 0.0f;   /* greedy by default: deterministic & reproducible */
    float topp = 0.9f;
    unsigned long long seed = 1234ULL;
    int verbose = 0;
    int probe_steps = 32;

    /* args: [-v] [-m model] [-probe "ids"] [-n steps] [-encode "text"] [prompt] [temp] */
    int positional = 0;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "-v") == 0) verbose = 1;
        else if (strcmp(argv[a], "-m") == 0 && a + 1 < argc) checkpoint = argv[++a];
        else if (strcmp(argv[a], "-probe") == 0 && a + 1 < argc) probe_ids = argv[++a];
        else if (strcmp(argv[a], "-n") == 0 && a + 1 < argc) probe_steps = atoi(argv[++a]);
        else if (strcmp(argv[a], "-encode") == 0 && a + 1 < argc) encode_text = argv[++a];
        else if (positional == 0) { prompt = argv[a]; positional++; }
        else if (positional == 1) { temperature = (float)atof(argv[a]); positional++; }
    }

    /* -encode: tokenize text with the Qwen tokenizer and print ids (no model).
     * "-encode -" reads UTF-8 text from stdin (argv is ANSI-mangled on Windows). */
    if (encode_text) {
        char *buf = encode_text;
        if (strcmp(encode_text, "-") == 0) {
            size_t cap = 4096, len = 0; buf = malloc(cap);
            int ch;
            while ((ch = fgetc(stdin)) != EOF) {
                if (len + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
                buf[len++] = (char)ch;
            }
            buf[len] = '\0';
        }
        QTokenizer qt; qt_build(&qt, "qwen_tokenizer.bin");
        int *ids = malloc((strlen(buf) + 16) * sizeof(int)), n = 0;
        qt_encode(&qt, buf, ids, &n);
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? "," : "\n");
        free(ids); qt_free(&qt);
        return 0;
    }

    Transformer t;
    read_checkpoint(checkpoint, &t);
    malloc_run_state(&t.state, &t.config);

    Config *c = &t.config;
    printf("\nConfig: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d "
           "(head_dim=%d, rope_theta=%.0f, %s RoPE%s)\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads,
           c->vocab_size, c->seq_len, c->head_dim, c->rope_theta,
           c->rope_neox ? "rotate-half" : "interleaved",
           c->has_qkv_bias ? ", qkv-bias" : "");

    if (probe_ids) {
        /* parse comma/space-separated ids */
        int *ids = malloc(strlen(probe_ids) * sizeof(int));
        int n = 0;
        char *buf = strdup(probe_ids);
        for (char *tk = strtok(buf, ", "); tk; tk = strtok(NULL, ", "))
            ids[n++] = atoi(tk);
        free(buf);
        printf("\n[probe] %d input ids, %d steps\n", n, probe_steps);
        run_probe(&t, ids, n, probe_steps);
        free(ids);
        free_transformer(&t);
        return 0;
    }

    Sampler sampler;
    build_sampler(&sampler, c->vocab_size, temperature, topp, seed);

    if (c->v2) {
        /* a real chat model: ChatML conversation loop with the Qwen tokenizer */
        QTokenizer qt;
        qt_build(&qt, "qwen_tokenizer.bin");
        chat(&t, &qt, &sampler);
        qt_free(&qt);
    } else {
        /* a legacy story model: continue the prompt with the SentencePiece tokenizer */
        Tokenizer tok;
        build_tokenizer(&tok, "tokenizer.bin", c->vocab_size);
        if (temperature == 0.0f)
            printf("\nprompt: \"%s\"  (greedy)\n", prompt);
        else
            printf("\nprompt: \"%s\"  (temperature=%.2f, top-p=%.2f)\n", prompt, temperature, topp);
        printf("--------------------------------------------------\n");
        generate(&t, &tok, &sampler, prompt, c->seq_len, verbose);
        free_tokenizer(&tok);
    }

    free_sampler(&sampler);
    free_transformer(&t);
    return 0;
}
