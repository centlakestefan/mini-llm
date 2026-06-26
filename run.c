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
        c->rms_eps = 1e-6f; c->rope_neox = 1; c->has_qkv_bias = 1;
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

int main(int argc, char **argv) {
    const char *checkpoint = "stories15M.bin";
    char *prompt = "Once upon a time";
    char *probe_ids = NULL;
    float temperature = 0.0f;   /* greedy by default: deterministic & reproducible */
    float topp = 0.9f;
    unsigned long long seed = 1234ULL;
    int verbose = 0;
    int probe_steps = 32;

    /* args: [-v] [-m model] [-probe "ids"] [-n steps] [prompt] [temperature] */
    int positional = 0;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "-v") == 0) verbose = 1;
        else if (strcmp(argv[a], "-m") == 0 && a + 1 < argc) checkpoint = argv[++a];
        else if (strcmp(argv[a], "-probe") == 0 && a + 1 < argc) probe_ids = argv[++a];
        else if (strcmp(argv[a], "-n") == 0 && a + 1 < argc) probe_steps = atoi(argv[++a]);
        else if (positional == 0) { prompt = argv[a]; positional++; }
        else if (positional == 1) { temperature = (float)atof(argv[a]); positional++; }
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

    Tokenizer tok;
    build_tokenizer(&tok, "tokenizer.bin", c->vocab_size);

    Sampler sampler;
    build_sampler(&sampler, c->vocab_size, temperature, topp, seed);

    if (temperature == 0.0f)
        printf("\nprompt: \"%s\"  (greedy)\n", prompt);
    else
        printf("\nprompt: \"%s\"  (temperature=%.2f, top-p=%.2f)\n", prompt, temperature, topp);
    printf("--------------------------------------------------\n");
    generate(&t, &tok, &sampler, prompt, c->seq_len, verbose);

    free_sampler(&sampler);
    free_tokenizer(&tok);
    free_transformer(&t);
    return 0;
}
