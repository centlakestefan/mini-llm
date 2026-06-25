/* run.c — a tiny llama.cpp-style inference engine, written from scratch in C.
 *
 * Loads a TinyStories model (Llama architecture) and generates text on the CPU:
 * tokenize a prompt, run the transformer forward one token at a time, sample the
 * next token from the logits, decode, repeat. The 15M model writes a little story.
 *
 * Usage:  run [-v] [prompt] [temperature]
 *   -v            print a per-token trace (prefill/decode markers + confidence)
 *   prompt        text to continue (default: "Once upon a time")
 *   temperature   0 = greedy (default, deterministic); >0 samples, e.g. 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- architecture hyperparameters: the 7 ints at the start of the file ---- */
typedef struct {
    int dim;        /* transformer width: size of the activation vector x      */
    int hidden_dim; /* inner dimension of the feed-forward network             */
    int n_layers;   /* number of transformer blocks                            */
    int n_heads;    /* number of query heads                                   */
    int n_kv_heads; /* number of key/value heads (<= n_heads => grouped-query) */
    int vocab_size; /* number of distinct tokens                               */
    int seq_len;    /* maximum context length                                  */
} Config;

/* ---- pointers into the big weight buffer, one per tensor ----
 * Shapes are written as comments. Per-layer tensors are stored contiguously
 * for all layers, so e.g. wq holds n_layers blocks of (dim x dim) back to back. */
typedef struct {
    float *token_embedding_table; /* (vocab_size, dim)   token id -> vector     */
    float *rms_att_weight;        /* (n_layers, dim)     RMSNorm before attn    */
    float *wq;                    /* (n_layers, dim, n_heads*head_size)         */
    float *wk;                    /* (n_layers, dim, n_kv_heads*head_size)      */
    float *wv;                    /* (n_layers, dim, n_kv_heads*head_size)      */
    float *wo;                    /* (n_layers, n_heads*head_size, dim)         */
    float *rms_ffn_weight;        /* (n_layers, dim)     RMSNorm before FFN     */
    float *w1;                    /* (n_layers, hidden_dim, dim)  FFN gate      */
    float *w2;                    /* (n_layers, dim, hidden_dim)  FFN down      */
    float *w3;                    /* (n_layers, hidden_dim, dim)  FFN up        */
    float *rms_final_weight;      /* (dim,)              final RMSNorm          */
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
    float *q;   /* (dim,)        query for the current token                   */
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
    int head_size = c->dim / c->n_heads;
    long long n_layers = c->n_layers;

    w->token_embedding_table = p; p += (long long)c->vocab_size * c->dim;
    w->rms_att_weight = p;        p += n_layers * c->dim;
    w->wq = p;                    p += n_layers * c->dim * (c->n_heads * head_size);
    w->wk = p;                    p += n_layers * c->dim * (c->n_kv_heads * head_size);
    w->wv = p;                    p += n_layers * c->dim * (c->n_kv_heads * head_size);
    w->wo = p;                    p += n_layers * (c->n_heads * head_size) * c->dim;
    w->rms_ffn_weight = p;        p += n_layers * c->dim;
    w->w1 = p;                    p += n_layers * c->dim * c->hidden_dim;
    w->w2 = p;                    p += n_layers * c->hidden_dim * c->dim;
    w->w3 = p;                    p += n_layers * c->dim * c->hidden_dim;
    w->rms_final_weight = p;      p += c->dim;
    p += c->seq_len * head_size / 2; /* skip legacy freq_cis_real (RoPE) */
    p += c->seq_len * head_size / 2; /* skip legacy freq_cis_imag (RoPE) */
    w->wcls = shared_weights ? w->token_embedding_table : p;
}

/* Read the whole checkpoint into memory and wire up the weight pointers. */
static void read_checkpoint(const char *path, Transformer *t) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "could not open '%s'\n", path); exit(1); }

    /* file size */
    fseek(f, 0, SEEK_END);
    t->file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    /* slurp the entire file */
    t->data = malloc(t->file_size);
    if (!t->data) { fprintf(stderr, "out of memory\n"); exit(1); }
    if (fread(t->data, 1, t->file_size, f) != (size_t)t->file_size) {
        fprintf(stderr, "short read on checkpoint\n"); exit(1);
    }
    fclose(f);

    /* the header is the first 7 ints; copy it out */
    t->config = *(Config *)t->data;

    /* sign of vocab_size flags whether the classifier weights are shared */
    int shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    if (t->config.vocab_size < 0) t->config.vocab_size = -t->config.vocab_size;

    /* weights begin right after the 7-int header */
    float *weights_ptr = (float *)((char *)t->data + sizeof(Config));
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared_weights);

    /* --- verify our layout against the real file size --- */
    long long expected = (char *)weights_ptr - (char *)t->data; /* = 28 (header) */
    {
        Config *c = &t->config;
        int head_size = c->dim / c->n_heads;
        long long n = c->n_layers;
        long long floats =
            (long long)c->vocab_size * c->dim +          /* token_embedding   */
            n * c->dim +                                 /* rms_att           */
            n * c->dim * (c->n_heads * head_size) +      /* wq                */
            n * c->dim * (c->n_kv_heads * head_size) +   /* wk                */
            n * c->dim * (c->n_kv_heads * head_size) +   /* wv                */
            n * (c->n_heads * head_size) * c->dim +      /* wo                */
            n * c->dim +                                 /* rms_ffn           */
            n * c->dim * c->hidden_dim +                 /* w1                */
            n * c->hidden_dim * c->dim +                 /* w2                */
            n * c->dim * c->hidden_dim +                 /* w3                */
            c->dim +                                     /* rms_final         */
            (long long)c->seq_len * head_size / 2 +      /* freq_real (skip)  */
            (long long)c->seq_len * head_size / 2;       /* freq_imag (skip)  */
        expected += floats * (long long)sizeof(float);
    }
    printf("checkpoint layout check: expected %lld bytes, file is %lld bytes -> %s\n",
           expected, t->file_size, expected == t->file_size ? "MATCH" : "MISMATCH!");
    if (expected != t->file_size) {
        fprintf(stderr, "layout mismatch: our tensor map does not fit the file\n");
        exit(1);
    }
}

/* Allocate the scratch buffers. kv_dim is the width of one token's K (or V)
 * across all kv heads — smaller than dim when grouped-query attention is used. */
static void malloc_run_state(RunState *s, Config *c) {
    int kv_dim = (c->dim * c->n_kv_heads) / c->n_heads;
    s->x   = calloc(c->dim, sizeof(float));
    s->xb  = calloc(c->dim, sizeof(float));
    s->xb2 = calloc(c->dim, sizeof(float));
    s->hb  = calloc(c->hidden_dim, sizeof(float));
    s->hb2 = calloc(c->hidden_dim, sizeof(float));
    s->q   = calloc(c->dim, sizeof(float));
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
 * gain. The eps (1e-5) avoids division by zero. This is ~all the "normalization"
 * the model does. */
static void rmsnorm(float *o, float *x, float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = ss / size + 1e-5f;
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

/* The full forward pass for one token at position `pos`: embedding lookup, then
 * each layer (RMSNorm -> attention with RoPE + KV cache -> residual, then
 * RMSNorm -> SwiGLU feed-forward -> residual), and finally a RMSNorm + classifier
 * that leaves the next-token logits in s->logits. */
static void forward(Transformer *t, int token, int pos) {
    Config *c = &t->config;
    TransformerWeights *w = &t->weights;
    RunState *s = &t->state;
    int dim = c->dim;
    int head_size = dim / c->n_heads;
    int kv_dim = (c->dim * c->n_kv_heads) / c->n_heads;
    int kv_mul = c->n_heads / c->n_kv_heads; /* how many query heads share one kv head */

    /* embedding lookup: x <- row `token` of the embedding table */
    memcpy(s->x, w->token_embedding_table + (long long)token * dim, dim * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        /* --- attention RMSNorm --- */
        rmsnorm(s->xb, s->x, w->rms_att_weight + (long long)l * dim, dim);

        /* this layer's K/V slots in the cache for the current position */
        long long loff = (long long)l * c->seq_len * kv_dim;
        float *k = s->key_cache   + loff + (long long)pos * kv_dim;
        float *v = s->value_cache + loff + (long long)pos * kv_dim;

        /* --- Q, K, V projections --- */
        matmul(s->q, s->xb, w->wq + (long long)l * dim * dim,    dim, dim);
        matmul(k,    s->xb, w->wk + (long long)l * dim * kv_dim, dim, kv_dim);
        matmul(v,    s->xb, w->wv + (long long)l * dim * kv_dim, dim, kv_dim);

        /* --- RoPE: rotate pairs (i, i+1) of q and k by an angle ~ pos --- */
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;                 /* position within the head */
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float angle = pos * freq;
            float fcr = cosf(angle), fci = sinf(angle);
            int rotn = (i < kv_dim) ? 2 : 1;              /* rotate q (and k if in range) */
            for (int vv = 0; vv < rotn; vv++) {
                float *vec = (vv == 0) ? s->q : k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        /* --- multi-head self-attention --- */
        for (int h = 0; h < c->n_heads; h++) {
            float *q   = s->q   + h * head_size;          /* this head's query  */
            float *att = s->att + (long long)h * c->seq_len; /* this head's scores */

            /* score query against every key from position 0..pos */
            for (int t2 = 0; t2 <= pos; t2++) {
                float *kk = s->key_cache + loff + (long long)t2 * kv_dim
                          + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * kk[i];
                att[t2] = score / sqrtf((float)head_size);
            }
            /* turn scores into weights */
            softmax(att, pos + 1);

            /* weighted sum of the values -> this head's slice of xb */
            float *xb = s->xb + h * head_size;
            for (int i = 0; i < head_size; i++) xb[i] = 0.0f;
            for (int t2 = 0; t2 <= pos; t2++) {
                float *vv2 = s->value_cache + loff + (long long)t2 * kv_dim
                           + (h / kv_mul) * head_size;
                float a = att[t2];
                for (int i = 0; i < head_size; i++) xb[i] += a * vv2[i];
            }
        }

        /* --- output projection back into the residual stream --- */
        matmul(s->xb2, s->xb, w->wo + (long long)l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        /* --- FFN sub-layer: RMSNorm -> SwiGLU -> residual --- */
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + (long long)l * dim, dim);

        /* two parallel projections into the wider hidden space */
        int hidden_dim = c->hidden_dim;
        matmul(s->hb,  s->xb, w->w1 + (long long)l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + (long long)l * dim * hidden_dim, dim, hidden_dim);

        /* SwiGLU: hb <- silu(w1·x) * (w3·x), where silu(z) = z * sigmoid(z) */
        for (int i = 0; i < hidden_dim; i++) {
            float z = s->hb[i];
            z *= 1.0f / (1.0f + expf(-z));   /* SiLU activation */
            s->hb[i] = z * s->hb2[i];        /* gated by the w3 projection */
        }

        /* project back down to dim and add into the residual stream */
        matmul(s->xb, s->hb, w->w2 + (long long)l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    /* --- final RMSNorm, then classifier: x -> logits over the vocabulary --- */
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, c->vocab_size);
}

/* ================================ tokenizer ================================ */

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

/* find a string in the sorted vocab; return its id or -1 */
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
    /* single-byte fallback pieces: byte b -> the 2-char string {b, '\0'} */
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

/* id -> printable piece. prev_token is needed to strip the leading space that
 * SentencePiece inserts right after BOS. */
static char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;     /* strip space after BOS */
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)     /* a raw-byte token */
        piece = (char *)t->byte_pieces + byte_val * 2;
    return piece;
}

/* text -> token ids. bos/eos: whether to prepend token 1 / append token 2.
 * Writes into tokens[], sets *n_tokens. */
static void encode(Tokenizer *t, char *text, int bos, int eos, int *tokens, int *n_tokens) {
    /* lazily build the sorted vocab for bsearch */
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

    /* SentencePiece "dummy prefix": a leading space token, if text is non-empty */
    if (text[0] != '\0') {
        int sp = str_lookup(" ", t->sorted, t->vocab_size);
        tokens[(*n_tokens)++] = sp;
    }

    /* 1) break the text into the smallest known pieces (chars / UTF-8 runs),
     *    falling back to raw <0xXX> byte tokens (ids 3..258) when unknown */
    size_t len = 0;
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) len = 0;                /* new UTF-8 codepoint starts */
        buf[len++] = *c;
        buf[len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && len < 4) continue; /* keep gathering bytes */
        int id = str_lookup(buf, t->sorted, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < len; i++)
                tokens[(*n_tokens)++] = (unsigned char)buf[i] + 3; /* byte fallback */
        }
        len = 0;
    }

    /* 2) greedily merge the best adjacent pair, repeatedly */
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
        if (best_idx == -1) break;                       /* no more merges possible */
        tokens[best_idx] = best_id;                      /* merge the pair */
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

/* xorshift* — a tiny deterministic PRNG so a given seed reproduces a story */
static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1DULL) >> 32;
}
static float random_f32(unsigned long long *state) { /* uniform in [0,1) */
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_argmax(float *p, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (p[i] > p[best]) best = i;
    return best;
}

/* sample from a distribution given a coin in [0,1): walk the cumulative sum */
static int sample_mult(float *p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}

static int compare_probindex(const void *a, const void *b) {
    float pa = ((ProbIndex *)a)->prob, pb = ((ProbIndex *)b)->prob;
    return (pa < pb) - (pa > pb);                 /* descending */
}

/* nucleus sampling: keep the most-probable tokens whose mass reaches topp,
 * then sample among them. Cuts off the unreliable long tail. */
static int sample_topp(float *p, int n, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    float cutoff = (1.0f - topp) / (n - 1);        /* cheap pre-filter */
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

/* pick the next token id from the raw logits */
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
    /* encode the prompt (with BOS) into token ids */
    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    int n_prompt = 0;
    encode(tok, prompt, /*bos=*/1, /*eos=*/0, prompt_tokens, &n_prompt);
    if (n_prompt < 1) { fprintf(stderr, "empty prompt\n"); exit(1); }

    int V = t->config.vocab_size;

    /* -v: list the prompt tokens the model is about to ingest (the "prefill") */
    if (verbose) {
        printf("[encode] \"%s\" -> %d tokens\n", prompt, n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            int tk = prompt_tokens[i];
            const char *pc = (tk == 1) ? "<s>" : decode(tok, 0, tk);
            printf("  [prefill] pos=%2d  id=%5d  \"%s\"\n", i, tk, pc);
        }
        printf("  ----- prompt ingested (%d tokens); now generating -----\n", n_prompt);
    }

    char story[16384]; story[0] = '\0';   /* reassembled text, for the -v [output] line */

    clock_t start = 0;
    int token = prompt_tokens[0];   /* start with BOS */
    int pos = 0, generated = 0;

    while (pos < steps) {
        forward(t, token, pos);                       /* logits for next token */

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];            /* still feeding the prompt */
        } else {
            float prob = 0.0f;
            if (verbose) {
                /* raw (temperature=1) confidence the model has in whatever it picks */
                float *cp = malloc(V * sizeof(float));
                memcpy(cp, t->state.logits, V * sizeof(float));
                softmax(cp, V);
                next = sample(sampler, t->state.logits);
                prob = cp[next];
                free(cp);
            } else {
                next = sample(sampler, t->state.logits);  /* model's turn to choose */
            }
            generated++;
            if (start == 0) start = clock();          /* time generation only */
            if (verbose && next != 1) {
                const char *pc = decode(tok, token, next);
                printf("  [decode]  pos=%2d  id=%5d  \"%s\"   p=%.1f%%\n",
                       pos + 1, next, pc, prob * 100.0f);
            }
        }
        pos++;

        if (next == 1) break;                         /* BOS marks end-of-story */

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

int main(int argc, char **argv) {
    const char *checkpoint = "stories15M.bin";
    char *prompt = "Once upon a time";
    float temperature = 0.0f;   /* greedy by default: deterministic & reproducible */
    float topp = 0.9f;
    unsigned long long seed = 1234ULL;
    int verbose = 0;

    /* args: [-v] [prompt] [temperature]  (-v can appear anywhere) */
    int positional = 0;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "-v") == 0) verbose = 1;
        else if (positional == 0) { prompt = argv[a]; positional++; }
        else if (positional == 1) { temperature = (float)atof(argv[a]); positional++; }
    }

    Transformer t;
    read_checkpoint(checkpoint, &t);
    malloc_run_state(&t.state, &t.config);

    Config *c = &t.config;

    printf("\nConfig: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d (head_size=%d)\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads,
           c->vocab_size, c->seq_len, c->dim / c->n_heads);

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
