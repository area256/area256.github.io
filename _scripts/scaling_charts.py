"""Draw the inline SVG charts for the scaling posts (training and serving).

Both charts are computed from our 0.5B model's shape, not measured:

    python3 _scripts/scaling_charts.py
"""
from llm_0_5b_charts import OUT, Scale, n, path, svg

# Our model: 24 layers, d_model 1280, 20 query heads, 4 key/value heads of size 64.
LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 24, 1280, 20, 4, 64
MICRO_BATCH = 4
PARAMS = 489.3e6
STATE_GB = PARAMS * 16 / 1e9          # fp32 weights + fp32 grads + two Adam moments
GPU_GB = 32


def gb(v):
    return f"{v:,.1f} GB" if v < 100 else f"{v:,.0f} GB"


# ---- Training: activation memory against sequence length -------------------------
def activation_memory():
    """Mixed-precision activation estimate from the Ultra-Scale Playbook:
    L·s·b·h·(34 + 5·a·s/h) bytes. The second term is the stored attention matrix,
    which FlashAttention-style kernels never write out."""
    seqs = [512, 1024, 2048, 4096, 8192]
    base = [LAYERS * s * MICRO_BATCH * HIDDEN * 34 / 1e9 for s in seqs]
    full = [LAYERS * s * MICRO_BATCH * HIDDEN * (34 + 5 * HEADS * s / HIDDEN) / 1e9 for s in seqs]
    room = GPU_GB - STATE_GB

    W, H, L, R, T, B = 760, 330, 70, 150, 34, 44
    X = Scale(512, 8192, L, W - R, log=True)
    Y = Scale(1, 1000, H - B, T, log=True)
    o = [f'<text class="axis-label" x="{L}" y="{T - 18}">activation memory for a micro-batch of 4 sequences, log scale</text>']
    for t in (1, 10, 100, 1000):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t:,} GB</text>')
    for s in seqs:
        o.append(f'<text class="tick" x="{n(X(s))}" y="{H - B + 20}" text-anchor="middle">{s:,}</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">sequence length, tokens</text>')

    o.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{n(Y(room))}" y2="{n(Y(room))}"/>')
    o.append(f'<text class="note" x="{L + 6}" y="{n(Y(room) - 6)}">{room:.0f} GB free</text>')

    o.append(f'<path class="line line-other" d="{path([(X(s), Y(v)) for s, v in zip(seqs, full)])}"/>')
    o.append(f'<path class="line" d="{path([(X(s), Y(v)) for s, v in zip(seqs, base)])}"/>')
    for s, v in zip(seqs, full):
        o.append(f'<circle class="dot dot-other" cx="{n(X(s))}" cy="{n(Y(v))}" r="4"><title>{s:,} tokens, attention matrix stored: {gb(v)}</title></circle>')
    for s, v in zip(seqs, base):
        o.append(f'<circle class="dot" cx="{n(X(s))}" cy="{n(Y(v))}" r="4"><title>{s:,} tokens, FlashAttention: {gb(v)}</title></circle>')
    o.append(f'<text class="value" x="{n(X(8192) + 8)}" y="{n(Y(full[-1]) + 4)}">attention matrix stored</text>')
    o.append(f'<text class="value" x="{n(X(8192) + 8)}" y="{n(Y(base[-1]) + 4)}">FlashAttention</text>')
    i = seqs.index(2048)
    o.append(f'<text class="note-strong" x="{n(X(2048))}" y="{n(Y(base[i]) + 20)}" text-anchor="middle">our run: {base[i]:.1f} GB</text>')
    o.append(f'<text class="note" x="{n(X(2048) - 8)}" y="{n(Y(full[i]) - 10)}" text-anchor="end">{full[i]:.0f} GB</text>')

    desc = (f"Estimated activation memory for our 24-layer model with 4 sequences per micro-batch. With FlashAttention it grows "
            f"linearly, from {base[0]:.1f} GB at 512 tokens to {base[-1]:.0f} GB at 8,192. Storing the attention matrix adds a "
            f"term that grows with the square of sequence length: {full[i]:.0f} GB at 2,048 tokens and {full[-1]:.0f} GB at 8,192. "
            f"After {STATE_GB:.1f} GB of weights, gradients and optimizer state, a 32 GB GPU has about {room:.0f} GB left.")
    (OUT / "scaling-activation-memory.svg").write_text(svg(W, H, "Activation memory against sequence length", desc, o))


# ---- Serving: KV cache per sequence against context length ------------------------
def kv_cache():
    ctx = [2048, 4096, 8192, 16384, 24576, 32768]
    variants = (("Multi-head, 20 KV heads", HEADS, "line-other", "dot-other"),
                ("Grouped-query, 4 KV heads (ours)", KV_HEADS, "", ""),
                ("Multi-query, 1 KV head", 1, "line-2", "dot-2"))
    per_token = lambda kv: 2 * LAYERS * kv * HEAD_DIM * 2  # K and V, bf16
    weights_gb = PARAMS * 2 / 1e9

    W, H, L, R, T, B = 760, 330, 56, 190, 34, 44
    X = Scale(0, 32768, L, W - R)
    Y = Scale(0, 4.5, H - B, T)
    o = [f'<text class="axis-label" x="{L}" y="{T - 18}">KV cache for one sequence, bf16</text>']
    for t in (0, 1, 2, 3, 4):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t} GB</text>')
    for c in (0, 8192, 16384, 24576, 32768):
        o.append(f'<text class="tick" x="{n(X(c))}" y="{H - B + 20}" text-anchor="middle">{c // 1024}K</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">context length, tokens</text>')

    o.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{n(Y(weights_gb))}" y2="{n(Y(weights_gb))}"/>')
    o.append(f'<text class="note" x="{n(W - R - 6)}" y="{n(Y(weights_gb) - 6)}" text-anchor="end">the model\'s own weights: {weights_gb:.2f} GB</text>')

    for name, kv, line_cls, dot_cls in variants:
        pts = [(0, 0.0)] + [(c, per_token(kv) * c / 1e9) for c in ctx]
        o.append(f'<path class="line {line_cls}" d="{path([(X(c), Y(v)) for c, v in pts])}"/>')
        for c, v in pts[1:]:
            o.append(f'<circle class="dot {dot_cls}" cx="{n(X(c))}" cy="{n(Y(v))}" r="4"><title>{name}, {c:,} tokens: {v:.2f} GB</title></circle>')
        c, v = pts[-1]
        o.append(f'<text class="value" x="{n(X(c) + 10)}" y="{n(Y(v) + 4)}">{name.split(",")[0]}: {v:.2f} GB</text>')

    desc = (f"KV cache for one sequence of our model in bf16. Per token it is {per_token(HEADS) // 1024} KB with 20 KV heads, "
            f"{per_token(KV_HEADS) // 1024} KB with the 4 our model uses, and {per_token(1) // 1024} KB with one. At 32K tokens that is "
            f"{per_token(HEADS) * 32768 / 1e9:.2f}, {per_token(KV_HEADS) * 32768 / 1e9:.2f} and {per_token(1) * 32768 / 1e9:.2f} GB, "
            f"against {weights_gb:.2f} GB for the weights.")
    (OUT / "scaling-kv-cache.svg").write_text(svg(W, H, "KV cache against context length", desc, o))


if __name__ == "__main__":
    activation_memory()
    kv_cache()
    print("wrote scaling-activation-memory.svg, scaling-kv-cache.svg")
