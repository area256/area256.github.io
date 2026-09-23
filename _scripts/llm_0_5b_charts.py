"""Draw the inline SVG charts for the "Pretraining a 0.5B model" post.

Reads the CSV exports in llm-0.5b-model/report/csv and writes one SVG per chart
into _includes/charts/. Colors come from CSS classes in assets/css/base.css, so
the charts follow the site's light and dark themes.

    python3 _scripts/llm_0_5b_charts.py ../llm-0.5b-model/report/csv
"""
import csv
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

CSV_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "../llm-0.5b-model/report/csv")
OUT = Path(__file__).resolve().parent.parent / "_includes" / "charts"
TOKENS_PER_STEP = 524_288


def read(name):
    with open(CSV_DIR / name) as f:
        return list(csv.DictReader(f))


def n(v):
    return f"{v:.1f}"


def a3(v):
    """Accuracy to three places, rounding halves up so 0.2115 reads 0.212 as in the report."""
    return str(Decimal(str(v)).quantize(Decimal("0.001"), ROUND_HALF_UP))


class Scale:
    def __init__(self, d0, d1, r0, r1, log=False):
        self.d0, self.d1, self.r0, self.r1, self.log = d0, d1, r0, r1, log

    def __call__(self, v):
        f = (lambda x: math.log10(x)) if self.log else (lambda x: x)
        return self.r0 + (f(v) - f(self.d0)) / (f(self.d1) - f(self.d0)) * (self.r1 - self.r0)


def svg(w, h, title, desc, body, cls="chart"):
    slug = title.lower().replace(" ", "-")[:24]
    return "\n".join([
        f'<svg class="{cls}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="{slug}-t {slug}-d">',
        f'<title id="{slug}-t">{title}</title>',
        f'<desc id="{slug}-d">{desc}</desc>',
        *body,
        "</svg>",
    ]) + "\n"


def path(points):
    return " ".join(f'{"M" if i == 0 else "L"}{n(x)},{n(y)}' for i, (x, y) in enumerate(points))


# ---- 1. Validation loss with the learning rate underneath -------------------------
def loss_and_lr():
    val = [(int(r["step"]), int(r["phase"]), int(r["tokens"]) / 1e9, float(r["val_loss"]))
           for r in read("val_loss.csv") if int(r["step"]) >= 1000]
    lr = [(int(r["tokens"]) / 1e9, float(r["lr"]), int(r["step"])) for r in read("train_metrics.csv")]
    lr = [p for p in lr if p[0] >= 0.5]

    W, L, R = 760, 44, 24
    T1, H1 = 34, 262          # loss panel top and bottom
    T2, H2 = 312, 392         # learning-rate panel top and bottom
    H = 436
    X = Scale(0.5, 2.2, L, W - R)
    Y = Scale(3.1, 3.6, H1, T1)
    Ylr = Scale(0, 3e-4, H2, T2)

    p1_end = next(p for p in val if p[0] == 2000)
    ref = p1_end[3]
    seg2 = [p1_end] + [p for p in val if p[1] == 2]
    for a, b in zip(seg2[1:], seg2[2:]):
        if a[3] >= ref > b[3]:
            rec_step = a[0] + (a[3] - ref) / (a[3] - b[3]) * (b[0] - a[0])
            break
    rec_tok = rec_step * TOKENS_PER_STEP / 1e9
    cost = round(rec_step - 2000, -1)

    o = [f'<text class="axis-label" x="{L}" y="{T1 - 18}">validation loss</text>']
    for t in (3.1, 3.2, 3.3, 3.4, 3.5, 3.6):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t:.1f}</text>')

    debt = [(X(p[2]), Y(p[3])) for p in seg2 if p[0] < rec_step] + [(X(rec_tok), Y(ref))]
    o.append(f'<path class="debt" d="{path(debt)} Z"/>')
    o.append(f'<line class="ref" x1="{n(X(p1_end[2]))}" x2="{n(X(rec_tok) + 40)}" y1="{n(Y(ref))}" y2="{n(Y(ref))}"/>')
    cx = X((p1_end[2] + rec_tok) / 2)
    o.append(f'<line class="leader" x1="{n(cx + 30)}" y1="{n(Y(3.37))}" x2="{n(cx + 58)}" y2="{n(Y(3.43))}"/>')
    o.append(f'<text class="note-strong" x="{n(cx + 62)}" y="{n(Y(3.43) - 2)}">Restart cost: about {cost:.0f} steps</text>')
    o.append(f'<text class="note" x="{n(cx + 62)}" y="{n(Y(3.43) + 14)}">to get back to {ref:.3f}</text>')

    for tok, label in ((2000 * TOKENS_PER_STEP / 1e9, "phase 2 starts"), (4000 * TOKENS_PER_STEP / 1e9, "phase 3 starts")):
        x = X(tok)
        left = tok > 2
        o.append(f'<line class="boundary" x1="{n(x)}" x2="{n(x)}" y1="{T1}" y2="{H2}"/>')
        o.append(f'<text class="note" x="{n(x - 6 if left else x + 6)}" y="{T1 + 12}" text-anchor="{"end" if left else "start"}">{label}</text>')

    for ph in (1, 2, 3):
        seg = [p for p in val if p[1] == ph]
        if ph > 1:
            seg = [[p for p in val if p[1] == ph - 1][-1]] + seg
        o.append(f'<path class="line{" line-live" if ph == 3 else ""}" d="{path([(X(p[2]), Y(p[3])) for p in seg])}"/>')
    for s, ph, tok, v in val:
        o.append(f'<circle class="hit" cx="{n(X(tok))}" cy="{n(Y(v))}" r="8"><title>Step {s}, {tok:.2f}B tokens: validation loss {v:.4f}</title></circle>')
    for s, txt, dx, dy, anchor in ((2000, "3.323", -8, 18, "end"), (4000, "3.139", 8, 18, "start"), (4100, "3.183", 8, -10, "start")):
        p = next(q for q in val if q[0] == s)
        o.append(f'<circle class="dot" cx="{n(X(p[2]))}" cy="{n(Y(p[3]))}" r="4"/>')
        o.append(f'<text class="value" x="{n(X(p[2]) + dx)}" y="{n(Y(p[3]) + dy)}" text-anchor="{anchor}">{txt}</text>')

    # learning-rate panel: same x scale, its own y scale
    o.append(f'<text class="axis-label" x="{L}" y="{T2 - 10}">learning rate</text>')
    for t, lab in ((0, "0"), (1e-4, "1e-4"), (2e-4, "2e-4"), (3e-4, "3e-4")):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Ylr(t))}" y2="{n(Ylr(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Ylr(t) + 4)}" text-anchor="end">{lab}</text>')
    o.append(f'<path class="line line-thin" d="{path([(X(t), Ylr(v)) for t, v, _ in lr])}"/>')
    for t, v, s in lr[::5]:
        o.append(f'<circle class="hit" cx="{n(X(t))}" cy="{n(Ylr(v))}" r="6"><title>Step {s}, {t:.2f}B tokens: learning rate {v:.2e}</title></circle>')

    for t in (0.5, 1.0, 1.5, 2.0):
        o.append(f'<text class="tick" x="{n(X(t))}" y="{H2 + 20}" text-anchor="middle">{t:.1f}B</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">training tokens</text>')

    desc = (f"Two panels sharing the training-token axis. Top: validation loss falls to {ref:.3f} at the end of phase 1, "
            f"rises to 3.383 after the learning rate is raised again for phase 2, and takes about {cost:.0f} steps to get back; "
            "phase 2 ends at 3.139 and phase 3 restarts again to 3.183. Bottom: the learning rate decays to 3e-5 in phase 1, "
            "jumps back to 2e-4 at the start of phase 2, decays to 2e-5, and jumps to 1.5e-4 for phase 3.")
    (OUT / "llm-0.5b-val-loss.svg").write_text(svg(W, H, "Validation loss and learning rate", desc, o))


# ---- 2. Average accuracy against training tokens -------------------------------
def accuracy_vs_tokens():
    bench = {r["label"]: r for r in read("benchmarks.csv")}
    acc = lambda k: float(bench[k]["avg_acc"])
    ours = [(0.262e9, acc("step500")), (0.524e9, acc("step1000")), (0.786e9, acc("step1500")),
            (1.049e9, acc("final@2000")), (2.097e9, acc("final@4000")), (4.194e9, acc("final@8000"))]
    pythia = [(1.07e9, acc("pythia-410m@1.07B-tok")), (2.1e9, acc("pythia-410m@2.1B-tok")), (300e9, acc("pythia-410m@300B-tok"))]
    others = [("Pythia-1B", 1.07e9, acc("pythia-1b@1.07B-tok"), 8, 16, "start"),
              ("SmolLM2-360M", 4e12, acc("SmolLM2-360M@4T-tok"), -10, -6, "end"),
              ("Qwen2.5-0.5B", 18e12, acc("Qwen2.5-0.5B@18T-tok"), 0, 22, "middle")]

    W, H, L, R, T, B = 760, 330, 44, 30, 34, 44
    X = Scale(1e8, 3e13, L, W - R, log=True)
    Y = Scale(0.25, 0.60, H - B, T)
    o = [f'<text class="axis-label" x="{L}" y="{T - 18}">average zero-shot accuracy, 9 tasks</text>']
    for t in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t:.2f}</text>')
    for t, lab in ((1e8, "100M"), (1e9, "1B"), (1e10, "10B"), (1e11, "100B"), (1e12, "1T"), (1e13, "10T")):
        o.append(f'<text class="tick" x="{n(X(t))}" y="{H - B + 20}" text-anchor="middle">{lab}</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">training tokens, log scale</text>')

    # Pythia-410M: measured points only; the long gap between 2.1B and 300B is dotted
    o.append(f'<path class="line line-2" d="{path([(X(t), Y(v)) for t, v in pythia[:2]])}"/>')
    o.append(f'<path class="line line-2 line-gap" d="{path([(X(t), Y(v)) for t, v in pythia[1:]])}"/>')
    o.append(f'<path class="line" d="{path([(X(t), Y(v)) for t, v in ours])}"/>')
    for t, v in pythia:
        o.append(f'<circle class="dot dot-2" cx="{n(X(t))}" cy="{n(Y(v))}" r="4"><title>Pythia-410M, {t / 1e9:g}B tokens: {a3(v)}</title></circle>')
    for t, v in ours:
        o.append(f'<circle class="dot" cx="{n(X(t))}" cy="{n(Y(v))}" r="4"><title>Our model, {t / 1e9:.2f}B tokens: {a3(v)}</title></circle>')
    for name, t, v, dx, dy, anchor in others:
        o.append(f'<circle class="dot dot-other" cx="{n(X(t))}" cy="{n(Y(v))}" r="4"><title>{name}, {t / 1e9:g}B tokens: {a3(v)}</title></circle>')
        o.append(f'<text class="note" x="{n(X(t) + dx)}" y="{n(Y(v) + dy)}" text-anchor="{anchor}">{name}</text>')

    t, v = ours[-1]
    o.append(f'<text class="value" x="{n(X(t) + 10)}" y="{n(Y(v) - 8)}">Ours, 4.2B: {a3(v)}</text>')
    t, v = pythia[1]
    o.append(f'<text class="value value-2" x="{n(X(t) + 10)}" y="{n(Y(v) + 16)}">Pythia-410M, 2.1B: {a3(v)}</text>')
    t, v = pythia[2]
    o.append(f'<text class="value value-2" x="{n(X(t))}" y="{n(Y(v) - 12)}" text-anchor="middle">Pythia-410M, 300B: {a3(v)}</text>')

    desc = (f"Average accuracy against training tokens on a log scale. Our model climbs from {a3(ours[0][1])} at 262M tokens "
            f"to {a3(ours[-1][1])} at 4.2B. Pythia-410M scores {a3(acc('pythia-410m@2.1B-tok'))} at 2.1B tokens and "
            f"{a3(acc('pythia-410m@300B-tok'))} at 300B. Pythia-1B scores {a3(acc('pythia-1b@1.07B-tok'))} at 1.07B. "
            f"SmolLM2-360M scores {a3(acc('SmolLM2-360M@4T-tok'))} at 4T tokens and Qwen2.5-0.5B {a3(acc('Qwen2.5-0.5B@18T-tok'))} at 18T.")
    (OUT / "llm-0.5b-accuracy.svg").write_text(svg(W, H, "Accuracy against training tokens", desc, o))


# ---- 3. One small chart per benchmark ------------------------------------------
TASKS = (("lambada_openai", "LAMBADA", None), ("sciq", "SciQ", 0.25), ("arc_easy", "ARC-Easy", 0.25),
         ("piqa", "PIQA", 0.5), ("hellaswag", "HellaSwag", 0.25), ("arc_challenge", "ARC-Challenge", 0.25))


def per_task():
    bench = {r["label"]: r for r in read("benchmarks.csv")}
    ckpts = (("step500", 0.262), ("step1000", 0.524), ("step1500", 0.786), ("final@2000", 1.049),
             ("final@4000", 2.097), ("final@8000", 4.194))
    pythia = bench["pythia-410m@300B-tok"]
    W, H, L, R, T, B = 220, 150, 30, 10, 10, 24
    X = Scale(0, 4.3, L, W - R)
    Y = Scale(0, 0.8, H - B, T)
    items = []
    for key, name, chance in TASKS:
        pts = [(tok, float(bench[label][key])) for label, tok in ckpts]
        o = []
        for t in (0, 0.2, 0.4, 0.6, 0.8):
            o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
            o.append(f'<text class="tick tick-sm" x="{L - 5}" y="{n(Y(t) + 3.5)}" text-anchor="end">{t:.1f}</text>')
        for t in (1, 2, 3, 4):  # no "0B" label: it would collide with the y-axis "0.0"
            o.append(f'<text class="tick tick-sm" x="{n(X(t))}" y="{H - B + 14}" text-anchor="middle">{t}B</text>')
        if chance is not None:
            o.append(f'<line class="chance" x1="{L}" x2="{W - R}" y1="{n(Y(chance))}" y2="{n(Y(chance))}"><title>Chance: {chance:.2f}</title></line>')
        pv = float(pythia[key])
        o.append(f'<line class="target" x1="{L}" x2="{W - R}" y1="{n(Y(pv))}" y2="{n(Y(pv))}"><title>Pythia-410M after 300B tokens: {a3(pv)}</title></line>')
        o.append(f'<path class="line" d="{path([(X(t), Y(v)) for t, v in pts])}"/>')
        for t, v in pts:
            o.append(f'<circle class="dot dot-sm" cx="{n(X(t))}" cy="{n(Y(v))}" r="3"><title>{name}, {t:.2f}B tokens: {a3(v)}</title></circle>')
        chance_txt = f", against chance at {chance:.2f}" if chance is not None else ""
        desc = f"{name} accuracy rises from {a3(pts[0][1])} at 262M tokens to {a3(pts[-1][1])} at 4.2B{chance_txt}. Fully trained Pythia-410M scores {a3(pv)}."
        items.append(f'<li>\n<h3>{name} <span class="panel-value">{a3(pts[-1][1])}</span></h3>\n'
                     + svg(W, H, f"{name} accuracy", desc, o, cls="chart chart-small") + "</li>")
    # The final value sits in each panel's heading, clear of the reference lines.
    html = '<ul class="small-multiples">\n' + "\n".join(items) + "\n</ul>\n"
    (OUT / "llm-0.5b-tasks.html").write_text(html)


# ---- 4. Gradient norm per layer, over training ----------------------------------
def grad_heatmap():
    rows = [r for r in read("grad_norm_by_layer.csv") if int(r["step"]) % 100 == 0]
    layers = [k for k in rows[0] if k.startswith("layer_")]
    vals = [[float(r[k]) for k in layers] for r in rows]
    lo, hi = 0.02, 0.5
    bins = 7
    edges = [lo * (hi / lo) ** (i / bins) for i in range(bins + 1)]

    W, L, R, T = 760, 58, 24, 16
    cell_h = 9
    H_plot = cell_h * len(layers)
    B = 66
    H = T + H_plot + B
    cw = (W - L - R) / len(rows)
    paths = {i: [] for i in range(bins)}
    for ci, col in enumerate(vals):
        for li, v in enumerate(col):
            b = min(bins - 1, max(0, int(math.log(v / lo) / math.log(hi / lo) * bins)))
            x = L + ci * cw
            y = T + li * cell_h
            paths[b].append(f"M{n(x)},{n(y)}h{n(cw - 1)}v{cell_h - 1}h{n(-(cw - 1))}z")
    o = []
    for b, segs in paths.items():
        if segs:
            o.append(f'<path class="heat heat-{b}" d="{"".join(segs)}"/>')
    for li in (0, 6, 12, 18, 23):
        o.append(f'<text class="tick" x="{L - 8}" y="{n(T + li * cell_h + cell_h - 1.5)}" text-anchor="end">layer {li}</text>')
    for s in (100, 1000, 2000, 3000, 4000):
        ci = next(i for i, r in enumerate(rows) if int(r["step"]) == s)
        o.append(f'<text class="tick" x="{n(L + ci * cw + cw / 2)}" y="{T + H_plot + 16}" text-anchor="middle">{s}</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{T + H_plot + 32}" text-anchor="end">training step</text>')
    for s in (2000, 4000):
        ci = next(i for i, r in enumerate(rows) if int(r["step"]) == s)
        x = L + ci * cw - 0.5
        o.append(f'<line class="boundary" x1="{n(x)}" x2="{n(x)}" y1="{T - 4}" y2="{T + H_plot + 4}"/>')
    for ci, (r, col) in enumerate(zip(rows, vals)):
        mid = sorted(col)[len(col) // 2]
        o.append(f'<rect class="hit-col" x="{n(L + ci * cw)}" y="{T}" width="{n(cw)}" height="{H_plot}">'
                 f'<title>Step {r["step"]}: layer 0 {col[0]:.3f}, median layer {mid:.3f}, layer 23 {col[-1]:.3f}</title></rect>')
    # legend: the seven bins, low to high
    lx, ly = L, H - 22
    o.append(f'<text class="tick" x="{lx}" y="{ly + 9}">smaller</text>')
    for b in range(bins):
        o.append(f'<rect class="heat heat-{b}" x="{lx + 52 + b * 22}" y="{ly}" width="20" height="10"/>')
    o.append(f'<text class="tick" x="{lx + 52 + bins * 22 + 6}" y="{ly + 9}">larger gradient norm ({edges[0]:.2f} to {edges[-1]:.1f}, log scale)</text>')

    desc = ("Heatmap of gradient norm for each of the 24 layers at every 100 steps. Layer 0 is consistently the darkest, "
            "with roughly four times the gradient of the early-middle layers, and the second half of the network runs darker "
            "than the first. All layers lighten together through phase 1, darken slightly when the learning rate restarts at "
            "steps 2000 and 4000, and no layer drifts away from the rest.")
    (OUT / "llm-0.5b-grad-heatmap.svg").write_text(svg(W, H, "Gradient norm by layer", desc, o))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    loss_and_lr()
    accuracy_vs_tokens()
    per_task()
    grad_heatmap()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("llm-0.5b-*"))))
