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
    """Validation loss over the whole run, with the learning rate underneath.

    Each continuation restarts the cosine, which pushes the loss back above where the
    previous phase finished. The shaded wedges are those debts; both are measured from
    the data rather than written in, so the chart stays honest if the run is extended.
    """
    val = [(int(r["step"]), int(r["phase"]), int(r["tokens"]) / 1e9, float(r["val_loss"]))
           for r in read("val_loss.csv") if int(r["step"]) >= 1000]
    lr = [(int(r["tokens"]) / 1e9, float(r["lr"]), int(r["step"])) for r in read("train_metrics.csv")]
    lr = [q for q in lr if q[0] >= 0.5]
    phases = sorted({p[1] for p in val})

    W, L, R = 760, 44, 24
    T1, H1 = 34, 262
    T2, H2 = 312, 392
    H = 436
    last_tok = val[-1][2]
    lo = min(p[3] for p in val)
    hi = max(p[3] for p in val)
    X = Scale(0.5, last_tok * 1.03, L, W - R)
    Y = Scale(math.floor(lo * 20) / 20, math.ceil(hi * 20) / 20, H1, T1)
    Ylr = Scale(0, max(v for _, v, _ in lr) * 1.05, H2, T2)

    o = [f'<text class="axis-label" x="{L}" y="{T1 - 18}">validation loss</text>']
    t = math.floor(lo * 10) / 10
    while t <= hi + 0.05:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t:.1f}</text>')
        t = round(t + 0.1, 2)

    # one shaded wedge per restart: from the phase boundary until the loss is back under it
    notes = []
    for ph in phases[1:]:
        prev = [p for p in val if p[1] == ph - 1][-1]
        ref = prev[3]
        seg = [prev] + [p for p in val if p[1] == ph]
        # Recovery is the first evaluation back at or under the previous phase's final loss.
        # Reporting the measured eval step, not an interpolated crossing, keeps this figure
        # identical to the one quoted in the post and the written report.
        rec = next((q for q in seg[1:] if q[3] <= ref), None)
        if rec is None:
            continue
        debt = [(X(q[2]), Y(q[3])) for q in seg if q[0] < rec[0]] + [(X(rec[2]), Y(ref))]
        o.append(f'<path class="debt" d="{path(debt)} Z"/>')
        o.append(f'<line class="ref" x1="{n(X(prev[2]))}" x2="{n(X(rec[2]))}" y1="{n(Y(ref))}" y2="{n(Y(ref))}"/>')
        notes.append((prev, ref, rec[2], rec[0] - prev[0]))

    for prev, ref, rec_tok, cost in notes:
        cx = X((prev[2] + rec_tok) / 2)
        top = Y(ref) - 30
        o.append(f'<line class="leader" x1="{n(cx)}" y1="{n(Y(ref) - 4)}" x2="{n(cx)}" y2="{n(top + 4)}"/>')
        o.append(f'<text class="note-strong" x="{n(cx)}" y="{n(top)}" text-anchor="middle">{cost:.0f} steps to recover</text>')

    for ph in phases[1:]:
        tok = [p for p in val if p[1] == ph][0][2]
        x = X(tok - TOKENS_PER_STEP * 100 / 1e9)
        o.append(f'<line class="boundary" x1="{n(x)}" x2="{n(x)}" y1="{T1}" y2="{H2}"/>')
        o.append(f'<text class="note" x="{n(x + 6)}" y="{T1 + 12}">phase {ph} starts</text>')

    for ph in phases:
        seg = [p for p in val if p[1] == ph]
        if ph > phases[0]:
            seg = [[p for p in val if p[1] == ph - 1][-1]] + seg
        o.append(f'<path class="line" d="{path([(X(p[2]), Y(p[3])) for p in seg])}"/>')
    for s_, ph, tok, v in val:
        o.append(f'<circle class="hit" cx="{n(X(tok))}" cy="{n(Y(v))}" r="8">'
                 f'<title>Step {s_}, {tok:.2f}B tokens: validation loss {v:.4f}</title></circle>')

    ends = [[p for p in val if p[1] == ph][-1] for ph in phases]
    for i, p_ in enumerate(ends):
        dy, anchor = (18, "end") if i < len(ends) - 1 else (-10, "end")
        o.append(f'<circle class="dot" cx="{n(X(p_[2]))}" cy="{n(Y(p_[3]))}" r="4"/>')
        o.append(f'<text class="value" x="{n(X(p_[2]) - 8)}" y="{n(Y(p_[3]) + dy)}" text-anchor="{anchor}">{p_[3]:.3f}</text>')

    o.append(f'<text class="axis-label" x="{L}" y="{T2 - 10}">learning rate</text>')
    for t_, lab in ((0, "0"), (1e-4, "1e-4"), (2e-4, "2e-4"), (3e-4, "3e-4")):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Ylr(t_))}" y2="{n(Ylr(t_))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Ylr(t_) + 4)}" text-anchor="end">{lab}</text>')
    o.append(f'<path class="line line-thin" d="{path([(X(t_), Ylr(v)) for t_, v, _ in lr])}"/>')
    for t_, v, s_ in lr[::10]:
        o.append(f'<circle class="hit" cx="{n(X(t_))}" cy="{n(Ylr(v))}" r="6">'
                 f'<title>Step {s_}, {t_:.2f}B tokens: learning rate {v:.2e}</title></circle>')

    for t_ in (1, 2, 3, 4):
        o.append(f'<text class="tick" x="{n(X(t_))}" y="{H2 + 20}" text-anchor="middle">{t_}B</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">training tokens</text>')

    costs = ", ".join(f"{c:.0f}" for *_, c in notes)
    desc = (f"Two panels sharing the training-token axis. Top: validation loss falls from "
            f"{val[0][3]:.3f} to {ends[-1][3]:.3f} across {len(phases)} phases. Each restart pushes it back above "
            f"the previous phase's final loss; the shaded wedges show those debts, costing {costs} steps. "
            f"Bottom: the learning rate, decayed and restarted once per phase.")
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
         ("boolq", "BoolQ", 0.5), ("piqa", "PIQA", 0.5), ("openbookqa", "OpenBookQA", 0.25),
         ("hellaswag", "HellaSwag", 0.25), ("winogrande", "WinoGrande", 0.5),
         ("arc_challenge", "ARC-Challenge", 0.25))


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


# ---- 3b. Validation loss against tokens, both axes logarithmic -------------------
def loss_loglog():
    """A straight line here means the run is still in the power-law regime."""
    val = [(int(r["tokens"]) / 1e9, float(r["val_loss"])) for r in read("val_loss.csv")]
    W, H, L, R, T, B = 760, 300, 52, 26, 34, 44
    X = Scale(val[0][0] * 0.85, val[-1][0] * 1.15, L, W - R, log=True)
    Y = Scale(2.9, 6.4, H - B, T, log=True)
    o = [f'<text class="axis-label" x="{L}" y="{T - 18}">validation loss, log scale</text>']
    for t in (3, 3.5, 4, 4.5, 5, 5.5, 6):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t:.1f}</text>')
    for t, lab in ((0.05, "50M"), (0.1, "100M"), (0.5, "500M"), (1, "1B"), (2, "2B"), (4, "4B")):
        if val[0][0] * 0.85 <= t <= val[-1][0] * 1.15:
            o.append(f'<text class="tick" x="{n(X(t))}" y="{H - B + 20}" text-anchor="middle">{lab}</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">training tokens, log scale</text>')

    # least squares on the last decade, to show the curve has not bent away from it
    tail = val[-len(val) // 2:]
    sx = sum(math.log10(t) for t, _ in tail); sy = sum(math.log10(v) for _, v in tail)
    sxx = sum(math.log10(t) ** 2 for t, _ in tail); sxy = sum(math.log10(t) * math.log10(v) for t, v in tail)
    k = len(tail)
    slope = (k * sxy - sx * sy) / (k * sxx - sx * sx)
    inter = (sy - slope * sx) / k
    fit = [(tail[0][0], 10 ** (inter + slope * math.log10(tail[0][0]))),
           (val[-1][0] * 1.12, 10 ** (inter + slope * math.log10(val[-1][0] * 1.12)))]
    o.append(f'<path class="line line-gap line-2" d="{path([(X(t), Y(v)) for t, v in fit])}"/>')
    o.append(f'<text class="note" x="{n(X(fit[1][0]))}" y="{n(Y(fit[1][1]) + 18)}" text-anchor="end">slope {slope:.3f}</text>')

    o.append(f'<path class="line" d="{path([(X(t), Y(v)) for t, v in val])}"/>')
    for t, v in val:
        o.append(f'<circle class="hit" cx="{n(X(t))}" cy="{n(Y(v))}" r="7">'
                 f'<title>{t:.2f}B tokens: validation loss {v:.4f}</title></circle>')
    for t, v in (val[0], val[-1]):
        o.append(f'<circle class="dot" cx="{n(X(t))}" cy="{n(Y(v))}" r="4"/>')
    o.append(f'<text class="value" x="{n(X(val[-1][0]) - 10)}" y="{n(Y(val[-1][1]) - 10)}" text-anchor="end">{val[-1][1]:.3f}</text>')

    desc = (f"Validation loss against training tokens, both axes logarithmic. The points fall on a near-straight "
            f"line from {val[0][1]:.2f} at {val[0][0]:.2f}B tokens to {val[-1][1]:.3f} at {val[-1][0]:.2f}B, with a fitted "
            f"slope of {slope:.3f} over the second half. A straight line means the run is still in the power-law "
            "regime and has not begun to flatten.")
    (OUT / "llm-0.5b-loglog.svg").write_text(svg(W, H, "Validation loss against tokens", desc, o))


# ---- 3c. Throughput and model-FLOPs utilisation ---------------------------------
def throughput():
    rows = [(int(r["step"]), float(r["tokens_per_sec"]) / 1000, float(r["mfu"]) * 100) for r in read("train_metrics.csv")]
    W, H, L, R, T, B = 760, 260, 52, 52, 34, 44
    X = Scale(0, rows[-1][0], L, W - R)
    Y = Scale(0, 55, H - B, T)
    Y2 = Scale(0, 100, H - B, T)
    o = [f'<text class="axis-label" x="{L}" y="{T - 18}">thousand tokens per second</text>']
    for t in (0, 10, 20, 30, 40, 50):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t}</text>')
    for t in (0, 25, 50, 75, 100):
        o.append(f'<text class="tick" x="{W - R + 8}" y="{n(Y2(t) + 4)}">{t}%</text>')
    o.append(f'<text class="axis-label" x="{W - R + 8}" y="{T - 18}">MFU</text>')
    for t in (2000, 4000, 6000, 8000):
        o.append(f'<text class="tick" x="{n(X(t))}" y="{H - B + 20}" text-anchor="middle">{t}</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">step</text>')
    o.append(f'<path class="line line-thin line-2" d="{path([(X(s_), Y2(m)) for s_, _, m in rows])}"/>')
    o.append(f'<path class="line line-thin" d="{path([(X(s_), Y(v)) for s_, v, _ in rows])}"/>')
    for s_, v, m in rows[::8]:
        o.append(f'<circle class="hit" cx="{n(X(s_))}" cy="{n(Y(v))}" r="6">'
                 f'<title>Step {s_}: {v:.1f}k tokens per second, {m:.1f}% MFU</title></circle>')
    med = sorted(v for _, v, _ in rows)[len(rows) // 2]
    medm = sorted(m for _, _, m in rows)[len(rows) // 2]
    desc = (f"Training throughput across the run, median {med:.1f} thousand tokens per second at {medm:.0f}% "
            "model-FLOPs utilisation. The dips are benchmark evaluations sharing the GPU with training.")
    (OUT / "llm-0.5b-throughput.svg").write_text(svg(W, H, "Throughput", desc, o))


# ---- 4. Gradient norm per layer, over training ----------------------------------
def layer_heatmap(csv_name, out_name, title, what, lo, hi, desc, log=True):
    rows = [r for r in read(csv_name) if int(r["step"]) % 100 == 0]
    layers = [k for k in rows[0] if k.startswith("layer_")]
    vals = [[float(r[k]) for k in layers] for r in rows]
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
            f = (math.log(v / lo) / math.log(hi / lo)) if log else ((v - lo) / (hi - lo))
            b = min(bins - 1, max(0, int(f * bins)))
            x = L + ci * cw
            y = T + li * cell_h
            paths[b].append(f"M{n(x)},{n(y)}h{n(cw - 1)}v{cell_h - 1}h{n(-(cw - 1))}z")
    o = []
    for b, segs in paths.items():
        if segs:
            o.append(f'<path class="heat heat-{b}" d="{"".join(segs)}"/>')
    for li in (0, 6, 12, 18, 23):
        o.append(f'<text class="tick" x="{L - 8}" y="{n(T + li * cell_h + cell_h - 1.5)}" text-anchor="end">layer {li}</text>')
    for s in (100, 2000, 4000, 6000, 8000):
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
    scale_txt = "log scale" if log else "linear scale"
    o.append(f'<text class="tick" x="{lx + 52 + bins * 22 + 6}" y="{ly + 9}">larger {what} ({edges[0]:.2f} to {edges[-1]:.1f}, {scale_txt})</text>')

    (OUT / out_name).write_text(svg(W, H, title, desc, o))


def grad_heatmap():
    layer_heatmap(
        "grad_norm_by_layer.csv", "llm-0.5b-grad-heatmap.svg", "Gradient norm by layer",
        "gradient norm", 0.02, 0.5,
        "Heatmap of gradient norm for each of the 24 layers at every 100 steps. Layer 0 is consistently the darkest, "
        "with roughly four times the gradient of the early-middle layers, and the second half of the network runs darker "
        "than the first. All layers lighten together as training proceeds, darken slightly when the learning rate restarts "
        "at steps 2000 and 4000, and no layer drifts away from the rest.")


def resid_heatmap():
    rows = [r for r in read("resid_rms_by_layer.csv") if int(r["step"]) % 100 == 0]
    layers = [k for k in rows[0] if k.startswith("layer_")]
    vals = [float(r[k]) for r in rows for k in layers]
    layer_heatmap(
        "resid_rms_by_layer.csv", "llm-0.5b-resid-heatmap.svg", "Residual stream size by layer",
        "activation size", min(vals), max(vals),
        "Heatmap of the root-mean-square size of the residual stream leaving each of the 24 layers, every 100 steps. "
        "The picture is banded by depth rather than by time: each layer adds to a running sum, so the stream grows "
        "steadily from layer 0 to layer 23 and the bottom of the chart stays light while the top stays dark. The "
        "gradient is smooth, with no single layer breaking away from its neighbours.",
        log=False)


def diagnostics():
    """Two things the loss curve cannot tell you: how confident the model is, and how
    fast each tensor is still moving."""
    diag = [(int(r["step"]), float(r["pred_entropy"])) for r in read("diagnostics.csv") if r["pred_entropy"]]
    ur_rows = read("update_ratios.csv")
    cols = [c for c in ("tok_emb", "L0.attn.wo", "L12.attn.wo", "L23.attn.wo") if c in ur_rows[0]]

    W, L, R = 760, 52, 116
    T1, H1 = 34, 190
    T2, H2 = 246, 402
    H = 446
    last = diag[-1][0]
    X = Scale(0, last, L, W - R)
    ehi = max(v for _, v in diag)
    Y = Scale(2.5, math.ceil(ehi), H1, T1)

    o = [f'<text class="axis-label" x="{L}" y="{T1 - 18}">prediction entropy, nats</text>']
    t = 3.0
    while t <= ehi + 0.5:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t:.0f}</text>')
        t += 1
    o.append(f'<path class="line" d="{path([(X(s_), Y(v)) for s_, v in diag])}"/>')
    for s_, v in diag[::4]:
        o.append(f'<circle class="hit" cx="{n(X(s_))}" cy="{n(Y(v))}" r="6">'
                 f'<title>Step {s_}: entropy {v:.2f} nats</title></circle>')
    o.append(f'<text class="value" x="{n(X(last) - 8)}" y="{n(Y(diag[-1][1]) - 8)}" text-anchor="end">{diag[-1][1]:.2f}</text>')

    lo = min(float(r[c]) for r in ur_rows for c in cols if float(r[c]) > 0)
    hi = max(float(r[c]) for r in ur_rows for c in cols)
    Y2 = Scale(lo * 0.7, hi * 1.4, H2, T2, log=True)
    o.append(f'<text class="axis-label" x="{L}" y="{T2 - 18}">update size relative to weight size, log scale</text>')
    e = math.floor(math.log10(lo))
    while e <= math.ceil(math.log10(hi)):
        gv = 10.0 ** e
        if lo * 0.7 <= gv <= hi * 1.4:
            o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y2(gv))}" y2="{n(Y2(gv))}"/>')
            o.append(f'<text class="tick" x="{L - 8}" y="{n(Y2(gv) + 4)}" text-anchor="end">1e{e}</text>')
        e += 1
    for i, c in enumerate(cols):
        pts = [(int(r["step"]), float(r[c])) for r in ur_rows if float(r[c]) > 0]
        cls = "line" if i == 0 else f"line line-{min(i + 1, 2)}"
        if i > 1:
            cls += " line-gap"
        o.append(f'<path class="{cls} line-thin" d="{path([(X(s_), Y2(v)) for s_, v in pts])}"/>')
        o.append(f'<text class="note" x="{W - R + 6}" y="{n(Y2(pts[-1][1]) + 4)}">{c}</text>')
    for s_ in (2000, 4000, 6000, 8000):
        o.append(f'<text class="tick" x="{n(X(s_))}" y="{H2 + 20}" text-anchor="middle">{s_}</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">step</text>')
    for s_ in (2000, 4000):
        o.append(f'<line class="boundary" x1="{n(X(s_))}" x2="{n(X(s_))}" y1="{T1}" y2="{H2}"/>')

    desc = (f"Two panels sharing the step axis. Top: mean prediction entropy falls from {diag[0][1]:.2f} nats to "
            f"{diag[-1][1]:.2f} as the model becomes more certain of its next token. Bottom: the size of each "
            "optimizer update relative to the weight it changes, on a log scale, for the embedding table and the "
            "attention output projections of layers 0, 12 and 23. All of them shrink as the learning rate decays, "
            "and step back up at each restart.")
    (OUT / "llm-0.5b-diagnostics.svg").write_text(svg(W, H, "Entropy and update size", desc, o))


def grad_norm_total():
    """Global gradient norm before clipping, with the clipped steps marked."""
    rows = [(int(r["step"]), float(r["grad_norm"]), r["grad_clipped"] == "1") for r in read("train_metrics.csv")]
    W, H, L, R, T, B = 760, 250, 46, 24, 34, 44
    X = Scale(0, rows[-1][0], L, W - R)
    hi = max(v for _, v, _ in rows) * 1.08
    Y = Scale(0, hi, H - B, T)
    o = [f'<text class="axis-label" x="{L}" y="{T - 18}">global gradient norm, before clipping</text>']
    t = 0.0
    while t <= hi:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t:.0f}</text>')
        t += 1
    o.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{n(Y(1))}" y2="{n(Y(1))}"/>')
    o.append(f'<text class="note" x="{W - R - 4}" y="{n(Y(1) - 6)}" text-anchor="end">clipping threshold, 1.0</text>')
    for s_ in (2000, 4000):
        o.append(f'<line class="boundary" x1="{n(X(s_))}" x2="{n(X(s_))}" y1="{T}" y2="{H - B}"/>')
    o.append(f'<path class="line line-thin" d="{path([(X(s_), Y(v)) for s_, v, _ in rows])}"/>')
    clipped = [(s_, v) for s_, v, c in rows if c]
    for s_, v in clipped:
        o.append(f'<circle class="dot dot-2" cx="{n(X(s_))}" cy="{n(Y(v))}" r="3.5">'
                 f'<title>Step {s_}: gradient norm {v:.2f}, clipped to 1.0</title></circle>')
    for s_, v, _ in rows[::10]:
        o.append(f'<circle class="hit" cx="{n(X(s_))}" cy="{n(Y(v))}" r="6"><title>Step {s_}: gradient norm {v:.3f}</title></circle>')
    for s_ in (2000, 4000, 6000, 8000):
        o.append(f'<text class="tick" x="{n(X(s_))}" y="{H - B + 20}" text-anchor="middle">{s_}</text>')
    o.append(f'<text class="axis-label" x="{W - R}" y="{H - 6}" text-anchor="end">step</text>')
    last_clip = max(s_ for s_, _ in clipped)
    desc = (f"Global gradient norm over the run, measured before clipping. It spikes above the clipping "
            f"threshold of 1.0 {len(clipped)} times, all within the first {last_clip} steps, then settles "
            f"between 0.2 and 0.4 for the remaining {rows[-1][0] - last_clip} steps and never triggers clipping again.")
    (OUT / "llm-0.5b-gradnorm.svg").write_text(svg(W, H, "Gradient norm", desc, o))


def benchmark_bars():
    """Every model on one axis, ours beside the public baselines."""
    b = {r["label"]: r for r in read("benchmarks.csv")}
    acc = lambda k: float(b[k]["avg_acc"])
    bars = [("Pythia-410M", "1.07B", acc("pythia-410m@1.07B-tok"), False),
            ("Pythia-1B", "1.07B", acc("pythia-1b@1.07B-tok"), False),
            ("Pythia-410M", "2.1B", acc("pythia-410m@2.1B-tok"), False),
            ("Ours", "1.05B", acc("final@2000"), True),
            ("Ours", "2.10B", acc("final@4000"), True),
            ("Ours, EMA", "4.19B", acc("ema@8000"), True),
            ("Ours", "4.19B", acc("final@8000"), True),
            ("Pythia-410M", "300B", acc("pythia-410m@300B-tok"), False),
            ("Qwen2.5-0.5B", "18T", acc("Qwen2.5-0.5B@18T-tok"), False),
            ("SmolLM2-360M", "4T", acc("SmolLM2-360M@4T-tok"), False)]
    bars.sort(key=lambda r: r[2])
    W, L, R, T, B = 760, 46, 20, 34, 62
    H = 330
    n_ = len(bars)
    slot = (W - L - R) / n_
    bw = min(46, slot * 0.6)
    hi = 0.62
    Y = Scale(0, hi, H - B, T)
    o = [f'<text class="axis-label" x="{L}" y="{T - 18}">average zero-shot accuracy, 9 tasks</text>']
    for t in (0, 0.2, 0.4, 0.6):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{n(Y(t))}" y2="{n(Y(t))}"/>')
        o.append(f'<text class="tick" x="{L - 8}" y="{n(Y(t) + 4)}" text-anchor="end">{t:.1f}</text>')
    for i, (name, tok, v, ours) in enumerate(bars):
        cx = L + (i + 0.5) * slot
        y = Y(v)
        cls = "bar" if ours else "bar bar-2"
        o.append(f'<rect class="{cls}" x="{n(cx - bw / 2)}" y="{n(y)}" width="{n(bw)}" height="{n(H - B - y)}" rx="2">'
                 f'<title>{name}, {tok} tokens: {a3(v)}</title></rect>')
        o.append(f'<text class="value" x="{n(cx)}" y="{n(y - 7)}" text-anchor="middle">{a3(v)}</text>')
        o.append(f'<text class="tick" x="{n(cx)}" y="{H - B + 16}" text-anchor="middle">{name}</text>')
        o.append(f'<text class="note" x="{n(cx)}" y="{H - B + 30}" text-anchor="middle">{tok}</text>')
    desc = ("Average zero-shot accuracy for every model measured, sorted low to high. Our checkpoints are the "
            f"filled bars: {a3(acc('final@2000'))} at 1.05B tokens, {a3(acc('final@4000'))} at 2.10B and "
            f"{a3(acc('final@8000'))} at 4.19B, with the weight-averaged version just behind at {a3(acc('ema@8000'))}. "
            f"Pythia-410M scores {a3(acc('pythia-410m@2.1B-tok'))} at a matched 2.1B tokens and "
            f"{a3(acc('pythia-410m@300B-tok'))} fully trained; SmolLM2-360M leads at {a3(acc('SmolLM2-360M@4T-tok'))}.")
    (OUT / "llm-0.5b-benchmark-bars.svg").write_text(svg(W, H, "Every model compared", desc, o))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    loss_and_lr()
    loss_loglog()
    throughput()
    accuracy_vs_tokens()
    per_task()
    grad_norm_total()
    benchmark_bars()
    grad_heatmap()
    resid_heatmap()
    diagnostics()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("llm-0.5b-*"))))
