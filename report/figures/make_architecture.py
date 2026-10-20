"""Draw the system architecture diagram used in the report (report/figures/architecture.png).

Usage:
    python report/figures/make_architecture.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).with_name("architecture.png")

fig, ax = plt.subplots(figsize=(14, 8.6), dpi=200)
ax.set_xlim(0, 140)
ax.set_ylim(0, 86)
ax.axis("off")

for y, h, title, colour in [
    (63, 22, "1  Data & experimentation (local)", "#EAF2FB"),
    (41, 20, "2  CI/CD  (GitHub Actions, on every push to main)", "#EEF7EE"),
    (20, 19, "3  Serving  (Docker / Kubernetes on Docker Desktop)", "#FDF3E7"),
    (1, 17, "4  Monitoring & logging", "#F4EEF8"),
]:
    ax.add_patch(
        FancyBboxPatch(
            (1, y), 138, h, boxstyle="round,pad=0,rounding_size=1.2", fc=colour, ec="#B8B8B8"
        )
    )
    ax.text(2.5, y + h - 2.2, title, fontsize=10.5, weight="bold", color="#333", va="center")


def box(x, y, w, h, title, sub="", ec="#4A6FA5"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=ec, lw=1.3
        )
    )
    title_y = y + h - 2.3 if sub else y + h / 2
    ax.text(x + w / 2, title_y, title, ha="center", va="center", fontsize=9.2, weight="bold")
    if sub:
        ax.text(
            x + w / 2,
            y + (h - 4) / 2 + 0.3,
            sub,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#444",
            linespacing=1.3,
        )
    return x, y, w, h


def anchor(b, side):
    x, y, w, h = b
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[
        side
    ]


def arrow(a, b, sa="r", sb="l", label="", style="-|>", ls="-", color="#555", rad=0.0):
    p, q = anchor(a, sa), anchor(b, sb)
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle=style,
            mutation_scale=11,
            color=color,
            lw=1.2,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.3
        ax.text(mx, my, label, ha="center", fontsize=7, color="#333", style="italic")


# 1 Data & experimentation
uci = box(4, 66, 17, 11, "UCI repository", "Cleveland heart\ndisease data")
dl = box(26, 66, 18, 11, "download_data.py", "saves to data/raw/")
cl = box(49, 66, 20, 11, "data_processing.py", "clean, binary target\n297 rows")
tr = box(74, 66, 22, 11, "train.py", "ColumnTransformer + LR / RF\nGridSearchCV, 5-fold CV")
ml = box(101, 72, 18, 9, "MLflow", "runs, plots, registry", ec="#0B6E99")
art = box(101, 64, 18, 7, "models/", "joblib + metadata.json", ec="#0B6E99")
nb = box(122, 66, 15, 11, "Notebooks", "EDA, training,\ninference")
for a, b in [(uci, dl), (dl, cl), (cl, tr)]:
    arrow(a, b)
arrow(tr, ml)
arrow(tr, art)

# 2 CI/CD
gh = box(4, 44, 17, 11, "git push", "GitHub repo (main)", ec="#2E7D32")
lint = box(27, 44, 17, 11, "Lint", "flake8 + black", ec="#2E7D32")
test = box(50, 44, 18, 11, "Unit tests", "pytest + coverage\nJUnit artifacts", ec="#2E7D32")
dck = box(76, 49, 24, 9, "Docker build & smoke test", "/health, /predict, 422", ec="#2E7D32")
trn = box(76, 43, 24, 5, "Train model + MLflow", ec="#2E7D32")
summ = box(107, 44, 18, 11, "Run summary", "metrics table,\nlogs, artifacts", ec="#2E7D32")
arrow(gh, lint)
arrow(lint, test, label="pass")
arrow(test, dck, rad=-0.15)
arrow(test, trn, rad=0.15)
arrow(dck, summ, rad=0.12)
arrow(trn, summ, rad=-0.12)

# 3 Serving
img = box(
    4, 22, 22, 12, "Docker image", "multi-stage: train in build\nslim FastAPI runtime\nnon-root, HEALTHCHECK",
    ec="#C26A00",
)
k8s = box(
    34, 22, 32, 12, "Kubernetes Deployment",
    "namespace heart-disease, 2 replicas\nrolling update, /health probes\nCPU / memory requests + limits",
    ec="#C26A00",
)
svc = box(74, 22, 24, 12, "LoadBalancer Service", "localhost:80 to pod :8000", ec="#C26A00")
cli = box(106, 22, 22, 12, "Clients", "POST /predict (JSON)\nSwagger UI at /docs", ec="#C26A00")
arrow(img, k8s)
arrow(k8s, svc, style="<|-|>")
arrow(svc, cli, style="<|-|>")

# 4 Monitoring & logging
logs = box(4, 3, 24, 10, "docker / kubectl logs", "one JSON line per request", ec="#7B1FA2")
api = box(36, 3, 28, 10, "FastAPI middleware", "request + prediction logging\nX-Request-ID, GET /metrics")
prom = box(72, 3, 24, 10, "Prometheus", "scrape every 5 s, 3 alert rules", ec="#7B1FA2")
graf = box(104, 3, 24, 10, "Grafana", "provisioned dashboard\nlatency, errors, risk mix", ec="#7B1FA2")
arrow(api, logs, sa="l", sb="r", label="stdout")
arrow(api, prom, label="scrape")
arrow(prom, graf, label="PromQL")
arrow(k8s, api, sa="b", sb="t", ls="--", color="#888")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}")
