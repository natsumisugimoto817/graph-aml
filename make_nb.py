"""Builds notebooks/01_graph_features_aml.ipynb (run: python make_nb.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Graph features for money laundering detection

## Problem
Money mules and laundering accounts rarely act alone: money moves **through chains of accounts**.
A model that only looks at each account's own behaviour can miss this.

## Approach
1. Build the **transfer network**: accounts are nodes, transfers are edges.
2. Compare three models on the same accounts:
   - **A. Account only**: each account's own behaviour (amounts, counts, payment types, pass-through).
   - **B. + Network structure**: number of counterparties, PageRank, ... (no labels used).
   - **C. + Closeness to known cases**: connections to accounts that were **already found** before.
3. **Time split**, close to real life: features use only transfers **before a cutoff T**.
   The target is accounts that are **not known yet** at T, and get involved in laundering **after T**.
   So the question is: *"Who should analysts look at next?"*
4. EBM to see **how** the network features change the risk.

## Data
[IBM Transactions for Anti Money Laundering (Kaggle)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml):
synthetic bank transfers with a laundering label. Use `HI-Small_Trans.csv` (about 5M transfers).
Put it in `data/raw/ibm_aml/`, or set the environment variable `AML_DATA_PATH`.

If the file is not found, the notebook makes a **small fake file** so the code still runs (results are meaningless).
""")

md("## 1. Setup")
code(r"""
import os
import sys
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append("../src")
import graph_utils as gu

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)
COLORS = {"A. account only": "#8a8985", "B. + network structure": "#eb6834", "C. + closeness to known": "#2a78d6"}
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True, "axes.titleweight": "bold",
})
""")

md(r"""
### Config
- `CUTOFF_QUANTILE`: the cutoff T is this quantile of transfer times. 0.6 = the first 60% of transfers are "the past".
- `TOP_K`: share of accounts analysts can review (0.01 = top 1%).
""")
code(r"""
DATA_PATH = os.environ.get("AML_DATA_PATH", "../data/raw/ibm_aml/HI-Small_Trans.csv")
NROWS = None              # e.g. 1_000_000 to try quickly on part of the file
CUTOFF_QUANTILE = 0.6
TEST_SIZE = 0.3
TOP_K = 0.01
EBM_SAMPLE = 200_000      # rows used for the EBM (EBM is slower than LightGBM). None = all
RANDOM_STATE = 42

LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
)

if not os.path.exists(DATA_PATH):
    print(f"{DATA_PATH} not found -> using a small synthetic file (results are NOT meaningful)")
    os.makedirs("../data/synthetic", exist_ok=True)
    DATA_PATH = gu.make_synthetic("../data/synthetic/synthetic_trans.csv")
""")

md("## 2. Load transfers")
code(r"""
t0 = time.time()
df = gu.load_transactions(DATA_PATH, nrows=NROWS)
print(f"{len(df):,} transfers   laundering: {df['is_laundering'].sum():,} ({df['is_laundering'].mean():.3%})   "
      f"load {time.time() - t0:.0f}s")
print("from", df["ts"].min(), "to", df["ts"].max())
print("currencies without a USD rate:", df.loc[df["amt_usd"].isna(), "cur_paid"].unique())

# transfers and laundering by day
daily = df.groupby(df["ts"].dt.date).agg(transfers=("is_laundering", "size"), laundering=("is_laundering", "sum"))
daily["laundering_rate"] = daily["laundering"] / daily["transfers"]
daily
""")

md(r"""
## 3. Time split
- **Known** = accounts involved in a laundering transfer before T (already found by investigators).
- **Target** = accounts not known at T that are involved in laundering after T.
- We score only accounts **active before T** and **not known yet**.
""")
code(r"""
before, T, accounts, known, y_all = gu.time_split(df, CUTOFF_QUANTILE)
print(f"cutoff T: {T}")
print(f"transfers before T: {len(before):,}")
print(f"accounts active before T: {len(accounts):,}")
print(f"known before T: {known.sum():,}")
print(f"new laundering accounts after T (target): {y_all.sum():,}  ({y_all[~known].mean():.3%} of not-known accounts)")
""")

md("## 4. Features")
code(r"""
t0 = time.time()
f_acct = gu.account_features(before, accounts)
C, A = gu.build_graph(before, accounts)
f_struct = gu.structure_features(C, accounts)
f_known = gu.known_neighbor_features(C, A, accounts, known)
print(f"graph: {C.shape[0]:,} accounts, {C.nnz:,} account pairs   features {time.time() - t0:.0f}s")

ACCT_COLS = list(f_acct.columns)
STRUCT_COLS = list(f_struct.columns)
KNOWN_COLS = list(f_known.columns)
FEATURE_SETS = {
    "A. account only": ACCT_COLS,
    "B. + network structure": ACCT_COLS + STRUCT_COLS,
    "C. + closeness to known": ACCT_COLS + STRUCT_COLS + KNOWN_COLS,
}

data = pd.concat([f_acct, f_struct, f_known], axis=1)
data["y"] = y_all
data = data[~known]                      # known accounts are not scored: they are already found
print(f"accounts to score: {len(data):,}   positives: {data['y'].sum():,}")
data[STRUCT_COLS + KNOWN_COLS].describe().T
""")

md(r"""
**How do new laundering accounts differ?** Average of a few features, positives vs others.
""")
code(r"""
show = ["n_sent", "n_recv", "pass_through", "fan_in", "fan_out", "pagerank",
        "n_known_counterparties", "n_counterparties_near_known", "hops_to_known"]
data.groupby("y")[show].mean().T.rename(columns={0: "others", 1: "new laundering"})
""")

md("## 5. Models: does the network help?")
code(r"""
train, test = train_test_split(data, test_size=TEST_SIZE, stratify=data["y"], random_state=RANDOM_STATE)
y_tr, y_te = train["y"].to_numpy(), test["y"].to_numpy()

scores, models, rows = {}, {}, []
t0 = time.time()
for name, cols in FEATURE_SETS.items():
    m = lgb.LGBMClassifier(**LGB_PARAMS).fit(train[cols], y_tr)
    p = m.predict_proba(test[cols])[:, 1]
    scores[name], models[name] = p, m
    rows.append({"model": name, "features": len(cols),
                 "AUC": roc_auc_score(y_te, p), "PR-AUC": average_precision_score(y_te, p),
                 f"recall @ top {TOP_K:.0%}": gu.recall_at_k(y_te, p, TOP_K),
                 f"precision @ top {TOP_K:.0%}": gu.precision_at_k(y_te, p, TOP_K)})
print(f"3 models: {time.time() - t0:.0f}s   test positives: {y_te.sum():,} of {len(y_te):,}")
results = pd.DataFrame(rows).set_index("model")
results
""")
code(r"""
fig, ax = plt.subplots(figsize=(7, 4.3))
for name, p in scores.items():
    prec, rec, _ = precision_recall_curve(y_te, p)
    ax.plot(rec * 100, prec * 100, color=COLORS[name], lw=2, label=f"{name} (PR-AUC {average_precision_score(y_te, p):.3f})")
ax.axhline(y_te.mean() * 100, color="#a3a3a3", ls=":", lw=1)
ax.set(title="Precision-recall on new laundering accounts (test)", xlabel="Recall (%)", ylabel="Precision (%)", xlim=(0, 100))
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
## 6. Business view
Analysts can review the top 1% of accounts. How many **new** laundering accounts does each model put in front of them?
""")
code(r"""
n_review = max(1, int(round(len(y_te) * TOP_K)))
pd.DataFrame({
    name: {"reviewed": n_review,
           "laundering accounts found": int(y_te[np.argsort(-p)[:n_review]].sum()),
           "all laundering accounts in test": int(y_te.sum())}
    for name, p in scores.items()
}).T
""")

md(r"""
## 7. Which features matter? (LightGBM, model C)
Gain importance: how much each feature improves the trees. Network features are marked.
""")
code(r"""
m = models["C. + closeness to known"]
imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURE_SETS["C. + closeness to known"])
imp = (imp / imp.sum()).sort_values().tail(15)
colors = ["#2a78d6" if c in KNOWN_COLS else "#eb6834" if c in STRUCT_COLS else "#8a8985" for c in imp.index]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(imp.index, imp.values * 100, color=colors)
ax.set(title="Top 15 features (share of gain, %)", xlabel="%")
ax.text(0.98, 0.02, "blue = closeness to known, orange = network structure, grey = account only",
        transform=ax.transAxes, ha="right", fontsize=8, color="#555")
plt.tight_layout(); plt.show()
""")

md(r"""
## 8. EBM: how the network features change the risk
Same features as model C, as an EBM (no interactions, so each feature has one shape).
""")
code(r"""
cols = FEATURE_SETS["C. + closeness to known"]
tr = train
if EBM_SAMPLE is not None and len(train) > EBM_SAMPLE:
    pos = train[train["y"] == 1]
    neg = train[train["y"] == 0].sample(EBM_SAMPLE - len(pos), random_state=RANDOM_STATE)
    tr = pd.concat([pos, neg])
    print(f"EBM on {len(tr):,} rows (all {len(pos):,} positives + sampled negatives)")

t0 = time.time()
ebm = ExplainableBoostingClassifier(interactions=0, random_state=RANDOM_STATE, n_jobs=-1).fit(tr[cols], tr["y"])
p_ebm = ebm.predict_proba(test[cols])[:, 1]
print(f"EBM {time.time() - t0:.0f}s   AUC {roc_auc_score(y_te, p_ebm):.4f}   PR-AUC {average_precision_score(y_te, p_ebm):.4f}"
      f"   recall @ top {TOP_K:.0%}: {gu.recall_at_k(y_te, p_ebm, TOP_K):.3f}")
""")
code(r"""
def plot_ebm_shapes(ebm, features, X_ref, ncols=3, clip=(0.01, 0.99)):
    names = list(ebm.term_names_)
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(features):]:
        ax.set_visible(False)
    for f, ax in zip(features, axes):
        d = ebm.explain_global().data(names.index(f))
        x = np.asarray(d["names"], dtype=float)[:-1]
        sc = np.asarray(d["scores"], dtype=float)
        ax.step(x, sc, where="post", color="#2a78d6", lw=2)
        lo, hi = np.nanpercentile(X_ref[f].astype(float), [clip[0] * 100, clip[1] * 100])
        if hi > lo:
            ax.set_xlim(lo, hi)
        ax.axhline(0, color="#a3a3a3", lw=1)
        ax.set_title(f, fontsize=10)
    fig.suptitle("EBM shapes (log-odds, above 0 = riskier)", fontweight="bold")
    plt.tight_layout(); plt.show()

imp_ebm = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)
print(imp_ebm.head(10).round(4).to_string())
network_top = [c for c in imp_ebm.index if c in STRUCT_COLS + KNOWN_COLS][:6]
plot_ebm_shapes(ebm, network_top, tr)
""")

md(r"""
**How to read**
- `hops_to_known`: 1 = direct counterparty of a known case, 2 = two steps away, 7 = not connected within 6 steps.
  If the risk is high at 1–2 and drops after, being close to known cases matters.
- `pass_through`: close to 1 = money in ≈ money out (typical for mules).
- `fan_in` / `fan_out`: many different counterparties can mean collecting or spreading money.
""")

md(r"""
## 9. One account in the network
A new laundering account from the test set that the model ranked high, with its direct counterparties
and theirs (2 steps). Known accounts (found before T) are red.
""")
code(r"""
def plot_neighborhood(acct_id, C, accounts, known, max_nodes=60, seed=RANDOM_STATE):
    U = gu.undirected(C)
    i = accounts.get_loc(acct_id)
    hop1 = U[i].indices
    hop2 = np.setdiff1d(np.unique(U[hop1].indices), np.r_[hop1, i]) if len(hop1) else np.array([], int)
    rng = np.random.default_rng(seed)
    keep2 = hop2[known[hop2]]                                  # always show known accounts at 2 steps
    rest = np.setdiff1d(hop2, keep2)
    room = max(0, max_nodes - 1 - len(hop1) - len(keep2))
    nodes = np.r_[i, hop1, keep2, rng.choice(rest, size=min(room, len(rest)), replace=False)].astype(int)
    G = nx.Graph()
    sub = U[nodes][:, nodes].tocoo()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(zip(sub.row, sub.col))
    pos = nx.spring_layout(G, seed=seed)
    col = ["#d64541" if known[n] else "#bdbdbd" for n in nodes]
    col[0] = "#2a78d6"
    size = [420] + [140] * (len(nodes) - 1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cfcfcf", width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col, node_size=size, linewidths=0)
    ax.set_title(f"{acct_id}: {len(hop1)} counterparties, {int(known[hop1].sum())} known", fontsize=10)
    ax.text(0.01, 0.01, "blue = this account   red = known before T   grey = others", transform=ax.transAxes,
            fontsize=8, color="#555")
    ax.axis("off"); plt.tight_layout(); plt.show()

p_c = scores["C. + closeness to known"]
cand = test.assign(score=p_c)
cand = cand[(cand["y"] == 1) & (cand["n_counterparties"].between(2, 40))].sort_values("score", ascending=False)
if len(cand):
    plot_neighborhood(cand.index[0], C, accounts, known)
else:
    print("no suitable account to show")
""")

md(r"""
## 10. Notes
Write your own observations after running:
- How much did network structure (B) and closeness to known cases (C) add over account-only features (A)?
- How many more new laundering accounts are in the top 1%?
- Which network features matter most, and what do their EBM shapes say?

**Limitations**
- The data is **synthetic** (IBM's simulator). Real networks are messier, and links between banks are only partly visible to one bank.
- Only accounts **active before T** are scored. Brand-new accounts need other signals.
- "Known" cases are only the ones that were **found**. Some laundering accounts are never found (the PU problem):
  unlabeled accounts close to known cases are good candidates for review, and for higher PU weights.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/01_graph_features_aml.ipynb")
print("written notebooks/01_graph_features_aml.ipynb")
