"""Graph features for anti-money laundering (AML) on bank transfer data.

Data: IBM "Transactions for Anti Money Laundering (AML)" (Kaggle), e.g. HI-Small_Trans.csv.
Each row is one transfer between two bank accounts, with a label "Is Laundering".

Idea:
  Money mules and laundering accounts are connected. So besides each account's own behaviour
  (how much, how often, which payment types), we look at its position in the transfer network:
  how many counterparties it has, how central it is, and how close it is to accounts that are
  already known to be involved in laundering.

Setup (time split, close to real life):
  - Cutoff T. Everything we use as input comes from transfers BEFORE T.
  - "Known" accounts = accounts involved in a laundering transfer before T (already found).
  - Target = accounts that are NOT known yet, and get involved in laundering AFTER T (new cases).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

RENAME = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_acct",
    "To Bank": "to_bank",
    "Account.1": "to_acct",
    "Amount Received": "amt_received",
    "Receiving Currency": "cur_received",
    "Amount Paid": "amt_paid",
    "Payment Currency": "cur_paid",
    "Payment Format": "fmt",
    "Is Laundering": "is_laundering",
}


# ---------------------------------------------------------------- loading
def load_transactions(path, nrows=None):
    """Read the Kaggle CSV and add: ts (datetime), src / dst (account ids), amt_usd."""
    df = pd.read_csv(path, nrows=nrows)
    missing = [c for c in RENAME if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected columns, missing {missing}. Found: {list(df.columns)}")
    df = df.rename(columns=RENAME)
    df["ts"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    # the same account number can exist in different banks, so the id is bank + account
    df["src"] = df["from_bank"].astype(str) + "_" + df["from_acct"].astype(str)
    df["dst"] = df["to_bank"].astype(str) + "_" + df["to_acct"].astype(str)
    rates = usd_rates(df)
    df["amt_usd"] = df["amt_paid"] * df["cur_paid"].map(rates)
    return df.drop(columns=["timestamp"])


def usd_rates(df):
    """USD per 1 unit of each currency, estimated from cross-currency transfers.

    For a transfer paid in currency P and received in currency R, both sides have the same value:
        amt_paid * usd(P) = amt_received * usd(R)  ->  log usd(P) - log usd(R) = log(amt_received / amt_paid)
    We take the median of this for each currency pair, solve for all currencies at once (least squares),
    with usd("US Dollar") = 1. Currencies never exchanged with the US Dollar group stay NaN."""
    m = (df["cur_paid"] != df["cur_received"]) & (df["amt_paid"] > 0) & (df["amt_received"] > 0)
    x = df.loc[m]
    if len(x) == 0:
        return pd.Series({"US Dollar": 1.0})
    logr = np.log(x["amt_received"] / x["amt_paid"])
    pairs = logr.groupby([x["cur_paid"], x["cur_received"]]).median()
    curs = sorted(set(df["cur_paid"]) | set(df["cur_received"]))
    idx = {c: k for k, c in enumerate(curs)}
    rows, b = [], []
    for (cp, cr), v in pairs.items():
        r = np.zeros(len(curs)); r[idx[cp]] += 1; r[idx[cr]] -= 1
        rows.append(r); b.append(v)
    if "US Dollar" in idx:                       # anchor: log usd(US Dollar) = 0
        r = np.zeros(len(curs)); r[idx["US Dollar"]] = 1
        rows.append(r * 100); b.append(0.0)
    M, b = np.array(rows), np.array(b)
    sol = np.linalg.lstsq(M, b, rcond=None)[0]

    # keep only currencies connected to the US Dollar through some chain of pairs
    adj = {c: set() for c in curs}
    for (cp, cr) in pairs.index:
        adj[cp].add(cr); adj[cr].add(cp)
    seen, stack = set(), ["US Dollar"] if "US Dollar" in idx else []
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c); stack.extend(adj[c] - seen)
    rates = {c: float(np.exp(sol[idx[c]])) if c in seen else np.nan for c in curs}
    rates["US Dollar"] = 1.0
    return pd.Series(rates)


# ---------------------------------------------------------------- time split and labels
def time_split(df, cutoff_quantile=0.6):
    """Split at a time cutoff T (a quantile of transfer times).

    Returns (before, T, accounts, known, y):
      before   : transfers before T (the only input for features)
      accounts : accounts active before T
      known    : boolean array, account already involved in laundering before T
      y        : 1 if the account is involved in laundering after T and was not known before
    """
    T = df["ts"].quantile(cutoff_quantile)
    before = df[df["ts"] < T]
    after = df[df["ts"] >= T]

    accounts = pd.Index(pd.unique(pd.concat([before["src"], before["dst"]])))
    lb = before[before["is_laundering"] == 1]
    la = after[after["is_laundering"] == 1]
    known_set = set(lb["src"]) | set(lb["dst"])
    later_set = set(la["src"]) | set(la["dst"])

    known = accounts.isin(list(known_set))
    y = (accounts.isin(list(later_set)) & ~known).astype(int)
    return before, T, accounts, np.asarray(known), np.asarray(y)


# ---------------------------------------------------------------- account features (no graph)
def account_features(tx, accounts):
    """Each account's own behaviour: counts, amounts, payment types, currencies, activity."""
    out = tx.groupby("src").agg(
        n_sent=("amt_usd", "size"), sum_sent_usd=("amt_usd", "sum"),
        max_sent_usd=("amt_usd", "max"), median_sent_usd=("amt_usd", "median"),
    )
    inn = tx.groupby("dst").agg(
        n_recv=("amt_usd", "size"), sum_recv_usd=("amt_usd", "sum"),
        max_recv_usd=("amt_usd", "max"), median_recv_usd=("amt_usd", "median"),
    )
    f = pd.DataFrame(index=accounts).join(out).join(inn)
    for c in ["n_sent", "sum_sent_usd", "n_recv", "sum_recv_usd"]:
        f[c] = f[c].fillna(0)

    # pass-through: money comes in and goes out at a similar size (typical for mules)
    hi = np.maximum(f["sum_sent_usd"], f["sum_recv_usd"])
    lo = np.minimum(f["sum_sent_usd"], f["sum_recv_usd"])
    f["pass_through"] = np.where(hi > 0, lo / hi.where(hi > 0, 1), 0.0)

    # share of each payment format among sent transfers
    fmt = pd.crosstab(tx["src"], tx["fmt"], normalize="index")
    fmt.columns = [f"share_sent_{str(c).lower().replace(' ', '_')}" for c in fmt.columns]
    f = f.join(fmt)

    f["n_currencies_sent"] = tx.groupby("src")["cur_paid"].nunique()
    f["share_cross_currency"] = (tx["cur_paid"] != tx["cur_received"]).groupby(tx["src"]).mean()

    # activity: number of active days, as sender or receiver
    days = pd.concat([
        pd.DataFrame({"acct": tx["src"], "day": tx["ts"].dt.floor("D")}),
        pd.DataFrame({"acct": tx["dst"], "day": tx["ts"].dt.floor("D")}),
    ])
    f["active_days"] = days.drop_duplicates().groupby("acct").size()
    f["active_days"] = f["active_days"].fillna(0)
    return f


# ---------------------------------------------------------------- graph
def build_graph(tx, accounts):
    """Directed transfer graph between accounts (self-transfers dropped).
    Returns (C, A): C = number of transfers i -> j, A = USD amount i -> j (both sparse, n x n)."""
    n = len(accounts)
    s = accounts.get_indexer(tx["src"])
    d = accounts.get_indexer(tx["dst"])
    keep = (s >= 0) & (d >= 0) & (s != d)
    s, d = s[keep], d[keep]
    amt = tx["amt_usd"].to_numpy()[keep]
    amt = np.nan_to_num(amt, nan=0.0)
    C = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n)).tocsr()
    A = sp.coo_matrix((amt, (s, d)), shape=(n, n)).tocsr()
    C.sum_duplicates()
    A.sum_duplicates()
    return C, A


def undirected(C):
    """1 if i and j transferred money in either direction, else 0."""
    U = ((C + C.T) > 0).astype(np.float64)
    U.setdiag(0)
    U.eliminate_zeros()
    return U.tocsr()


def pagerank(C, alpha=0.85, tol=1e-9, max_iter=200):
    """PageRank on the directed graph, weighted by number of transfers.
    Scaled so the average account has 1."""
    n = C.shape[0]
    out = np.asarray(C.sum(axis=1)).ravel()
    inv = np.divide(1.0, out, out=np.zeros(n), where=out > 0)
    P = sp.diags(inv) @ C                       # row-stochastic
    PT = P.T.tocsr()
    dangling = out == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = alpha * (PT @ r) + (alpha * r[dangling].sum() + (1 - alpha)) / n
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r * n


def structure_features(C, accounts):
    """Graph features that do NOT use any labels."""
    U = undirected(C)
    deg = np.asarray(U.sum(axis=1)).ravel()
    B = (C > 0).astype(np.float64)
    f = pd.DataFrame(index=accounts)
    f["fan_out"] = np.asarray(B.sum(axis=1)).ravel()          # number of accounts it sent to
    f["fan_in"] = np.asarray(B.sum(axis=0)).ravel()           # number of accounts it received from
    f["n_counterparties"] = deg
    nbr_deg = U @ deg
    f["avg_counterparty_degree"] = np.divide(nbr_deg, deg, out=np.zeros_like(deg), where=deg > 0)
    f["pagerank"] = pagerank(C)
    return f


def known_neighbor_features(C, A, accounts, known, max_hops=6):
    """Graph features that use the accounts already known before T (the 'seeds').

    - n_known_counterparties : direct counterparties that are known
    - share_known_counterparties
    - n_counterparties_near_known : counterparties that themselves touch a known account
    - usd_from_known / usd_to_known : money received from / sent to known accounts
    - hops_to_known : steps to the nearest known account (max_hops + 1 = not connected)
    """
    U = undirected(C)
    seed = known.astype(np.float64)
    deg = np.asarray(U.sum(axis=1)).ravel()

    f = pd.DataFrame(index=accounts)
    k1 = U @ seed
    f["n_known_counterparties"] = k1
    f["share_known_counterparties"] = np.divide(k1, deg, out=np.zeros_like(deg), where=deg > 0)
    f["n_counterparties_near_known"] = U @ (k1 > 0).astype(np.float64)
    f["usd_from_known"] = A.T @ seed
    f["usd_to_known"] = A @ seed
    f["hops_to_known"] = hops_to_set(U, known, max_hops=max_hops)
    return f


def hops_to_set(U, mask, max_hops=6):
    """Shortest number of steps from each node to any node in mask (multi-source BFS).
    A known node itself gets 0. Not reachable within max_hops -> max_hops + 1."""
    n = U.shape[0]
    if mask.sum() == 0:
        return np.full(n, max_hops + 1, dtype=float)
    # add one extra node connected to every seed, then BFS from it; subtract the extra step
    seeds = np.flatnonzero(mask)
    extra = sp.coo_matrix((np.ones(len(seeds)), (np.full(len(seeds), n), seeds)), shape=(n + 1, n + 1))
    G = sp.bmat([[U, None], [None, sp.csr_matrix((1, 1))]]).tocsr() + extra + extra.T
    dist = shortest_path(G, directed=False, unweighted=True, indices=[n]).ravel()[:n] - 1
    dist[~np.isfinite(dist)] = max_hops + 1
    return np.minimum(dist, max_hops + 1)


# ---------------------------------------------------------------- evaluation
def recall_at_k(y, score, k=0.01):
    """Share of all positives found in the top k (fraction) of scores."""
    n_top = max(1, int(round(len(y) * k)))
    top = np.argsort(-score)[:n_top]
    return y[top].sum() / max(1, y.sum())


def precision_at_k(y, score, k=0.01):
    n_top = max(1, int(round(len(y) * k)))
    top = np.argsort(-score)[:n_top]
    return y[top].mean()


# ---------------------------------------------------------------- synthetic data (to test the code)
def make_synthetic(path, n_accounts=4000, n_tx=60_000, n_rings=40, seed=0):
    """Small fake file with the same columns as the Kaggle CSV. Only for checking that the code runs."""
    rng = np.random.default_rng(seed)
    banks = rng.integers(1, 50, n_accounts)
    accts = [f"{rng.integers(16**8):08X}" for _ in range(n_accounts)]
    start = pd.Timestamp("2022-09-01")
    fmts = ["ACH", "Cheque", "Credit Card", "Wire", "Cash", "Reinvestment"]
    curs = ["US Dollar", "Euro", "Yuan", "Rupee"]
    rate = {"US Dollar": 1.0, "Euro": 1.1, "Yuan": 0.14, "Rupee": 0.012}
    rows = []
    for _ in range(n_tx):
        i, j = rng.integers(n_accounts, size=2)
        cur = rng.choice(curs, p=[0.6, 0.2, 0.1, 0.1])
        amt = float(rng.lognormal(6, 1.5))
        cur_r = cur if rng.uniform() < 0.9 else rng.choice(curs)
        rows.append((start + pd.Timedelta(minutes=int(rng.integers(0, 14 * 24 * 60))), banks[i], accts[i],
                     banks[j], accts[j], amt, cur_r, amt, cur, rng.choice(fmts), 0))
    # laundering rings: money passes along a short chain of accounts over a few days
    used = []
    for r in range(n_rings):
        chain = rng.choice(n_accounts, size=rng.integers(3, 7), replace=False)
        if used and rng.uniform() < 0.5:          # rings often share accounts with earlier rings
            chain[0] = rng.choice(used)
        used.extend(chain.tolist())
        t0 = start + pd.Timedelta(hours=int(rng.integers(0, 12 * 24)))
        amt = float(rng.lognormal(9, 0.5))
        for step, (i, j) in enumerate(zip(chain[:-1], chain[1:])):
            amt_usd = amt * 0.97 ** step
            rows.append((t0 + pd.Timedelta(hours=10 * step), banks[i], accts[i], banks[j], accts[j],
                         amt_usd, "US Dollar", amt_usd, "US Dollar", "ACH", 1))
    df = pd.DataFrame(rows, columns=["Timestamp", "From Bank", "Account", "To Bank", "Account",
                                     "Amount Received", "Receiving Currency", "Amount Paid",
                                     "Payment Currency", "Payment Format", "Is Laundering"])
    # make currencies consistent with the amounts
    df["Amount Paid"] = df["Amount Paid"] / df["Payment Currency"].map(rate)
    df["Amount Received"] = df["Amount Received"] / df["Receiving Currency"].map(rate)
    df = df.sort_values("Timestamp")
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y/%m/%d %H:%M")
    df.to_csv(path, index=False)
    return path
