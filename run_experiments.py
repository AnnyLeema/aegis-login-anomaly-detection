#!/usr/bin/env python3
"""
Experiment suite for the AEGIS manuscript.

Produces the numbers for the tables marked [TBD] in the paper:
  Table 5  Baseline comparison (IF vs LOF / OCSVM / Autoencoder / COPOD*, + supervised references)
  Table 6  Ablation (IF only, drift only, rules only, IF+drift, IF+drift+rules)
  Table 7  Leave-one-attack-out generalisation (supervised RF vs unsupervised IF)
  Table 8  Drift study (static profile vs EMA alpha in {0.05, 0.10, 0.20}, plus a poison-guard variant)

Usage
  python run_experiments.py --data dataset/access_logs.csv --out results --seeds 5
  python run_experiments.py --smoke --out results_smoke      # synthetic stand-in, ONLY to check the script runs

Rules for honest reporting
  * Models are fitted on the training split only; encoders/scaler/profiles use the training split only.
  * The unsupervised models never see labels. Labels are used only for scoring and for the supervised references.
  * Decision thresholds come from the training-score quantile (1 - contamination), never from the test set.
  * Fusion weights below are fixed a priori. If you tune them, tune on a validation split carved from train.
  * Never put --smoke numbers in the paper.
*Needs `pip install pyod`; skipped automatically if missing.
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score)

warnings.filterwarnings("ignore")
CAT = ["entity_type", "country", "city", "device", "os", "browser", "auth_method",
       "resource_accessed", "day_of_week"]
DROP = ["entity_id", "ip_address", "attack_type", "risk_score", "event_id", "session_id",
        "department", "timestamp", "label"]
CONTAM = 0.03
W = {"temporal": 0.25, "geographic": 0.30, "device": 0.20, "resource": 0.15, "failure": 0.10}  # as in drift.py
FUSE = {"if": 0.60, "drift": 0.25, "rule": 0.15}                                               # a-priori weights
IMPLAUSIBLE = {("Mobile", "Windows"), ("Desktop", "iOS"), ("Tablet", "macOS")}


# ----------------------------------------------------------------------------- data
def make_smoke_data(n=20000, seed=0):
    """Tiny stand-in generator so the script can be smoke-tested. NOT your dataset."""
    r = np.random.default_rng(seed)
    ents = [f"USR{i:04d}" for i in range(300)]
    home = {e: r.choice(["India", "USA", "UK", "Germany", "Japan"]) for e in ents}
    hr = {e: r.normal(13, 2.5) for e in ents}
    rows = []
    for _ in range(n):
        e = r.choice(ents); anom = r.random() < CONTAM
        a = r.choice(["Impossible Travel", "Brute Force", "Device Spoofing", "Lateral Movement", "Credential Stuffing"]) if anom else None
        dev, os_ = r.choice([("Desktop", "Windows"), ("Laptop", "macOS"), ("Mobile", "Android")])
        res = r.choice(["Dashboard", "Email", "File Share", "CRM"]); st = "Success"; c = home[e]; h = int(np.clip(r.normal(hr[e], 1.5), 0, 23))
        if a == "Impossible Travel": c = r.choice(["Russia", "Brazil"]); h = int(r.integers(1, 5))
        if a == "Brute Force" or a == "Credential Stuffing": st = "Failed"
        if a == "Device Spoofing": dev, os_ = "Mobile", "Windows"
        if a == "Lateral Movement": res = r.choice(["Admin Panel", "Database"])
        ts = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=int(r.integers(0, 181 * 1440)))
        rows.append(dict(entity_id=e, entity_type="user", timestamp=ts, country=c, city=c + "-city", device=dev, os=os_,
                         browser="Chrome", auth_method=r.choice(["Password", "MFA"]), resource_accessed=res,
                         session_duration=int(r.integers(5, 120)), login_status=st, day_of_week=ts.day_name(),
                         login_hour=h, label=int(anom), attack_type=a, ip_address="1.1.1.1", risk_score=0))
    return pd.DataFrame(rows)


def featurise(df):
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["month"] = ts.dt.month.fillna(1).astype(int)
    df["is_weekend"] = ts.dt.dayofweek.fillna(0).ge(5).astype(int)
    df["login_status"] = (df["login_status"] == "Success").astype(int)
    return df


class Prep:
    """Fit on train only; unseen categories map to 0 (same behaviour as predict.py)."""
    def fit(self, df):
        X = df.drop(columns=[c for c in DROP if c in df.columns])
        self.cols = X.columns.tolist(); self.enc = {}
        for c in CAT:
            self.enc[c] = LabelEncoder().fit(X[c].astype(str))
        self.sc = StandardScaler().fit(self._enc(X).values.astype(float)); return self

    def _enc(self, X):
        X = X.copy()
        for c in CAT:
            cl = set(self.enc[c].classes_)
            X[c] = X[c].astype(str).map(lambda v: self.enc[c].transform([v])[0] if v in cl else 0)
        return X[self.cols]

    def transform(self, df):
        X = df.drop(columns=[c for c in DROP if c in df.columns])
        return self.sc.transform(self._enc(X).values.astype(float))


# ----------------------------------------------------------------------------- profiles / drift / rules
def build_profiles(df):
    P = {}
    for e, g in df.groupby("entity_id"):
        P[e] = dict(mean_hour=g.login_hour.mean(), std_hour=g.login_hour.std() or 1.0,
                    countries=set(g.country), devices=set(zip(g.device, g.os)), resources=set(g.resource_accessed),
                    failure_rate=1 - g.login_status.mean())
    G = dict(mean_hour=df.login_hour.mean(), std_hour=df.login_hour.std() or 1.0, countries=set(df.country),
             devices=set(zip(df.device, df.os)), resources=set(df.resource_accessed), failure_rate=1 - df.login_status.mean())
    return P, G


def drift_row(p, ev):
    z = abs(ev["login_hour"] - p["mean_hour"]) / (p["std_hour"] or 1.0)
    failed = 0 if ev["login_status"] == 1 else 1
    s = {"temporal": min(z / 3.0, 1.0), "geographic": 0.0 if ev["country"] in p["countries"] else 1.0,
         "device": 0.0 if (ev["device"], ev["os"]) in p["devices"] else 1.0,
         "resource": 0.0 if ev["resource_accessed"] in p["resources"] else 1.0,
         "failure": abs(failed - p["failure_rate"])}
    return sum(s[k] * W[k] for k in W)


def drift_scores(df, P, G):
    return np.array([drift_row(P.get(e, G), ev) for e, ev in zip(df.entity_id, df.to_dict("records"))])


def rule_flags(df):
    d = df
    spoof = [(a, b) in IMPLAUSIBLE for a, b in zip(d.device, d.os)]
    lateral = (d.entity_type.str.lower() == "user") & d.resource_accessed.isin(["Admin Panel", "Database"])
    off = (d.login_hour < 6) | (d.login_hour > 22)
    return (np.array(spoof) | lateral.values | (off & (d.login_status == 0)).values).astype(float)


# ----------------------------------------------------------------------------- metrics
def metrics(y, score, thr):
    pred = (score >= thr).astype(int)
    return dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, score),
                pr_auc=average_precision_score(y, score))


def q(train_scores):
    return np.quantile(train_scores, 1 - CONTAM)


def agg(rows):
    out = {}
    for name, lst in rows.items():
        df = pd.DataFrame(lst)
        out[name] = {c: f"{df[c].mean():.3f} ± {df[c].std(ddof=0):.3f}" for c in df.columns}
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------- experiments
def run_seed(df, seed, ablation=True):
    tr, te = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    prep = Prep().fit(tr); Xtr, Xte = prep.transform(tr), prep.transform(te); yte = te["label"].values
    res, abl, loao = {}, {}, {}

    # --- unsupervised detectors (labels unused in fit)
    t0 = time.time()
    iforest = IsolationForest(n_estimators=200, contamination=CONTAM, random_state=seed, n_jobs=-1).fit(Xtr)
    s_tr, s_te = -iforest.score_samples(Xtr), -iforest.score_samples(Xte)
    res["Isolation Forest"] = metrics(yte, s_te, q(s_tr)); res["Isolation Forest"]["fit_s"] = time.time() - t0

    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=CONTAM, n_jobs=-1).fit(Xtr)
    a, b = -lof.score_samples(Xtr[:20000]), -lof.score_samples(Xte)
    res["LOF"] = metrics(yte, b, q(a)); res["LOF"]["fit_s"] = time.time() - t0

    t0 = time.time()
    rng = np.random.default_rng(seed); sub = rng.choice(len(Xtr), min(10000, len(Xtr)), replace=False)
    oc = OneClassSVM(nu=CONTAM, gamma="scale").fit(Xtr[sub])
    a, b = -oc.decision_function(Xtr[sub]), -oc.decision_function(Xte)
    res["One-Class SVM"] = metrics(yte, b, q(a)); res["One-Class SVM"]["fit_s"] = time.time() - t0

    t0 = time.time()
    ae = MLPRegressor(hidden_layer_sizes=(12, 6, 12), max_iter=60, random_state=seed).fit(Xtr, Xtr)
    err = lambda X: ((ae.predict(X) - X) ** 2).mean(1)
    res["Autoencoder (MLP)"] = metrics(yte, err(Xte), q(err(Xtr))); res["Autoencoder (MLP)"]["fit_s"] = time.time() - t0

    try:
        from pyod.models.copod import COPOD
        t0 = time.time(); cp = COPOD(contamination=CONTAM).fit(Xtr)
        res["COPOD"] = metrics(yte, cp.decision_function(Xte), q(cp.decision_function(Xtr))); res["COPOD"]["fit_s"] = time.time() - t0
    except Exception:
        pass

    # --- supervised references (use labels; upper-bound, not like-for-like)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1).fit(Xtr, tr["label"])
    p = rf.predict_proba(Xte)[:, 1]; res["Random Forest (supervised)"] = metrics(yte, p, 0.5)
    hg = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, tr["label"])
    p = hg.predict_proba(Xte)[:, 1]; res["Gradient Boosting (supervised)"] = metrics(yte, p, 0.5)

    # --- ablation: profiles/drift/rules built from TRAIN only
    if ablation:
        P, G = build_profiles(tr)
        d_tr, d_te = drift_scores(tr, P, G), drift_scores(te, P, G)
        r_tr, r_te = rule_flags(tr), rule_flags(te)
        lo, hi = s_tr.min(), s_tr.max(); nrm = lambda s: np.clip((s - lo) / (hi - lo + 1e-12), 0, 1)
        abl["IF only"] = metrics(yte, s_te, q(s_tr))
        abl["Drift only (static profile)"] = metrics(yte, d_te, q(d_tr))
        abl["Rules only"] = metrics(yte, r_te + 1e-6 * d_te, q(r_tr + 1e-6 * d_tr))
        f = lambda s, d, r, a=FUSE: a["if"] * nrm(s) + a["drift"] * d + a["rule"] * r
        abl["IF + drift"] = metrics(yte, nrm(s_te) * 0.7 + 0.3 * d_te, q(nrm(s_tr) * 0.7 + 0.3 * d_tr))
        abl["IF + drift + rules (full hybrid)"] = metrics(yte, f(s_te, d_te, r_te), q(f(s_tr, d_tr, r_tr)))

    # --- leave-one-attack-out: supervised must generalise to an attack type it never saw
    if "attack_type" in df.columns:
        for atk in sorted(tr.loc[tr.label == 1, "attack_type"].dropna().unique()):
            keep = ~((tr.label == 1) & (tr.attack_type == atk))
            rf2 = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1).fit(Xtr[keep.values], tr.label[keep])
            m = ((te.label == 1) & (te.attack_type == atk)).values
            if m.sum() == 0: continue
            loao[atk] = dict(n_test=int(m.sum()), RF_recall_unseen=float((rf2.predict_proba(Xte[m])[:, 1] >= 0.5).mean()),
                             IF_recall=float((s_te[m] >= q(s_tr)).mean()))
    return res, abl, loao


class EMAProfile:
    """Categorical weights + EMA of hour mean/std. alpha=None -> static (never updated)."""
    def __init__(self, p, alpha, tau=0.02):
        self.alpha, self.tau = alpha, tau
        self.mean, self.std = p["mean_hour"], p["std_hour"]
        n = max(len(p["countries"]), 1); self.c = {k: 1 / n for k in p["countries"]}
        n = max(len(p["devices"]), 1); self.d = {k: 1 / n for k in p["devices"]}
        n = max(len(p["resources"]), 1); self.r = {k: 1 / n for k in p["resources"]}
        self.fr = p["failure_rate"]

    def score(self, ev):
        z = abs(ev["login_hour"] - self.mean) / (self.std or 1.0); failed = 0 if ev["login_status"] == 1 else 1
        s = {"temporal": min(z / 3, 1), "geographic": float(self.c.get(ev["country"], 0) < self.tau),
             "device": float(self.d.get((ev["device"], ev["os"]), 0) < self.tau),
             "resource": float(self.r.get(ev["resource_accessed"], 0) < self.tau), "failure": abs(failed - self.fr)}
        return sum(s[k] * W[k] for k in W)

    def update(self, ev):
        if self.alpha is None: return
        a = self.alpha
        self.mean = (1 - a) * self.mean + a * ev["login_hour"]
        self.std = max(0.5, (1 - a) * self.std + a * abs(ev["login_hour"] - self.mean))
        for dct, key in ((self.c, ev["country"]), (self.d, (ev["device"], ev["os"])), (self.r, ev["resource_accessed"])):
            for k in list(dct): dct[k] *= (1 - a)
            dct[key] = dct.get(key, 0) + a
        self.fr = (1 - a) * self.fr + a * (0 if ev["login_status"] == 1 else 1)


def drift_study(df, seed, thr, n_ent=150, base=60, drift_len=60):
    """Two scenarios per entity.
       A  slow insider: from step `base`, a growing share of events hits an unseen resource at an increasing hour shift.
       B  benign relocation: from step `base`, ALL events come from a new country (legitimate move).
       Static profile flags B forever (false alarms) but also flags A; EMA adapts to B but may absorb A."""
    rng = np.random.default_rng(seed); P, G = build_profiles(df)
    ents = [e for e in P if (df.entity_id == e).sum() >= 20]; ents = list(rng.choice(ents, min(n_ent, len(ents)), replace=False))
    unseen_res = lambda p: next((r for r in ["Admin Panel", "Database", "HR Portal", "VPN"] if r not in p["resources"]), "Admin Panel")
    out = {}
    variants = [("static (no update)", None, False), ("EMA α=0.05", .05, False), ("EMA α=0.10", .10, False),
                ("EMA α=0.20", .20, False), ("EMA α=0.10 + poison guard", .10, True)]
    for name, alpha, guard in variants:
        a_det, a_first, b_fa = [], [], []
        for e in ents:
            g = df[df.entity_id == e]; recs = g.sample(base + drift_len, replace=True, random_state=int(rng.integers(1e6))).to_dict("records")
            for scen in ("A", "B"):
                prof = EMAProfile(P[e], alpha); first = None; flagged = []
                for t, ev in enumerate(recs):
                    ev = dict(ev)
                    if t >= base:
                        k = (t - base + 1) / drift_len
                        if scen == "A" and rng.random() < k:
                            ev["resource_accessed"] = unseen_res(P[e]); ev["login_hour"] = int((ev["login_hour"] + 6 * k) % 24)
                        if scen == "B":
                            ev["country"] = "Singapore"; ev["login_status"] = 1
                    s = prof.score(ev)
                    if t >= base:
                        flagged.append(s >= thr)
                        if scen == "A" and first is None and s >= thr: first = t - base
                    if (not guard) or s < thr: prof.update(ev)   # guard=True: flagged events never update the baseline
                if scen == "A": a_det.append(np.mean(flagged)); a_first.append(first if first is not None else drift_len)
                else: b_fa.append(np.mean(flagged))
        out[name] = dict(slow_attack_flag_rate=np.mean(a_det), steps_to_first_flag=np.mean(a_first), benign_move_false_alarm_rate=np.mean(b_fa))
    return pd.DataFrame(out).T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = make_smoke_data() if a.smoke else pd.read_csv(a.data)
    df = featurise(df); print(f"{len(df)} rows, anomaly rate {df.label.mean():.4f}")
    R, A, L = {}, {}, {}
    for s in range(a.seeds):
        r, ab, lo = run_seed(df, 42 + s)
        for k, v in r.items(): R.setdefault(k, []).append(v)
        for k, v in ab.items(): A.setdefault(k, []).append(v)
        for k, v in lo.items(): L.setdefault(k, []).append(v)
        print(f"seed {42 + s} done")
    agg(R).to_csv(f"{a.out}/table5_baselines.csv"); agg(A).to_csv(f"{a.out}/table6_ablation.csv")
    pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in L.items()}).T.round(3).to_csv(f"{a.out}/table7_leave_one_attack_out.csv")
    P, G = build_profiles(df); thr = np.quantile(drift_scores(df, P, G), 1 - CONTAM)
    ds = drift_study(df, 42, thr); ds.round(3).to_csv(f"{a.out}/table8_drift_study.csv")
    for f in ["table5_baselines", "table6_ablation", "table7_leave_one_attack_out", "table8_drift_study"]:
        print(f"\n== {f}\n", pd.read_csv(f"{a.out}/{f}.csv", index_col=0).to_string())
    json.dump(dict(rows=len(df), seeds=a.seeds, contamination=CONTAM, fusion=FUSE, drift_weights=W), open(f"{a.out}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
