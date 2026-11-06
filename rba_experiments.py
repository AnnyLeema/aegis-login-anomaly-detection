"""AEGIS on the RBA login dataset (Wiefling et al.). Run in Colab on rba_sample.csv."""
import sys, time, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score

CSV = next((a for a in sys.argv[1:] if a.endswith(".csv")), "rba_sample.csv")   # also works when pasted into a notebook cell
SEEDS = [42, 43, 44]
TRAIN_FRAC = 0.60
ALPHA = 0.10
W = dict(time=.25, geo=.30, device=.20, asn=.15, fail=.10)   # same weights as the paper (resource -> ASN)

# ---------------------------------------------------------------- load
df = pd.read_csv(CSV)
df["ts"] = pd.to_datetime(df["Login Timestamp"])
df = df.sort_values("ts").reset_index(drop=True)
df["hour"] = df.ts.dt.hour; df["dow"] = df.ts.dt.dayofweek
df["rtt"] = df["Round-Trip Time [ms]"].fillna(-1)
df["ok"] = df["Login Successful"].astype(int)
df["browser"] = df["Browser Name and Version"].astype(str).str.replace(r"[\d\.]+", "", regex=True).str.strip()
df["os"] = df["OS Name and Version"].astype(str).str.replace(r"[\d\.]+", "", regex=True).str.strip()
df["dev"] = df["Device Type"].astype(str) + "|" + df["os"]
df["y_ato"] = df["Is Account Takeover"].astype(int)
df["y_ip"] = df["Is Attack IP"].astype(int)
print(f"rows={len(df):,} users={df['User ID'].nunique():,} ATO={df.y_ato.sum()} attackIP={df.y_ip.sum()}")

# ---------------------------------------------------------------- per-user causal profile (EMA, alpha=0.10)
def run_profiles(d):
    prof = {}
    n = len(d)
    drift = np.zeros(n); new_c = np.zeros(n); new_a = np.zeros(n); new_d = np.zeros(n); nprev = np.zeros(n)
    uid = d["User ID"].values; hr = d.hour.values; ctry = d.Country.values
    asn = d.ASN.values; dv = d.dev.values; ok = d.ok.values
    for i in range(n):
        u = uid[i]
        p = prof.get(u)
        if p is None:
            p = prof[u] = dict(n=0, hist=np.full(24, 1/24), geo=defaultdict(float), dev=defaultdict(float),
                               asn=defaultdict(float), fail=0.0)
        if p["n"] >= 3:
            s_t = 1 - p["hist"][hr[i]] / p["hist"].max()
            s_g = 1 - p["geo"].get(ctry[i], 0.0)
            s_d = 1 - p["dev"].get(dv[i], 0.0)
            s_a = 1 - p["asn"].get(asn[i], 0.0)
            s_f = abs((1 - ok[i]) - p["fail"])
            drift[i] = W["time"]*s_t + W["geo"]*s_g + W["device"]*s_d + W["asn"]*s_a + W["fail"]*s_f
        new_c[i] = ctry[i] not in p["geo"]; new_a[i] = asn[i] not in p["asn"]; new_d[i] = dv[i] not in p["dev"]
        nprev[i] = p["n"]
        # EMA update after scoring
        h = p["hist"]; h *= (1 - ALPHA); h[hr[i]] += ALPHA
        for k, v in ((ctry[i], "geo"), (dv[i], "dev"), (asn[i], "asn")):
            dct = p[v]
            for kk in list(dct): dct[kk] *= (1 - ALPHA)
            dct[k] += ALPHA
        p["fail"] = ALPHA*(1 - ok[i]) + (1 - ALPHA)*p["fail"]; p["n"] += 1
    return drift, new_c, new_a, new_d, nprev

t0 = time.time()
df["drift"], df["new_country"], df["new_asn"], df["new_dev"], df["nprev"] = run_profiles(df)
print(f"profiles built in {time.time()-t0:.0f}s")

# rule layer (transparent, uses only history + the event)
df["rule"] = (0.5*df.new_country + 0.2*df.new_asn + 0.15*df.new_dev
              + 0.15*(((df.hour < 5) | (df.hour > 22)) & (df.ok == 0))) * (df.nprev >= 3)

# ---------------------------------------------------------------- features
for c in ["Country", "ASN", "Device Type", "os", "browser"]:
    df[c + "_enc"] = df[c].astype("category").cat.codes
base = ["hour", "dow", "rtt", "ok", "Country_enc", "ASN_enc", "Device Type_enc", "os_enc", "browser_enc"]
hist = base + ["new_country", "new_asn", "new_dev", "drift"]
cut = int(len(df) * TRAIN_FRAC)
tr, te = df.iloc[:cut], df.iloc[cut:]
print(f"train={len(tr):,} test={len(te):,} | test ATO={te.y_ato.sum()} test attackIP={te.y_ip.sum()}")

def norm(s, ref):                       # 0..1 using training-score bounds (like the paper's 0-100 risk)
    lo, hi = np.min(ref), np.max(ref); return np.clip((s - lo) / (hi - lo + 1e-12), 0, 1)

def metrics(y, s, ref_scores, name, seed):
    out = []
    for lab in ("y_ato", "y_ip"):
        yy = y[lab].values
        if yy.sum() == 0: continue
        fpr, tpr, _ = roc_curve(yy, s)
        thr = np.quantile(ref_scores, 0.97)
        out.append(dict(method=name, label=lab, seed=seed, roc_auc=roc_auc_score(yy, s),
                        pr_auc=average_precision_score(yy, s),
                        recall_at_1pct_fpr=float(np.interp(0.01, fpr, tpr)),
                        f1_at_q97=f1_score(yy, (s >= thr).astype(int), zero_division=0)))
    return out

rows = []
rng_sub = np.random.default_rng(0)
for seed in SEEDS:
    sub = tr.sample(min(20000, len(tr)), random_state=seed)
    # Isolation Forest on raw login features
    sc = StandardScaler().fit(tr[base]); Xtr, Xte = sc.transform(tr[base]), sc.transform(te[base])
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=seed, n_jobs=-1).fit(Xtr)
    s_tr, s_te = -iso.score_samples(Xtr), -iso.score_samples(Xte)
    rows += metrics(te, s_te, s_tr, "IF (raw features)", seed)
    # Isolation Forest + profile features
    sc2 = StandardScaler().fit(tr[hist]); iso2 = IsolationForest(n_estimators=200, contamination=0.03, random_state=seed, n_jobs=-1).fit(sc2.transform(tr[hist]))
    rows += metrics(te, -iso2.score_samples(sc2.transform(te[hist])), -iso2.score_samples(sc2.transform(tr[hist])), "IF + profile features", seed)
    # hybrid fusion 0.60 / 0.25 / 0.15 (fixed a priori, as in the paper)
    h_tr = .60*norm(s_tr, s_tr) + .25*tr.drift.values + .15*tr.rule.values
    h_te = .60*norm(s_te, s_tr) + .25*te.drift.values + .15*te.rule.values
    rows += metrics(te, h_te, h_tr, "AEGIS hybrid (0.60/0.25/0.15)", seed)
    # LOF and One-Class SVM on a training subsample
    Xs = sc.transform(sub[base])
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Xs)
    rows += metrics(te, -lof.score_samples(Xte), -lof.score_samples(Xs), "LOF", seed)
    oc = OneClassSVM(nu=0.03, gamma="scale").fit(Xs[:8000])
    rows += metrics(te, -oc.score_samples(Xte), -oc.score_samples(Xs[:8000]), "One-Class SVM", seed)
    # supervised reference (needs labels; uses profile features)
    for lab in ("y_ato", "y_ip"):
        if tr[lab].sum() >= 5 and te[lab].sum() >= 1:
            rf = RandomForestClassifier(200, class_weight="balanced_subsample", min_samples_leaf=3, n_jobs=-1, random_state=seed).fit(tr[hist], tr[lab])
            p_te, p_tr = rf.predict_proba(te[hist])[:, 1], rf.predict_proba(tr[hist])[:, 1]
            r = metrics(te, p_te, p_tr, "Random Forest (supervised, reference)", seed)
            rows += [x for x in r if x["label"] == lab]

# deterministic scores (no seed): drift only, rules only
for name, col in (("Drift score only", "drift"), ("Rules only", "rule")):
    rows += metrics(te, te[col].values, tr[col].values, name, 0)

res = pd.DataFrame(rows)
agg = (res.groupby(["label", "method"])[["roc_auc", "pr_auc", "recall_at_1pct_fpr", "f1_at_q97"]]
          .agg(["mean", "std"]).round(3))
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(agg.to_string())
res.to_csv("rba_results_raw.csv", index=False); agg.to_csv("rba_results_summary.csv")
print("saved rba_results_raw.csv and rba_results_summary.csv")
