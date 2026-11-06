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


# =====================================================================================================
# EXTENSION: (1) fusion weights learned on a validation block, (2) bootstrap 95% confidence intervals
# =====================================================================================================
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

def ecdf(ref, x):                       # score -> its percentile among the reference scores (0..1)
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)

cut2 = int(len(tr) * 0.75)              # time-ordered inner split of the TRAIN part: 75% fit, 25% validation
fit, val = tr.iloc[:cut2], tr.iloc[cut2:]
print(f"fit={len(fit):,} val={len(val):,} | val ATO={val.y_ato.sum()} val attackIP={val.y_ip.sum()}")
GRID = [(i/20, j/20, (20-i-j)/20) for i in range(21) for j in range(21-i)]   # weights on a 0.05 grid summing to 1
FIXED = np.array([.60, .25, .15])

rows2, chosen, keep = [], [], {}
for seed in SEEDS:
    sc2 = StandardScaler().fit(fit[hist])
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=seed, n_jobs=-1).fit(sc2.transform(fit[hist]))
    s_fit, s_val, s_te = (-iso.score_samples(sc2.transform(x[hist])) for x in (fit, val, te))
    C = {}
    for nm, x, s in (("fit", fit, s_fit), ("val", val, s_val), ("te", te, s_te)):
        C[nm] = np.c_[ecdf(s_fit, s), ecdf(fit.drift.values, x.drift.values), ecdf(fit.rule.values, x.rule.values)]
    for lab in ("y_ato", "y_ip"):
        if val[lab].sum() < 3 or te[lab].sum() < 1: continue
        best = max(GRID, key=lambda w: roc_auc_score(val[lab].values, C["val"] @ np.array(w)))
        chosen.append(dict(seed=seed, label=lab, w_IF=best[0], w_drift=best[1], w_rule=best[2]))
        lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(C["val"], val[lab].values)
        cands = {
            "IF + profile features (75% train)": (C["te"][:, 0], C["fit"][:, 0]),
            "Drift score only": (C["te"][:, 1], C["fit"][:, 1]),
            "Rules only": (C["te"][:, 2], C["fit"][:, 2]),
            "Fixed weights 0.60/0.25/0.15": (C["te"] @ FIXED, C["fit"] @ FIXED),
            "Learned weights (grid, validation ROC-AUC)": (C["te"] @ np.array(best), C["fit"] @ np.array(best)),
            "Learned weights (logistic regression)": (lr.decision_function(C["te"]), lr.decision_function(C["fit"])),
        }
        for nm, (s_t, s_f) in cands.items():
            m = metrics(te, s_t, s_f, nm, seed)
            rows2 += [x for x in m if x["label"] == lab]
            if seed == SEEDS[0]: keep[(lab, nm)] = s_t
    print("seed", seed, "done")

res2 = pd.DataFrame(rows2)
agg2 = res2.groupby(["label", "method"])[["roc_auc", "pr_auc", "recall_at_1pct_fpr", "f1_at_q97"]].agg(["mean", "std"]).round(3)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("\n=== Fusion comparison (models trained on 75% of the training block) ===")
print(agg2.to_string())
print("\n=== Chosen grid weights per seed ===")
print(pd.DataFrame(chosen).to_string(index=False))
res2.to_csv("rba_fusion_raw.csv", index=False); agg2.to_csv("rba_fusion_summary.csv")

# ---- stratified bootstrap 95% CI (seed-42 scores; positives and negatives resampled separately)
B = 500; rg = np.random.default_rng(0); ci = []
def auc_fast(sp, sn):
    r = rankdata(np.r_[sp, sn]); n1 = len(sp)
    return (r[:n1].sum() - n1*(n1+1)/2) / (n1 * len(sn))
for (lab, nm), s in keep.items():
    y = te[lab].values; sp, sn = s[y == 1], s[y == 0]
    a, rc = [], []
    for _ in range(B):
        bp = rg.choice(sp, len(sp)); bn = rg.choice(sn, len(sn))
        a.append(auc_fast(bp, bn)); rc.append(np.mean(bp >= np.quantile(bn, 0.99)))
    ci.append(dict(label=lab, method=nm, n_pos=len(sp),
                   roc_auc=round(auc_fast(sp, sn), 3), roc_lo=round(np.percentile(a, 2.5), 3), roc_hi=round(np.percentile(a, 97.5), 3),
                   recall1=round(np.mean(sp >= np.quantile(sn, 0.99)), 3), rec_lo=round(np.percentile(rc, 2.5), 3), rec_hi=round(np.percentile(rc, 97.5), 3)))
cidf = pd.DataFrame(ci); print("\n=== Bootstrap 95% CI (seed 42) ==="); print(cidf.to_string(index=False))
cidf.to_csv("rba_bootstrap_ci.csv", index=False)
print("\nsaved rba_fusion_raw.csv, rba_fusion_summary.csv, rba_bootstrap_ci.csv")
