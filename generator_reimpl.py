#!/usr/bin/env python3
"""
Independent re-implementation of the AEGIS synthetic login-telemetry generator,
written from the specification in Section 3.2 of the manuscript.

  1,000 entities (90% user / 8% admin / 2% service), 12 countries, 5 device/OS profiles,
  ~97,000 normal events (Jan-Jun 2026) + 3,000 injected anomalies = 100,000 events (3.0%),
  five anomaly scenarios: Impossible Travel, Brute Force, Credential Stuffing,
  Device Spoofing, Lateral Movement.

Usage: python generator_reimpl.py --out access_logs_repro.csv --seed 42
"""
import argparse
import numpy as np, pandas as pd

CITIES = {"India": ["New Delhi", "Mumbai", "Bengaluru"], "USA": ["New York", "San Francisco", "Chicago"],
          "UK": ["London", "Manchester"], "Germany": ["Berlin", "Munich"], "Japan": ["Tokyo", "Osaka"],
          "Canada": ["Toronto", "Vancouver"], "Australia": ["Sydney", "Melbourne"], "France": ["Paris", "Lyon"],
          "Singapore": ["Singapore"], "UAE": ["Dubai", "Abu Dhabi"], "Russia": ["Moscow", "Saint Petersburg"],
          "Brazil": ["Sao Paulo", "Rio de Janeiro"]}
COUNTRIES = list(CITIES)
HOME_P = np.array([.28, .18, .10, .08, .07, .06, .05, .05, .04, .04, .03, .02]); HOME_P /= HOME_P.sum()
DEVICES = [("Desktop", "Linux"), ("Desktop", "Windows"), ("Laptop", "macOS"), ("Mobile", "Android"), ("Tablet", "iOS")]
DEV_P = [.10, .35, .25, .18, .12]
BAD_DEVICES = [("Mobile", "Windows"), ("Desktop", "iOS"), ("Tablet", "macOS")]
USER_RES = ["Dashboard", "Email", "File Share", "VPN", "CRM", "HR Portal"]
ADMIN_RES = USER_RES + ["Admin Panel", "Database"]
SVC_RES = ["Database", "File Share", "VPN"]
AUTH = ["Password", "MFA", "SSO", "Biometric"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MIX = {"Impossible Travel": .28, "Brute Force": .24, "Credential Stuffing": .14, "Device Spoofing": .16, "Lateral Movement": .18}


def make(seed=42, n_entities=1000, n_total=100_000, contamination=0.03):
    r = np.random.default_rng(seed)
    n_anom = int(round(n_total * contamination)); n_norm = n_total - n_anom
    ents = []
    for i in range(n_entities):
        t = r.choice(["user", "admin", "service"], p=[.90, .08, .02])
        hc = r.choice(COUNTRIES, p=HOME_P); city = r.choice(CITIES[hc])
        dev = DEVICES[r.choice(5, p=DEV_P)] if t != "service" else ("Desktop", "Linux")
        alt = DEVICES[r.choice(5, p=DEV_P)]
        if t == "service": mu, sd = float(r.choice([1, 2, 3])), 0.8
        elif r.random() < .10: mu, sd = float(r.choice([22.5, 23.5, 0.5, 1.5])), 1.3     # night-shift operator
        else: mu, sd = float(r.uniform(10, 14.5)), 1.8                                 # office hours ~09-17
        pool = {"user": USER_RES, "admin": ADMIN_RES, "service": SVC_RES}[t]
        res = list(r.choice(pool, size=min(len(pool), int(r.integers(3, 5))), replace=False))
        browser = r.choice(["Safari", "Edge", "Chrome", "Firefox"] if dev[1] in ("macOS", "iOS") else ["Chrome", "Firefox", "Edge"],
                           p=[.5, .1, .3, .1] if dev[1] in ("macOS", "iOS") else [.55, .25, .20])
        ents.append(dict(id=f"USR{i:04d}", type=t, home=hc, city=city, dev=dev, alt=alt, mu=mu, sd=sd, res=res,
                         browser=browser, auth=r.choice(AUTH, p=[.25, .40, .25, .10]),
                         ip=f"{r.integers(11, 220)}.{r.integers(0, 255)}.{r.integers(0, 255)}", dur=float(r.uniform(15, 70))))
    rows = []
    start = pd.Timestamp("2026-01-01"); span = 181 * 1440

    def ts_for(e, hour=None):
        while True:
            day = int(r.integers(0, 181)); d = start + pd.Timedelta(days=day)
            if e["type"] != "service" and d.dayofweek >= 5 and r.random() > .12: continue
            break
        h = int(round(r.normal(e["mu"], e["sd"]))) % 24 if hour is None else hour
        return d + pd.Timedelta(hours=h, minutes=int(r.integers(0, 60)))

    def row(e, ts, **kw):
        base = dict(entity_id=e["id"], entity_type=e["type"], timestamp=ts, country=e["home"], city=e["city"],
                    ip_address=f"{e['ip']}.{int(r.integers(1, 255))}", device=e["dev"][0], os=e["dev"][1], browser=e["browser"],
                    auth_method=e["auth"], resource_accessed=str(r.choice(e["res"])), login_status="Success",
                    session_duration=int(max(1, r.normal(e["dur"], 12))), label=0, attack_type=None)
        base.update(kw); base["day_of_week"] = DAYS[ts.dayofweek]; base["login_hour"] = ts.hour; return base

    # ---- normal events
    w = r.dirichlet(np.ones(n_entities) * 8); counts = r.multinomial(n_norm, w)
    for e, c in zip(ents, counts):
        for _ in range(c):
            ts = ts_for(e); kw = {}
            u = r.random()
            if u < .02: kw["city"] = str(r.choice(CITIES[e["home"]]))                        # same-country variation
            elif u < .025:                                                                   # legitimate travel, working hours
                c2 = str(r.choice([x for x in COUNTRIES if x != e["home"]])); kw.update(country=c2, city=str(r.choice(CITIES[c2])))
            if r.random() < .07: kw.update(device=e["alt"][0], os=e["alt"][1])               # occasional second device
            if r.random() < .03: kw["login_status"] = "Failed"                                # ordinary typos
            if r.random() < .04 and e["type"] != "service": kw["resource_accessed"] = str(r.choice(USER_RES))
            if r.random() < .015: kw["auth_method"] = str(r.choice(AUTH))
            rows.append(row(e, ts, **kw))

    # ---- anomalies (event counts follow MIX; bursts for brute force / stuffing)
    target = {k: int(round(n_anom * v)) for k, v in MIX.items()}; target["Lateral Movement"] += n_anom - sum(target.values())
    users = [e for e in ents if e["type"] == "user"]
    def foreign(e):
        c2 = str(r.choice([x for x in COUNTRIES if x != e["home"]])); return c2, str(r.choice(CITIES[c2]))
    for atk, n in target.items():
        made = 0
        while made < n:
            if atk == "Impossible Travel":
                e = ents[int(r.integers(n_entities))]; c2, ci = foreign(e)
                rows.append(row(e, ts_for(e, hour=int(r.integers(2, 5))), country=c2, city=ci, ip_address=f"{r.integers(11,220)}.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(1,255)}",
                                login_status="Success" if r.random() < .8 else "Failed", label=1, attack_type=atk)); made += 1
            elif atk == "Brute Force":
                e = ents[int(r.integers(n_entities))]; k = min(int(r.integers(5, 11)), n - made); t0 = ts_for(e, hour=int(r.integers(0, 24)))
                ip = f"{r.integers(11,220)}.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(1,255)}"; c2, ci = foreign(e) if r.random() < .5 else (e["home"], e["city"])
                for j in range(k):
                    rows.append(row(e, t0 + pd.Timedelta(minutes=int(j * r.integers(1, 3))), country=c2, city=ci, ip_address=ip, login_status="Failed",
                                    auth_method="Password", label=1, attack_type=atk)); made += 1
            elif atk == "Credential Stuffing":
                k = int(r.integers(3, 6)); chosen = [ents[int(i)] for i in r.choice(n_entities, k, replace=False)]
                ip = f"{r.integers(11,220)}.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(1,255)}"; t0 = start + pd.Timedelta(minutes=int(r.integers(0, span)))
                c2, ci = foreign(chosen[0])
                for e in chosen:
                    for j in range(int(r.integers(1, 4))):
                        if made >= n: break
                        rows.append(row(e, t0 + pd.Timedelta(minutes=int(r.integers(0, 15))), country=c2, city=ci, ip_address=ip, login_status="Failed",
                                        auth_method="Password", label=1, attack_type=atk)); made += 1
            elif atk == "Device Spoofing":
                e = ents[int(r.integers(n_entities))]; d, o = BAD_DEVICES[int(r.integers(3))]
                rows.append(row(e, ts_for(e), device=d, os=o, label=1, attack_type=atk)); made += 1
            else:  # Lateral Movement
                e = users[int(r.integers(len(users)))]
                rows.append(row(e, ts_for(e, hour=None if r.random() < .5 else int(r.integers(0, 6))), resource_accessed=str(r.choice(["Admin Panel", "Database"])),
                                label=1, attack_type=atk)); made += 1
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True); df["risk_score"] = 0
    df.insert(0, "event_id", range(len(df)))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="access_logs_repro.csv"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); d = make(a.seed); d.to_csv(a.out, index=False)
    print(len(d), d.label.mean(), d.entity_id.nunique()); print(d.attack_type.value_counts())
