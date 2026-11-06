"""Create rba_sample.csv from the full RBA data set (Wiefling et al., doi:10.5281/zenodo.6782156).
Keeps every login of each user with a confirmed account takeover plus every login of a repeatable 2% sample of the other users.
Usage: python rba_make_sample.py rba-dataset.csv rba_sample.csv"""
import sys, pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "rba-dataset.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "rba_sample.csv"
cols = ["User ID", "Country", "Region", "City", "ASN", "OS Name and Version", "Browser Name and Version",
        "Device Type", "Login Timestamp", "Round-Trip Time [ms]", "Login Successful", "Is Attack IP", "Is Account Takeover"]

ato_users = set()                                    # pass 1: users with a confirmed takeover
for ch in pd.read_csv(CSV, usecols=["User ID", "Is Account Takeover"], chunksize=1_000_000):
    ato_users |= set(ch.loc[ch["Is Account Takeover"] == True, "User ID"])
print("users with takeover:", len(ato_users))

parts = []                                           # pass 2: takeover users + 2% of users (User ID % 50 == 0)
for ch in pd.read_csv(CSV, usecols=cols, chunksize=1_000_000):
    parts.append(ch[(ch["User ID"] % 50 == 0) | ch["User ID"].isin(ato_users)])
df = pd.concat(parts)
df.to_csv(OUT, index=False)
print(df.shape); print(df["Is Account Takeover"].value_counts()); print(df["Is Attack IP"].value_counts())
