import pandas as pd

TABLES_TO_KEEP = ["Teams", "Coaches", "Scoring", "Master", "TeamsPost", "AwardsCoaches"]

MASTER_COLUMNS = ["playerID", "coachID", "birthYear", "height", "weight",
                  "shootCatch", "pos", "firstNHL", "lastNHL"]


def _keep_nhl(df):
    if "lgID" in df.columns:
        df = df[df["lgID"] == "NHL"]
    return df


def _to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_missing(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df.dropna(subset=cols) if cols else df


def _fill_zero(df, cols):
    cols = [c for c in cols if c in df.columns]
    df[cols] = df[cols].fillna(0)
    return df


def _clean_teams(df):
    numeric = ["year", "G", "W", "L", "T", "OTL", "Pts", "GF", "GA", "PIM", "BenchMinor",
               "PPG", "PPC", "SHA", "PKG", "PKC", "SHF", "rank"]
    df = _to_numeric(df, numeric)
    return _drop_missing(df, ["year", "tmID", "G", "Pts"])


def _clean_coaches(df):
    df = df.drop(columns=["notes"], errors="ignore")
    df = _to_numeric(df, ["year", "stint", "g", "w", "l", "t", "postg", "postw", "postl", "postt"])
    df = _drop_missing(df, ["coachID", "year", "tmID", "g", "w", "l", "t"])
    return _fill_zero(df, [c for c in df.columns if c.startswith("post")])


def _clean_scoring(df):
    non_numeric = ["playerID", "tmID", "lgID", "pos"]
    df = _to_numeric(df, [c for c in df.columns if c not in non_numeric])
    df = _drop_missing(df, ["playerID", "year", "tmID", "GP", "PIM"])
    df = df[df["GP"] > 0].copy()
    return _fill_zero(df, [c for c in df.columns if c.startswith("Post")])


def _clean_master(df):
    df = df[[c for c in MASTER_COLUMNS if c in df.columns]].copy()
    df = _to_numeric(df, ["birthYear", "height", "weight", "firstNHL", "lastNHL"])
    has_id = df["playerID"].notna() | df["coachID"].notna()
    return df[has_id]


def _clean_teams_post(df):
    df = _to_numeric(df, ["year", "G", "W", "L", "T", "GF", "GA", "PIM"])
    return _drop_missing(df, ["year", "tmID"])


def _clean_awards_coaches(df):
    df = df.drop(columns=["note"], errors="ignore")
    df = _to_numeric(df, ["year"])
    return _drop_missing(df, ["coachID", "award", "year"])


CLEANERS = {
    "Teams": _clean_teams,
    "Coaches": _clean_coaches,
    "Scoring": _clean_scoring,
    "Master": _clean_master,
    "TeamsPost": _clean_teams_post,
    "AwardsCoaches": _clean_awards_coaches,
}


def clean_datasets(datasets, verbose=True):
    cleaned = {}
    for name in TABLES_TO_KEEP:
        if name not in datasets:
            print(f"Warning: table '{name}' not found, skipping.")
            continue
        df = datasets[name]
        rows_before = len(df)
        df = _keep_nhl(df).drop_duplicates().copy()
        df = CLEANERS[name](df).reset_index(drop=True)
        cleaned[name] = df
        if verbose:
            print(f"{name:15s} {rows_before:>6} -> {len(df):>6} rows")
    return cleaned