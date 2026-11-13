"""Make the small per-notebook data extracts in data/.

Each notebook downloads its extract from
https://github.com/AllenDowney/DataQnA/raw/main/data/<name>.csv.gz
and reads it with pd.read_csv, so readers don't need PyTables.

Run from the repo root in the DataQnA environment (reading the HDF
sources needs PyTables, which the environment has):

    python data/make_extracts.py

Sources are downloaded into data/sources/ (ignored by git) if missing.
The gzip header timestamp is fixed, so regenerating an unchanged
extract gives an identical file.
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA = Path(__file__).parent
SOURCES = DATA / "sources"

EODS = "https://github.com/AllenDowney/ElementsOfDataScience/raw"
DATAQNA = "https://github.com/AllenDowney/DataQnA/raw/main/data"

GZIP = {"method": "gzip", "mtime": 0}


def source(url):
    """Download url into data/sources/ if it's not there; return the path."""
    SOURCES.mkdir(exist_ok=True)
    path = SOURCES / Path(url).name
    if not path.exists():
        urlretrieve(url, path)
        print("Downloaded", path)
    return path


def write(df, name):
    path = DATA / f"{name}.csv.gz"
    df.to_csv(path, index=False, compression=GZIP)
    print(f"Wrote {path} {df.shape} {path.stat().st_size / 1e6:.2f} MB")


def confidence():
    """BRFSS 2021 sex (_SEX) and height in cm (HTM4), from Elements of Data Science."""
    path = source(f"{EODS}/v1/data/brfss_2021.hdf")
    brfss = pd.read_hdf(path, "brfss")
    write(brfss[["_SEX", "HTM4"]], "confidence_brfss")


def likert_mean():
    """GSS variables for likert_mean, from DataQnA's gss_qna_extract.hdf."""
    path = source(f"{DATAQNA}/gss_qna_extract.hdf")
    gss = pd.read_hdf(path, "gss")
    columns = ["year", "sex", "polviews", "homosex", "conservatism"]
    write(gss[columns], "likert_mean_gss")


def resample_logit():
    """GSS variables for resample_logit, from Elements of Data Science."""
    path = source(f"{EODS}/master/data/gss_eda.hdf")
    gss = pd.read_hdf(path, "gss")
    columns = ["AGE", "EDUC", "SEX", "GUNLAW", "GRASS"]
    write(gss[columns], "resample_logit_gss")


def poisson_process():
    """NHL 2023-24 goal times in seconds, one row per goal.

    The HDF file has one Series per game (keyed by game ID). Games with no
    goals get one row with an empty time, so the notebook can keep them.
    """
    path = source(f"{DATAQNA}/nhl_2023_2024.hdf")
    rows = []
    with pd.HDFStore(path, "r") as store:
        for key in store.keys():
            game = key.strip("/")
            times = store[key]
            if len(times) == 0:
                rows.append((game, None))
            rows.extend((game, t) for t in times)
    df = pd.DataFrame(rows, columns=["game", "time"]).astype({"time": "Int64"})
    write(df, "poisson_process_nhl")


if __name__ == "__main__":
    confidence()
    likert_mean()
    resample_logit()
    poisson_process()
