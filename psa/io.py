"""Reading AMDA ASCII exports and simple gap utilities.

AMDA (Automated Multi-Dataset Analysis, Génot et al. 2021,
doi:10.1016/j.pss.2021.105214) writes '#'-prefixed header lines followed by
whitespace-separated rows.  With ``REQUEST_TIME_FORMAT : DOY TIME`` the first
column is a 16-digit token ``YYYYDDDHHMMSSmmm`` (year, day-of-year, hour,
minute, second, millisecond).  Missing values are written as ``nan``.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ID_SUFFIX = re.compile(r"_\d{6,}")      # AMDA appends a long request id
_INDEX = re.compile(r"\[(\d+)\]$")


@dataclass
class TimeSeries:
    """A uniformly or non-uniformly sampled multi-parameter time series."""

    time: np.ndarray                     # datetime64[ms]
    columns: dict[str, np.ndarray]       # name -> (N,) or (N, k) float arrays
    header: dict[str, str] = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        """Seconds since the first sample."""
        return (self.time - self.time[0]).astype("timedelta64[ms]").astype(float) / 1e3

    @property
    def dt(self) -> float:
        """Sampling interval in seconds; raises if the cadence is not uniform."""
        d = np.diff(self.t)
        if not np.allclose(d, d[0], rtol=0, atol=1e-6):
            raise ValueError("non-uniform cadence: %s" % np.unique(d)[:10])
        return float(d[0])

    def __len__(self) -> int:
        return len(self.time)


def _open(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def parse_doy_time(tokens) -> np.ndarray:
    """Convert ``YYYYDDDHHMMSSmmm`` tokens to ``datetime64[ms]``."""
    tok = np.asarray(tokens, dtype="U16")
    if np.any(np.char.str_len(tok) != 16):
        raise ValueError("expected 16-character DOY TIME tokens")
    year = tok.astype("U4").astype(int)
    rest = np.array([s[4:] for s in tok], dtype="U12")
    doy = np.array([s[0:3] for s in rest]).astype(int)
    hh = np.array([s[3:5] for s in rest]).astype(int)
    mm = np.array([s[5:7] for s in rest]).astype(int)
    ss = np.array([s[7:9] for s in rest]).astype(int)
    ms = np.array([s[9:12] for s in rest]).astype(int)
    jan1 = (year - 1970).astype("datetime64[Y]").astype("datetime64[ms]")
    offset_ms = ((((doy - 1) * 24 + hh) * 60 + mm) * 60 + ss) * 1000 + ms
    return jan1 + offset_ms.astype("timedelta64[ms]")


def read_amda(path) -> TimeSeries:
    """Read an AMDA 'all-in-one-file' ASCII export (optionally gzipped).

    Vector parameters (``name[0]``, ``name[1]``, ...) are grouped into (N, k)
    arrays; the AMDA request id suffix is stripped from the names, so e.g.
    ``b_ulys_rtn_7879902028113337413[0..2]`` becomes ``columns['b_ulys_rtn']``.
    """
    header: dict[str, str] = {}
    times: list[str] = []
    rows: list[list[str]] = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                body = line[1:].strip()
                if " : " in body:
                    key, val = body.split(" : ", 1)
                    header.setdefault(key.strip(), val.strip())
                continue
            parts = line.split()
            if parts:
                times.append(parts[0])
                rows.append(parts[1:])
    if "DATA_COLUMNS" not in header:
        raise ValueError("no DATA_COLUMNS header line found")
    names = [c.strip() for c in header["DATA_COLUMNS"].split(",")][1:]
    values = np.array(rows, dtype=float)
    if values.shape[1] != len(names):
        raise ValueError("DATA_COLUMNS lists %d columns, rows have %d" % (len(names), values.shape[1]))

    groups: dict[str, list[tuple[int, int]]] = {}
    for col, name in enumerate(names):
        m = _INDEX.search(name)
        comp = int(m.group(1)) if m else -1
        base = _ID_SUFFIX.sub("", _INDEX.sub("", name))
        groups.setdefault(base, []).append((comp, col))
    columns = {}
    for base, members in groups.items():
        members.sort()
        idx = [c for _, c in members]
        columns[base] = values[:, idx[0]] if members[0][0] == -1 else values[:, idx]
    return TimeSeries(time=parse_doy_time(times), columns=columns, header=header)


def nan_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of runs of True in a 1-D boolean mask."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    return starts, np.flatnonzero(d == -1) - starts


def interpolate_gaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs (per column) in index space; edges take the
    nearest valid value.  Returns a new array."""
    y = np.array(x, dtype=float, copy=True)
    one_d = y.ndim == 1
    if one_d:
        y = y[:, None]
    idx = np.arange(len(y))
    for j in range(y.shape[1]):
        bad = np.isnan(y[:, j])
        if bad.all():
            raise ValueError("column %d is entirely NaN" % j)
        if bad.any():
            y[bad, j] = np.interp(idx[bad], idx[~bad], y[~bad, j])
    return y[:, 0] if one_d else y
