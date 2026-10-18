import numpy as np

from psa import io

SAMPLE = """# AMDA_ABOUT : Created by AMDA
# INTERVAL_START : 1994-04-10T00:00:00.000
# DATA_COLUMNS : Time UT, b_ulys_rtn_123456789[0], b_ulys_rtn_123456789[1], b_ulys_rtn_123456789[2], n_p_ulys_123456789
#
  1994100000000000       -0.798        0.667       -0.181      0.28404
  1994100000001000          nan          nan          nan      0.28404
  1994365235959999        1e-3          2.0          3.0          nan
"""


def test_read_amda(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text(SAMPLE)
    ts = io.read_amda(p)
    assert set(ts.columns) == {"b_ulys_rtn", "n_p_ulys"}
    assert ts.columns["b_ulys_rtn"].shape == (3, 3)
    assert str(ts.time[0]) == "1994-04-10T00:00:00.000"
    assert str(ts.time[1]) == "1994-04-10T00:00:01.000"
    assert str(ts.time[2]) == "1994-12-31T23:59:59.999"
    assert np.isnan(ts.columns["b_ulys_rtn"][1]).all()
    assert ts.header["INTERVAL_START"] == "1994-04-10T00:00:00.000"


def test_gap_utilities():
    x = np.array([0.0, np.nan, np.nan, 3.0, np.nan])
    s, n = io.nan_runs(np.isnan(x))
    assert list(s) == [1, 4] and list(n) == [2, 1]
    np.testing.assert_allclose(io.interpolate_gaps(x), [0, 1, 2, 3, 3])
