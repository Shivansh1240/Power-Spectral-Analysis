#!/usr/bin/env python3
"""Full turbulence analysis of an AMDA Ulysses export (default: 1994-04-10/11).

Usage:  python scripts/run_analysis.py [--data data/ULL_1994_Apr.txt.gz] [--out results]

Writes ``<out>/summary.json``, ``<out>/RESULTS.md`` and figures in
``<out>/figures``.  Every number quoted in the README comes from this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psa import alfvenicity, fitting, io, legacy, plasma, qc, spectra, structure, wavelets  # noqa: E402

# ---------------------------------------------------------------- settings
NPERSEG_LONG = 2 ** 15      # 9.1 h segments: low-frequency end and 1/f break
NPERSEG_SHORT = 2 ** 12     # 68 min gap-free segments: inertial and ion ranges
F_MAX_ABS = 0.1             # Hz; absolute spectra above this are distorted by resampling (see qc)
F_MAX_RATIO = 0.25          # Hz; ratio diagnostics (helicity, compressibility) shown up to here
PLASMA_SPACING = 480        # s; native SWOOPS cadence (4- or 8-min, mostly 8 here)
N_BOOT = 2000
RNG = np.random.default_rng(20240410)

C = dict(tr="#1f4e79", r="#c0392b", t="#27ae60", n="#8e44ad", fit="#e67e22", ref="0.45", grey="0.75")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def shade_unreliable(ax, fmax=F_MAX_ABS, label=True):
    ax.axvspan(fmax, 0.5, color="0.92", zorder=0, label="resampling-affected" if label else None)


def mark_scales(ax, sc, y=0.97):
    for key, lab in (("f_ci_Hz", r"$f_{ci}$"), ("f_di_Hz", r"$f_{d_i}$")):
        ax.axvline(sc[key], color=C["ref"], ls=":", lw=1)
        ax.text(sc[key], y, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                color="0.3", bbox=dict(fc="white", ec="none", pad=0.5))


def log_edges(c):
    """Bin edges (geometric midpoints) for log-spaced bin centres."""
    c = np.asarray(c, float)
    m = np.sqrt(c[1:] * c[:-1])
    return np.concatenate([[c[0] ** 2 / m[0]], m, [c[-1] ** 2 / m[-1]]])


def slope_changes(x, w=60, min_sep=150, thr=8.0):
    """Times where a piecewise-linear (interpolated) series changes slope."""
    x = io.interpolate_gaps(x)
    d = np.zeros(len(x))
    for j in range(x.shape[1]):
        c = x[:, j]
        sl = np.full(len(c), np.nan)
        sr = np.full(len(c), np.nan)
        sl[w:] = (c[w:] - c[:-w]) / w
        sr[:-w] = (c[w:] - c[:-w]) / w
        dd = np.nan_to_num(np.abs(sr - sl))
        noise = np.median(dd[dd > 0]) + 1e-30
        d += dd / noise
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(d, height=thr * x.shape[1], distance=min_sep)
    return peaks


def gap_injection(B, dt, nperseg, shifts_h=(6, 12, 18)):
    """Bias of linear gap filling, measured by adding the real gap pattern,
    shifted in time, to the data and comparing gap-filled spectra over the
    same segments.  Returns the median PSD ratio per decade band and the
    added gap fraction."""
    gap = np.isnan(B).any(axis=1)
    ref = spectra.welch(B, dt, nperseg, max_gap_fraction=1.0)
    bands = [(1e-3, 1e-2), (1e-2, 1e-1)]
    out = {"bands_Hz": bands, "shifts_h": list(shifts_h), "added_gap_fraction": [], "median_ratio": []}
    for sh in shifts_h:
        inj = np.roll(gap, int(sh * 3600 / dt)) & ~gap
        B2 = B.copy()
        B2[inj] = np.nan
        r = spectra.welch(B2, dt, nperseg, max_gap_fraction=1.0).trace / ref.trace
        out["added_gap_fraction"].append(float(inj.mean()))
        out["median_ratio"].append([float(np.median(r[(ref.f >= a) & (ref.f < b)])) for a, b in bands])
    return out


def legacy_leakage_demo(n, fb=0.05, a1=-5 / 3, a2=-3.0, seed=7):
    """Legacy vs. new slope on a synthetic, non-periodic broken power law
    (a periodic synthetic series would hide leakage, so a sub-span is used)."""
    from psa.synthetic import broken_power_law, colored_noise
    x = colored_noise(4 * n, 1.0, broken_power_law(fb, a1, a2), rng=np.random.default_rng(seed))[:n]
    leg = legacy.legacy_slope(x, fit_range=(1.2 * fb, 0.4))[0]
    s = spectra.welch(x, 1.0, NPERSEG_SHORT)
    new = fitting.fit_power_law(s.f, s.trace, 1.2 * fb, 0.4, s.dof, s.ordinate_corr)
    leg_in = legacy.legacy_slope(x, fit_range=(1e-3, 1e-2))[0]
    new_in = fitting.fit_power_law(s.f, s.trace, 1e-3, 1e-2, s.dof, s.ordinate_corr)
    return {"true_alpha_steep": a2, "legacy_steep": leg, "new_steep": [new.alpha, new.alpha_err],
            "true_alpha_inertial": a1, "legacy_inertial": leg_in, "new_inertial": [new_in.alpha, new_in.alpha_err]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "ULL_1994_Apr.txt.gz"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    S = {}   # summary

    # ------------------------------------------------------------ load
    ts = io.read_amda(args.data)
    B = ts.columns["b_ulys_rtn"]
    n = ts.columns["n_p_ulys"]
    V = ts.columns["v_ulys_rtn"]
    dt = ts.dt
    t = ts.t
    th = t / 3600
    Bmag = spectra.magnitude(B)
    gapB = np.isnan(B).any(axis=1)
    gapP = np.isnan(n) | np.isnan(V).any(axis=1)
    gs, gl = io.nan_runs(gapB)
    knots = slope_changes(np.column_stack([n, V]))
    S["data"] = {
        "file": Path(args.data).name, "start": str(ts.time[0]), "stop": str(ts.time[-1]),
        "n_samples": len(ts), "dt_s": dt, "duration_h": float(t[-1] / 3600),
        "B_gaps": {"count": int(len(gs)), "total_s": int(gl.sum()), "longest_s": int(gl.max()),
                   "fraction": float(gapB.mean())},
        "plasma_gap_fraction": float(gapP.mean()),
        "plasma_slope_change_spacing_s_median": float(np.median(np.diff(knots))),
        "plasma_slope_change_spacing_s_hist": {str(k): int(v) for k, v in
                                               zip(*np.unique(np.round(np.diff(knots), -1), return_counts=True))},
        "amda_header": {k: ts.header[k] for k in ("AMDA_VERSION", "REQUEST_OUTPUT_PARAMS", "INTERVAL_START",
                                                  "INTERVAL_STOP") if k in ts.header},
    }

    # ------------------------------------------------------------ plasma context
    Vmag = np.linalg.norm(V, axis=1)
    sc = plasma.characteristic_scales(np.nanmean(Bmag), np.nanmean(n), np.nanmean(Vmag))
    Bmean = np.nanmean(B, axis=0)
    polarity = float(np.sign(Bmean[0]))
    dB_rms = float(np.sqrt(np.nansum(np.nanvar(B, axis=0))))
    flow_angle = np.degrees(np.arccos(np.nanmean(V[:, 0] / Vmag)))
    S["plasma"] = {**sc, "B_mean_vector_nT": Bmean.tolist(), "abs_mean_B_vector_nT": float(np.linalg.norm(Bmean)),
                   "polarity_sign_BR": polarity, "dB_rms_nT": dB_rms, "dB_rms_over_B": dB_rms / sc["B_nT"],
                   "std_absB_over_mean": float(np.nanstd(Bmag) / np.nanmean(Bmag)),
                   "mean_flow_angle_from_R_deg": float(flow_angle),
                   "parker_angle_deg_mean_field": float(np.degrees(np.arccos(abs(Bmean[0]) / np.linalg.norm(Bmean))))}

    # ------------------------------------------------------------ QC: resampling artefact
    tq, rq = qc.hf_ratio_series(B, dt)
    jump = qc.gap_jump_test(B, dt)
    S["qc"] = {"hf_ratio_band_Hz": [0.35, 0.5], "ref_band_Hz": [0.1, 0.15],
               "hf_ratio_p5_p50_p95": np.percentile(rq, [5, 50, 95]).tolist(),
               "hf_ratio_max_over_min": float(rq.max() / rq.min()),
               "gap_jumps": [{"t_h": float(a / 3600), "gap_s": float(b), "dlog10": float(c)}
                             for a, b, c in zip(jump["gap_time_s"], jump["gap_length_s"], jump["gap_dlog10"])],
               "median_abs_dlog10_at_gaps": jump["median_abs_gap"],
               "median_abs_dlog10_control": jump["median_abs_control"],
               "p90_abs_dlog10_control": jump["p90_abs_control"],
               "F_MAX_ABS_Hz": F_MAX_ABS, "F_MAX_RATIO_Hz": F_MAX_RATIO,
               "uniform_offset_interp_power_loss_at_F_MAX": float(1 - qc.interpolation_transfer(F_MAX_ABS))}

    # ------------------------------------------------------------ Welch spectra
    L = spectra.welch(B, dt, NPERSEG_LONG, max_gap_fraction=1.0)
    Sh = spectra.welch(B, dt, NPERSEG_SHORT, max_gap_fraction=0.0, cross=True, keep_segments=True)
    Shf = spectra.welch(B, dt, NPERSEG_SHORT, max_gap_fraction=1.0)
    Mg = spectra.welch(Bmag, dt, NPERSEG_SHORT, max_gap_fraction=0.0)
    S["welch"] = {
        "long": {"nperseg": NPERSEG_LONG, "segments": L.n_segments, "dof": L.dof, "gaps": "interpolated"},
        "short": {"nperseg": NPERSEG_SHORT, "segments": Sh.n_segments, "dof": Sh.dof, "gaps": "segments rejected"},
        "window": "hann", "overlap": 0.5, "detrend": "linear", "ordinate_corr_c_w": Sh.ordinate_corr,
        "gap_injection_test": gap_injection(B, dt, NPERSEG_SHORT),
    }

    # ------------------------------------------------------------ breaks and indices
    b1 = fitting.fit_broken_power_law(L.f, L.trace, 2.0 / (NPERSEG_LONG * dt), 0.03, L.dof, L.ordinate_corr)
    b2 = fitting.fit_broken_power_law(Sh.f, Sh.trace, 3e-3, F_MAX_ABS, Sh.dof, Sh.ordinate_corr)
    Hbar = qc.interpolation_transfer(Sh.f)
    b2c = fitting.fit_broken_power_law(Sh.f, Sh.trace / Hbar, 3e-3, F_MAX_ABS, Sh.dof, Sh.ordinate_corr)
    b1_k = {k0: fitting.fit_broken_power_law(L.f, L.trace, k0 / (NPERSEG_LONG * dt), 0.03, L.dof, L.ordinate_corr)
            for k0 in (3, 4)}
    f_lo, f_hi = 3.0 * b1.fb, b2.fb / 2.0
    inert = fitting.fit_power_law(Sh.f, Sh.trace, f_lo, f_hi, Sh.dof, Sh.ordinate_corr)
    binned = spectra.log_bin(Sh.f, Sh.trace, 20)
    lsq_a, lsq_e = fitting.fit_power_law_lsq(binned.f, binned.y, f_lo, f_hi)
    kin = fitting.fit_power_law(Sh.f, Sh.trace, b2.fb, F_MAX_ABS, Sh.dof, Sh.ordinate_corr)
    kinc = fitting.fit_power_law(Sh.f, Sh.trace / Hbar, b2c.fb, F_MAX_ABS, Sh.dof, Sh.ordinate_corr)

    # bootstrap over non-overlapping gap-free segments
    idx = fitting.non_overlapping(Sh.segment_starts, NPERSEG_SHORT)
    segtr = Sh.segment_psd[idx].sum(axis=2)
    boot = fitting.bootstrap_segments(
        segtr, lambda P: fitting.fit_power_law(Sh.f, P, f_lo, f_hi, Sh.dof, 1.0).alpha, N_BOOT, RNG)
    # fit-range sensitivity
    grid = [(a * b1.fb, b2.fb / c) for a in (2, 3, 5) for c in (1.5, 2, 3)]
    range_alphas = [fitting.fit_power_law(Sh.f, Sh.trace, lo, hi, Sh.dof, Sh.ordinate_corr).alpha for lo, hi in grid]
    # sub-intervals (non-stationarity)
    quarters = []
    nq = 4
    qlen = len(B) // nq
    for q in range(nq):
        sl = slice(q * qlen, (q + 1) * qlen)
        try:
            sq = spectra.welch(B[sl], dt, NPERSEG_SHORT, max_gap_fraction=0.0)
            fq = fitting.fit_power_law(sq.f, sq.trace, f_lo, f_hi, sq.dof, sq.ordinate_corr)
            quarters.append({"q": q + 1, "t_h": [q * qlen / 3600, (q + 1) * qlen / 3600], "segments": sq.n_segments,
                             "alpha": fq.alpha, "alpha_err": fq.alpha_err, "B_nT": float(np.nanmean(Bmag[sl])),
                             "V_kms": float(np.nanmean(Vmag[sl])), "n_cm3": float(np.nanmean(n[sl]))})
        except ValueError as exc:
            quarters.append({"q": q + 1, "error": str(exc)})
    qa = np.array([q["alpha"] for q in quarters if "alpha" in q])

    S["fits"] = {
        "break_1f_to_inertial": {"f_b_Hz": b1.fb, "f_b_ci68_Hz": list(b1.fb_ci), "alpha_below": b1.alpha1,
                                 "alpha_below_err": b1.alpha1_err, "alpha_above": b1.alpha2,
                                 "alpha_above_err": b1.alpha2_err, "range_Hz": [b1.fmin, b1.fmax],
                                 "start_bin_sensitivity": {str(k): {"f_b_Hz": v.fb, "alpha_below": v.alpha1,
                                                                    "alpha_below_err": v.alpha1_err}
                                                           for k, v in b1_k.items()}},
        "break_ion": {"f_b_Hz": b2.fb, "f_b_ci68_Hz": list(b2.fb_ci), "alpha_below": b2.alpha1,
                      "alpha_below_err": b2.alpha1_err, "alpha_above": b2.alpha2, "alpha_above_err": b2.alpha2_err,
                      "range_Hz": [b2.fmin, b2.fmax], "f_b_over_f_ci": b2.fb / sc["f_ci_Hz"],
                      "f_b_over_f_di": b2.fb / sc["f_di_Hz"]},
        "break_ion_interp_corrected": {"f_b_Hz": b2c.fb, "f_b_ci68_Hz": list(b2c.fb_ci), "alpha_above": b2c.alpha2,
                                       "alpha_above_err": b2c.alpha2_err},
        "inertial": {"range_Hz": [f_lo, f_hi], "alpha_mle": inert.alpha, "alpha_err_stat": inert.alpha_err,
                     "alpha_lsq_logbinned": lsq_a, "alpha_lsq_err": lsq_e,
                     "bootstrap": {"segments": int(len(idx)), "n_boot": N_BOOT, "std": float(boot.std()),
                                   "p16_p50_p84": np.percentile(boot, [16, 50, 84]).tolist()},
                     "range_sensitivity_min_max": [float(min(range_alphas)), float(max(range_alphas))],
                     "quarters": quarters, "quarters_mean": float(qa.mean()), "quarters_std": float(qa.std(ddof=1)),
                     "z_vs_minus_5_3": (inert.alpha + 5 / 3) / float(boot.std()),
                     "z_vs_minus_3_2": (inert.alpha + 1.5) / float(boot.std())},
        "kinetic_onset": {"range_Hz": [b2.fb, F_MAX_ABS], "alpha": kin.alpha, "alpha_err": kin.alpha_err,
                          "alpha_interp_corrected": kinc.alpha, "alpha_interp_corrected_err": kinc.alpha_err},
    }

    # ------------------------------------------------------------ legacy comparison
    try:
        with np.errstate(all="ignore"):
            raw_slope = legacy.legacy_slope(B[:, 0])[0]
        raw_result = "slope=%r" % raw_slope
    except Exception as exc:  # the original script on gappy data
        raw_result = "%s: %s" % (type(exc).__name__, exc)
    leg_slope, leg_f, leg_P = legacy.legacy_slope(io.interpolate_gaps(B[:, 0]))
    Sr = spectra.welch(B[:, 0], dt, NPERSEG_SHORT, max_gap_fraction=0.0)
    new_r = fitting.fit_power_law(Sr.f, Sr.trace, 1e-3, 1e-2, Sr.dof, Sr.ordinate_corr)
    new_tr = fitting.fit_power_law(Sh.f, Sh.trace, 1e-3, 1e-2, Sh.dof, Sh.ordinate_corr)
    S["legacy_comparison"] = {"original_script_on_raw_file": raw_result,
                              "legacy_slope_BR_gaps_filled_1_10mHz": leg_slope,
                              "new_mle_BR_1_10mHz": [new_r.alpha, new_r.alpha_err],
                              "new_mle_trace_1_10mHz": [new_tr.alpha, new_tr.alpha_err],
                              "synthetic_leakage_demo": legacy_leakage_demo(len(B))}

    # ------------------------------------------------------------ compressibility & helicity (Fourier)
    sig_m = spectra.reduced_magnetic_helicity(Sh)
    comp = Mg.trace / Sh.trace

    # ------------------------------------------------------------ wavelets
    W = wavelets.analyse(B, V, dt, 1e-4, F_MAX_RATIO, dj=0.125, block_seconds=6 * 3600)
    bias = wavelets.power_law_bias(inert.alpha)
    th_c = 0.5 * (W.theta_edges[1:] + W.theta_edges[:-1])
    win = (W.f >= f_lo) & (W.f <= f_hi)
    cnt_all = W.theta_count.sum(axis=0)

    def theta_slopes(mask):
        P = W.theta_psd(mask)
        cnt = W.theta_count[mask].sum(axis=0)
        out_ = np.full(len(th_c), np.nan)
        for k in range(len(th_c)):
            ok = win & np.isfinite(P[:, k]) & (cnt[:, k] >= 200)
            if ok.sum() >= 0.8 * win.sum():
                out_[k] = np.polyfit(np.log(W.f[ok]), np.log(P[ok, k]), 1)[0]
        return out_

    a_theta, a_theta_err = wavelets.jackknife(theta_slopes, W.n_blocks)
    Pth = W.theta_psd()
    j_mid = int(np.argmin(np.abs(W.f - np.sqrt(f_lo * f_hi))))
    ratio_par_perp = Pth[j_mid, 0] / Pth[j_mid, -1]
    hel_rect, hel_rect_err = wavelets.jackknife(lambda m: W.helicity(m, rectify=True), W.n_blocks)
    S["wavelet"] = {
        "omega0": wavelets.OMEGA0, "dj": 0.125, "f_range_Hz": [float(W.f.min()), float(W.f.max())],
        "level_bias_for_alpha": bias,
        "median_ratio_wavelet_over_welch_in_inertial": float(np.median(
            W.psd_trace[win] / np.interp(W.f[win], Sh.f, Sh.trace))),
        "theta_bins_deg": W.theta_edges.tolist(),
        "theta_count_fraction_mid_scale": (cnt_all[j_mid] / cnt_all[j_mid].sum()).tolist(),
        "alpha_vs_theta": a_theta.tolist(), "alpha_vs_theta_jackknife_err": a_theta_err.tolist(),
        "P_theta0_10_over_P_theta80_90_at_f_Hz": [float(W.f[j_mid]), float(ratio_par_perp)],
        "par_fraction_inertial_median": float(np.median((W.psd_par / W.psd_trace)[win])),
        "mag_fraction_inertial_median": float(np.median((W.psd_mag / W.psd_trace)[win])),
        "par_fraction_at_f": {"%g" % fr: float(np.interp(fr, W.f[::-1], (W.psd_par / W.psd_trace)[::-1]))
                              for fr in (0.01, 0.05, 0.1, 0.2)},
        "mag_fraction_at_f": {"%g" % fr: float(np.interp(fr, W.f[::-1], (W.psd_mag / W.psd_trace)[::-1]))
                              for fr in (0.01, 0.05, 0.1, 0.2)},
    }
    groups = {"quasi_parallel_0_30": slice(0, 3), "oblique_40_90": slice(4, 9)}

    def hel_group(mask, sl):
        num = W.hel_num[mask].sum(0)
        den = W.hel_den[mask].sum(0)
        num = (num - num[:, ::-1])[:, :9]
        den = (den + den[:, ::-1])[:, :9]
        return num[:, sl].sum(1) / den[:, sl].sum(1)

    hel_groups = {k: wavelets.jackknife(lambda m, sl=sl: hel_group(m, sl), W.n_blocks) for k, sl in groups.items()}
    S["wavelet"]["helicity_rectified_groups"] = {
        k: {"%gHz" % fr: [float(v[0][int(np.argmin(np.abs(W.f - fr)))]), float(v[1][int(np.argmin(np.abs(W.f - fr)))])]
            for fr in (0.02, 0.05, 0.1, 0.2)} for k, v in hel_groups.items()}
    for fr in (0.05, 0.1, 0.2):
        j = int(np.argmin(np.abs(W.f - fr)))
        S["wavelet"]["helicity_rectified_at_%gHz" % fr] = {
            "f_Hz": float(W.f[j]), "sigma_m": hel_rect[j].tolist(), "err": hel_rect_err[j].tolist()}

    # ------------------------------------------------------------ structure functions
    lags = structure.log_lags(4 * 3600, 64)
    SF = structure.structure_functions(B, lags)
    FL = structure.flatness(B, lags)
    lags_w = lags[lags <= 600]
    FLw = []
    for s0 in range(0, len(B) - 3600 + 1, 3600):
        seg = B[s0:s0 + 3600]
        if np.isnan(seg).mean() <= 0.1:
            FLw.append(structure.flatness(seg, lags_w).mean(axis=1))
    FLw = np.median(np.array(FLw), axis=0)
    sel = (lags >= 1 / f_hi) & (lags <= 1 / f_lo)
    zeta2, zeta2_cov = np.polyfit(np.log(lags[sel]), np.log(SF[2][sel]), 1, cov=True)
    dBB = np.sqrt(SF[2]) / sc["B_nT"]
    S["structure"] = {"tau_fit_s": [float(lags[sel].min()), float(lags[sel].max())],
                      "zeta2": float(zeta2[0]), "zeta2_err_formal": float(np.sqrt(zeta2_cov[0, 0])),
                      "implied_alpha": float(-zeta2[0] - 1),
                      "flatness_global_at_tau_s": {str(int(L_)): float(FL[i].mean()) for i, L_ in enumerate(lags)
                                                   if L_ in (1, 10, 100, 1000, 10000) or i == len(lags) - 1},
                      "flatness_1h_windows_median_at_tau_s": {str(int(L_)): float(FLw[i]) for i, L_ in
                                                              enumerate(lags_w) if L_ in (1, 10, 100) or
                                                              i == len(lags_w) - 1},
                      "dB_over_B_at_tau_1_over_fb1": float(np.interp(1 / b1.fb, lags, dBB)),
                      "tau_where_dB_over_B_equals_1_s": float(np.interp(1.0, dBB, lags)) if dBB.max() > 1 else None}

    # ------------------------------------------------------------ Alfvenicity (plasma resolution)
    Bs = alfvenicity.matched_smooth(B, PLASMA_SPACING)
    dec = 60
    n_med = alfvenicity.running_median(n[::dec], 61)          # 1-h running median
    n_outliers = int(np.sum(n[::dec] < 0.5 * n_med))
    al = alfvenicity.alfvenicity_spectra(V[::dec], Bs[::dec], n_med, dt * dec, 512, polarity)
    # top of band: triangle responses for 240-s and 480-s spacings differ by < 8% below 1/(8*480 s)
    band = (al["f"] >= 2 * al["f"][0]) & (al["f"] <= 1.0 / (8 * PLASMA_SPACING))
    Ep, Em, Ev, Eb = (al[k][band].sum() for k in ("E_plus", "E_minus", "E_v", "E_b"))
    S["alfvenicity"] = {"band_Hz": [float(al["f"][band].min()), float(al["f"][band].max())],
                        "sigma_c_band": float((Ep - Em) / (Ep + Em)), "sigma_r_band": float((Ev - Eb) / (Ev + Eb)),
                        "E_plus_over_E_minus_band": float(Ep / Em), "segments": al["n_segments"], "dof": al["dof"],
                        "polarity": polarity, "density": "1-h running median",
                        "density_samples_below_half_running_median_60s": n_outliers}

    # ================================================================ figures
    # 1. overview
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for k, (lab, col) in enumerate(zip(("$B_R$", "$B_T$", "$B_N$"), (C["r"], C["t"], C["n"]))):
        ax[0].plot(th, B[:, k], lw=0.3, color=col, label=lab)
    ax[0].set_ylabel("B (nT)")
    ax[0].legend(ncol=3, loc="upper right", fontsize=8)
    ax[1].plot(th, Bmag, lw=0.3, color=C["tr"])
    ax[1].set_ylabel("|B| (nT)")
    ax[2].plot(th, n, lw=0.8, color="k")
    ax[2].set_ylabel(r"$n_p$ (cm$^{-3}$)")
    ax[3].plot(th, V[:, 0], lw=0.8, color="k")
    ax[3].set_ylabel(r"$V_R$ (km/s)")
    ax[3].set_xlabel("hours since %s UT" % str(ts.time[0])[:19].replace("T", " "))
    for a in ax:
        for s0, ln in zip(gs, gl):
            a.axvspan(s0 / 3600, (s0 + ln) / 3600, color="orange", alpha=0.35, lw=0)
    ax[0].set_title("Ulysses FGM (1 s) and SWOOPS moments via AMDA; orange = magnetic-field gaps")
    fig.tight_layout()
    fig.savefig(figs / "fig01_overview.png")
    plt.close(fig)

    # 2. trace PSD, compensated, local slope
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.5, 1.5]))
    lo, hi = L.confidence()
    ax[0].fill_between(L.f, lo, hi, color=C["grey"], alpha=0.5, lw=0, label="95% CI (9.1-h Welch)")
    ax[0].loglog(L.f, L.trace, color="0.4", lw=0.5)
    vis = Sh.f >= 3.0 / (NPERSEG_SHORT * dt)   # lowest bins are depressed by per-segment detrending
    ax[0].loglog(Sh.f[vis], Sh.trace[vis], color=C["tr"], lw=0.6, label="68-min Welch, gap-free segments")
    ff = np.logspace(np.log10(b1.fmin), np.log10(b1.fmax), 200)
    ax[0].loglog(ff, b1.model(ff) * 3, color=C["fit"], lw=1.4,
                 label=r"break 1: $f_b$=%.2g Hz, $\alpha$=%.2f / %.2f (×3)" % (b1.fb, b1.alpha1, b1.alpha2))
    ff = np.logspace(np.log10(b2.fmin), np.log10(b2.fmax), 200)
    ax[0].loglog(ff, b2.model(ff) / 3, color=C["r"], lw=1.4,
                 label=r"break 2: $f_b$=%.3g Hz, $\alpha$=%.2f / %.2f (÷3)" % (b2.fb, b2.alpha1, b2.alpha2))
    ax[0].set_ylabel(r"trace PSD (nT$^2$ Hz$^{-1}$)")
    ax[0].legend(fontsize=7.5, loc="lower left")
    shade_unreliable(ax[0], label=False)
    mark_scales(ax[0], sc)
    for a, (P_, f_, lab) in zip([ax[1]], [(Sh.trace, Sh.f, None)]):
        bb = spectra.log_bin(f_, P_ * f_ ** (5 / 3), 20, fmin=3.0 / (NPERSEG_SHORT * dt))
        a.loglog(bb.f, bb.y, "o-", ms=2.5, color=C["tr"], lw=0.8)
        bl = spectra.log_bin(L.f, L.trace * L.f ** (5 / 3), 20, fmax=1e-3)
        a.loglog(bl.f, bl.y, "s-", ms=2.5, color="0.4", lw=0.8)
    ax[1].axvspan(f_lo, f_hi, color=C["fit"], alpha=0.12, lw=0)
    ax[1].set_ylabel(r"$f^{5/3}\,P$")
    shade_unreliable(ax[1], label=False)
    fc1, a1, e1 = fitting.local_slope(L.f, L.trace, L.dof, L.ordinate_corr, 0.5, 0.1, fmin=L.f[1], fmax=1.5e-3)
    fc2, a2, e2 = fitting.local_slope(Sh.f, Sh.trace, Sh.dof, Sh.ordinate_corr, 0.3, 0.05, fmin=1e-3, fmax=0.5)
    ax[2].errorbar(fc1, a1, e1, fmt="s", ms=3, color="0.4", lw=0.8, label="9.1-h Welch, 0.5-dec window")
    ax[2].errorbar(fc2, a2, e2, fmt="o", ms=3, color=C["tr"], lw=0.8, label="68-min Welch, 0.3-dec window")
    for v_, lab in ((-1, "−1"), (-1.5, "−3/2"), (-5 / 3, "−5/3"), (-2.8, "−2.8")):
        ax[2].axhline(v_, color=C["ref"], lw=0.8, ls="--")
        ax[2].text(1.1e-4, v_, lab, fontsize=7.5, va="bottom", color="0.3")
    ax[2].set_ylim(-4, 0)
    ax[2].set_ylabel(r"local index $\alpha$")
    ax[2].set_xlabel("spacecraft-frame frequency (Hz)")
    ax[2].legend(fontsize=7.5, loc="lower left")
    shade_unreliable(ax[2], label=False)
    ax[2].set_xlim(L.f[0], 0.5)
    ax[0].set_title("Magnetic trace spectrum; grey band = resampling-affected (> %.2g Hz)" % F_MAX_ABS)
    fig.tight_layout()
    fig.savefig(figs / "fig02_trace_spectrum.png")
    plt.close(fig)

    # 3. estimator comparison
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(Sh.f[vis], Sh.trace[vis], color=C["tr"], lw=0.6, label="Welch 68 min, gap-free segments")
    ax.loglog(Shf.f[vis], Shf.trace[vis] * 1.5, color=C["t"], lw=0.6, label="Welch 68 min, gaps interpolated (×1.5)")
    ax.loglog(W.f, W.psd_trace / bias * 2.25, color=C["r"], lw=1.4,
              label="Morlet global spectrum, gap-masked, bias-corrected (×2.25)")
    ax.loglog(leg_f, 2 * leg_P * 0.1, color="0.5", lw=0.6,
              label=r"legacy: $|X|^2/n$ of $B_R$, 50-pt moving avg (×2 units, ×0.1)")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"PSD (nT$^2$ Hz$^{-1}$), offset for clarity")
    ax.legend(fontsize=7.5)
    shade_unreliable(ax, label=False)
    ax.set_xlim(1e-4, 0.5)
    ax.set_title("Four estimators of the same spectrum (vertically offset)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "fig03_estimators.png")
    plt.close(fig)

    # 4. QC
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].semilogy(tq / 3600, rq, ".", ms=3, color=C["tr"])
    for s0 in gs:
        ax[0].axvline(s0 / 3600, color="orange", lw=0.8, alpha=0.8)
    ax[0].set_xlabel("hours")
    ax[0].set_ylabel("P(0.35–0.5 Hz) / P(0.1–0.15 Hz)")
    ax[0].set_title("High-frequency power ratio, 10-min windows\n(orange = data gaps)", fontsize=9)
    ax[1].hist(jump["control_abs_dlog10"], bins=30, color=C["grey"], label="random cut points")
    for v_ in np.abs(jump["gap_dlog10"]):
        ax[1].axvline(v_, color="orange", lw=1.2)
    ax[1].set_xlabel(r"|$\Delta\log_{10}$ ratio| across cut")
    ax[1].set_title("Jumps across gaps (orange) vs random cuts", fontsize=9)
    ax[1].legend(fontsize=8)
    ff = np.linspace(0.001, 0.5, 400)
    ax[2].plot(ff, qc.interpolation_transfer(ff, a=0.5), color=C["r"], label="offset a = 0.5")
    ax[2].plot(ff, qc.interpolation_transfer(ff), color=C["tr"], label="a uniform on [0, 1]")
    ax[2].axvline(F_MAX_ABS, color="k", ls="--", lw=0.8)
    ax[2].set_xlabel("frequency (Hz)")
    ax[2].set_ylabel(r"$|H(f)|^2$")
    ax[2].set_title("Linear-interpolation transfer function", fontsize=9)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fig04_qc_resampling.png")
    plt.close(fig)

    # 5. anisotropy
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    cmap = plt.get_cmap("viridis")
    for k in range(len(th_c)):
        ok = cnt_all[:, k] >= 200
        ax[0].loglog(W.f[ok], Pth[ok, k] / bias, color=cmap(k / (len(th_c) - 1)), lw=1,
                     label=r"%d–%d°" % (W.theta_edges[k], W.theta_edges[k + 1]))
    ax[0].axvspan(f_lo, f_hi, color=C["fit"], alpha=0.1, lw=0)
    shade_unreliable(ax[0], label=False)
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel(r"trace PSD (nT$^2$ Hz$^{-1}$)")
    ax[0].legend(fontsize=6.5, ncol=2, title=r"$\theta_{VB}$", title_fontsize=7)
    ax[0].set_xlim(1e-4, F_MAX_RATIO)
    ax[1].errorbar(th_c, a_theta, a_theta_err, fmt="o-", color=C["tr"], capsize=3)
    for v_, lab in ((-2, "−2 (critical balance, ∥)"), (-5 / 3, "−5/3 (GS95, ⊥)"), (-1.5, "−3/2 (IK / Boldyrev)")):
        ax[1].axhline(v_, color=C["ref"], ls="--", lw=0.8)
        ax[1].text(1, v_ + 0.01, lab, fontsize=7.5, color="0.3")
    ax[1].set_xlabel(r"$\theta_{VB}$ (deg)")
    ax[1].set_ylabel(r"spectral index over [%.1e, %.1e] Hz" % (f_lo, f_hi))
    ax[1].set_title("Index vs local field angle (jackknife errors, 6 blocks)", fontsize=9)
    ax[2].bar(th_c, cnt_all[j_mid] / cnt_all[j_mid].sum(), width=9, color=C["grey"])
    ax[2].set_xlabel(r"$\theta_{VB}$ (deg)")
    ax[2].set_ylabel("fraction of valid coefficients")
    ax[2].set_title("Angle sampling at f = %.3g Hz" % W.f[j_mid], fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig05_anisotropy.png")
    plt.close(fig)

    # 6. compressibility
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bb = spectra.log_bin(Sh.f, comp, 15)
    ax.loglog(bb.f, bb.y, "o-", ms=3, color=C["tr"], label=r"$P_{|B|}/P_{tr}$ (Fourier)")
    ax.loglog(W.f, W.psd_mag / W.psd_trace, color=C["t"], lw=1.2, label=r"$P_{|B|}/P_{tr}$ (wavelet)")
    ax.loglog(W.f, W.psd_par / W.psd_trace, color=C["r"], lw=1.2,
              label=r"$P_\parallel/P_{tr}$ (wavelet, local $\mathbf{B}_0$)")
    ax.axhline(1 / 3, color=C["ref"], ls="--", lw=0.8)
    ax.text(1.2e-4, 0.35, "isotropic 1/3", fontsize=7.5, color="0.3")
    mark_scales(ax, sc)
    ax.axvspan(F_MAX_ABS, F_MAX_RATIO, color="0.95", zorder=0)
    ax.set_xlim(1e-4, F_MAX_RATIO)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power ratio")
    ax.legend(fontsize=8)
    ax.set_title("Magnetic compressibility (ratios are insensitive to a common linear filter)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig06_compressibility.png")
    plt.close(fig)

    # 7. helicity
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    H = W.helicity()
    cnt_h = W.hel_count.sum(axis=0)
    Hm = np.where(cnt_h >= 200, H, np.nan)
    ext_t = W.hel_edges
    pc = ax[0].pcolormesh(ext_t, log_edges(W.f), Hm, cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\theta_{VB}$ (deg, unfolded)")
    ax[0].set_ylabel("frequency (Hz)")
    ax[0].set_title(r"$\sigma_m(f, \theta_{VB})$ as measured", fontsize=9)
    fig.colorbar(pc, ax=ax[0])
    Hr = np.where(cnt_h[:, :9] + cnt_h[:, 9:][:, ::-1] >= 200, hel_rect, np.nan)
    pc = ax[1].pcolormesh(W.theta_edges, log_edges(W.f), Hr, cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="auto")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"folded $\theta_{VB}$ (deg)")
    ax[1].set_title(r"rectified $\sigma_m^*$ (as if $\mathbf{B}_0$ anti-sunward)", fontsize=9)
    fig.colorbar(pc, ax=ax[1])
    for a in ax[:2]:
        a.axhline(F_MAX_ABS, color="k", ls="--", lw=0.8)
        a.axhline(sc["f_ci_Hz"], color="k", ls=":", lw=0.8)
        a.set_ylim(1e-3, F_MAX_RATIO)
    for (k, (v, e)), col, lab in zip(hel_groups.items(), (C["n"], C["r"]),
                                     (r"$\theta_{VB}<30°$", r"$40°<\theta_{VB}<90°$")):
        ok = W.f >= 1e-3
        ax[2].fill_between(W.f[ok], (v - e)[ok], (v + e)[ok], color=col, alpha=0.25, lw=0)
        ax[2].semilogx(W.f[ok], v[ok], color=col, lw=1.2, label=lab + " (rectified, wavelet)")
    bb = spectra.log_bin(Sh.f, sig_m, 15, fmin=1e-3)
    ax[2].semilogx(bb.f, bb.y, "k.", ms=4, label="Fourier, all angles, not rectified")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].axvline(F_MAX_ABS, color="k", ls="--", lw=0.8)
    ax[2].set_xscale("log")
    ax[2].set_ylim(-0.6, 0.6)
    ax[2].set_xlabel("frequency (Hz)")
    ax[2].set_ylabel(r"$\sigma_m^*$")
    ax[2].legend(fontsize=7.5, loc="lower left")
    ax[2].set_xlim(1e-3, 0.5)
    ax[2].set_title(r"$\sigma_m^*$ by angle group ($\pm1$ jackknife s.e.)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig07_helicity.png")
    plt.close(fig)

    # 8. scalogram
    fig, ax = plt.subplots(figsize=(10, 4))
    comp_sg = np.log10(W.scalogram * W.f[:, None] ** (5 / 3))
    half = 0.5 * (W.scalogram_t[1] - W.scalogram_t[0])
    t_edges = np.append(W.scalogram_t - half, W.scalogram_t[-1] + half) / 3600
    pc = ax.pcolormesh(t_edges, log_edges(W.f), comp_sg, cmap="magma", shading="auto",
                       vmin=np.nanpercentile(comp_sg, 2), vmax=np.nanpercentile(comp_sg, 98))
    ax.set_yscale("log")
    ax.axhline(F_MAX_ABS, color="w", ls="--", lw=0.8)
    ax.set_xlabel("hours")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(r"Morlet scalogram, $\log_{10}(f^{5/3} P_{tr})$, 2-min bins; white = invalid (edges or gaps)",
                 fontsize=9)
    fig.colorbar(pc, ax=ax)
    fig.tight_layout()
    fig.savefig(figs / "fig08_scalogram.png")
    plt.close(fig)

    # 9. structure functions
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    ax[0].loglog(lags, SF[2], "o", ms=3, color=C["tr"])
    ax[0].loglog(lags[sel], np.exp(np.polyval(zeta2, np.log(lags[sel]))) * 1.5, color=C["fit"],
                 label=r"$\zeta_2$ = %.3f → $\alpha$ = %.3f (×1.5)" % (zeta2[0], -zeta2[0] - 1))
    ax[0].set_xlabel(r"$\tau$ (s)")
    ax[0].set_ylabel(r"$S_2(\tau)$ (nT$^2$)")
    ax[0].legend(fontsize=8)
    for k, (lab, col) in enumerate(zip(("R", "T", "N"), (C["r"], C["t"], C["n"]))):
        ax[1].semilogx(lags, FL[:, k], "o-", ms=2.5, lw=0.8, color=col, label=lab)
    ax[1].semilogx(lags_w, FLw, "k^-", ms=3, lw=1, label="mean of R,T,N in 1-h windows (median)")
    ax[1].axhline(3, color=C["ref"], ls="--")
    ax[1].axvspan(0.5, 1 / F_MAX_ABS, color="0.92", zorder=0)
    ax[1].set_xlabel(r"$\tau$ (s)")
    ax[1].set_ylabel("flatness")
    ax[1].legend(fontsize=7.5)
    ax[1].set_title("Flatness (whole interval: coloured; 1-h windows: black)", fontsize=9)
    ax[2].loglog(lags, dBB, "o-", ms=3, color=C["tr"])
    ax[2].axhline(1, color=C["ref"], ls="--")
    ax[2].axvline(1 / b1.fb, color=C["fit"], ls=":", label=r"$1/f_{b,1}$")
    ax[2].set_xlabel(r"$\tau$ (s)")
    ax[2].set_ylabel(r"$\delta B(\tau)/\langle|B|\rangle$")
    ax[2].legend(fontsize=8)
    ax[2].set_title("Normalised amplitude (Matteini et al. 2018 test)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig09_structure_functions.png")
    plt.close(fig)

    # 10. Alfvenicity
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, lab, col in (("E_plus", "$E^+$ (outward)", C["r"]), ("E_minus", "$E^-$ (inward)", C["tr"]),
                        ("E_v", "$E_v$", C["t"]), ("E_b", "$E_b$", C["n"])):
        ax[0].loglog(al["f"], al[k], lw=1, color=col, label=lab)
    ax[0].axvspan(1.0 / (8 * PLASMA_SPACING), al["f"].max(), color="0.92", zorder=0)
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel(r"PSD (km$^2$ s$^{-2}$ Hz$^{-1}$)")
    ax[0].legend(fontsize=8)
    ax[1].semilogx(al["f"], al["sigma_c"], "o-", ms=3, color=C["r"], label=r"$\sigma_c$")
    ax[1].semilogx(al["f"], al["sigma_r"], "s-", ms=3, color=C["tr"], label=r"$\sigma_r$")
    ax[1].axvspan(1.0 / (8 * PLASMA_SPACING), al["f"].max(), color="0.92", zorder=0)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Cross helicity and residual energy (grey = beyond plasma resolution)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig10_alfvenicity.png")
    plt.close(fig)

    # 11. stationarity
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    qs = [q for q in quarters if "alpha" in q]
    ax[0].errorbar([np.mean(q["t_h"]) for q in qs], [q["alpha"] for q in qs], [q["alpha_err"] for q in qs],
                   fmt="o", capsize=3, color=C["tr"])
    ax[0].axhline(-5 / 3, color=C["ref"], ls="--")
    ax[0].axhline(-1.5, color=C["ref"], ls=":")
    ax[0].set_xlabel("hours (centre of 9-h quarter)")
    ax[0].set_ylabel(r"inertial index $\alpha$")
    ax[0].set_title("Sub-interval indices (statistical errors)", fontsize=9)
    ax[1].hist(boot, bins=40, color=C["grey"])
    ax[1].axvline(inert.alpha, color=C["tr"], label="MLE")
    ax[1].axvline(-5 / 3, color="k", ls="--", label="−5/3")
    ax[1].axvline(-1.5, color="k", ls=":", label="−3/2")
    ax[1].set_xlabel(r"bootstrap $\alpha$ (%d non-overlapping segments)" % len(idx))
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fig11_stationarity.png")
    plt.close(fig)

    # ================================================================ write
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    write_markdown(out / "RESULTS.md", S)
    print(json.dumps({k: S["fits"][k] for k in ("break_1f_to_inertial", "break_ion", "kinetic_onset")}, indent=1,
                     default=float))
    print("inertial:", {k: v for k, v in S["fits"]["inertial"].items() if k != "quarters"})
    return S


def write_markdown(path, S):
    f = S["fits"]
    i = f["inertial"]
    p = S["plasma"]
    w = S["wavelet"]
    d = S["data"]
    rows = [
        ("Interval", "%s → %s (%.1f h, %d samples at %g s)" % (d["start"], d["stop"], d["duration_h"],
                                                               d["n_samples"], d["dt_s"])),
        ("B gaps", "%d gaps, %d s total (%.2f%%), longest %d s" % (d["B_gaps"]["count"], d["B_gaps"]["total_s"],
                                                                 100 * d["B_gaps"]["fraction"],
                                                                 d["B_gaps"]["longest_s"])),
        ("⟨|B|⟩, ⟨n_p⟩, ⟨V⟩", "%.3f nT, %.3f cm⁻³, %.1f km/s" % (p["B_nT"], p["n_cm3"], p["V_kms"])),
        ("V_A, V_A/V", "%.1f km/s, %.3f" % (p["V_A_kms"], p["V_A_over_V"])),
        ("d_i, f_ci, f_di", "%.0f km, %.4f Hz, %.3f Hz" % (p["d_i_km"], p["f_ci_Hz"], p["f_di_Hz"])),
        ("δB_rms/⟨|B|⟩, σ(|B|)/⟨|B|⟩", "%.2f, %.3f" % (p["dB_rms_over_B"], p["std_absB_over_mean"])),
        ("1/f → inertial break", "%.2e Hz (68%%: %.2e–%.2e); α = %.2f ± %.2f below" % (
            f["break_1f_to_inertial"]["f_b_Hz"], *f["break_1f_to_inertial"]["f_b_ci68_Hz"],
            f["break_1f_to_inertial"]["alpha_below"], f["break_1f_to_inertial"]["alpha_below_err"])),
        ("Inertial index (MLE)", "%.3f ± %.3f (stat) over %.2e–%.2e Hz" % (i["alpha_mle"], i["alpha_err_stat"],
                                                                       *i["range_Hz"])),
        ("  segment bootstrap σ", "%.3f (%d segments)" % (i["bootstrap"]["std"], i["bootstrap"]["segments"])),
        ("  fit-range spread", "%.3f to %.3f" % tuple(i["range_sensitivity_min_max"])),
        ("  9-h quarters", "%s (std %.3f)" % (", ".join("%.3f" % q["alpha"] for q in i["quarters"] if "alpha" in q),
                                              i["quarters_std"])),
        ("  LSQ on log bins", "%.3f ± %.3f" % (i["alpha_lsq_logbinned"], i["alpha_lsq_err"])),
        ("Ion-scale break", "%.3f Hz (68%%: %.3f–%.3f) = %.1f f_ci = %.2f f_di" % (
            f["break_ion"]["f_b_Hz"], *f["break_ion"]["f_b_ci68_Hz"], f["break_ion"]["f_b_over_f_ci"],
            f["break_ion"]["f_b_over_f_di"])),
        ("Index f_b → 0.1 Hz", "%.2f ± %.2f raw; %.2f ± %.2f if resampling = linear interp. (uniform offset)" % (
            f["kinetic_onset"]["alpha"], f["kinetic_onset"]["alpha_err"],
            f["kinetic_onset"]["alpha_interp_corrected"], f["kinetic_onset"]["alpha_interp_corrected_err"])),
        ("S₂ slope ζ₂ → α", "%.3f → %.3f" % (S["structure"]["zeta2"], S["structure"]["implied_alpha"])),
        ("α(θ_VB) 0–10° / 80–90°", "%.2f ± %.2f / %.2f ± %.2f" % (
            w["alpha_vs_theta"][0], w["alpha_vs_theta_jackknife_err"][0], w["alpha_vs_theta"][-1],
            w["alpha_vs_theta_jackknife_err"][-1])),
        ("P_∥/P_tr, P_|B|/P_tr (inertial)", "%.3f, %.3f" % (w["par_fraction_inertial_median"],
                                                            w["mag_fraction_inertial_median"])),
        ("σ_c, σ_r (%.1e–%.1e Hz)" % tuple(S["alfvenicity"]["band_Hz"]), "%.2f, %.2f" % (
            S["alfvenicity"]["sigma_c_band"], S["alfvenicity"]["sigma_r_band"])),
        ("Legacy script on raw file", S["legacy_comparison"]["original_script_on_raw_file"]),
        ("Legacy slope (B_R, gaps filled, 1–10 mHz)", "%.3f" % S["legacy_comparison"]["legacy_slope_BR_gaps_filled_1_10mHz"]),
        ("New MLE, B_R / trace, 1–10 mHz", "%.3f ± %.3f / %.3f ± %.3f" % (
            *S["legacy_comparison"]["new_mle_BR_1_10mHz"], *S["legacy_comparison"]["new_mle_trace_1_10mHz"])),
    ]
    lines = ["# Results (auto-generated by `scripts/run_analysis.py`)", "", "| quantity | value |", "|---|---|"]
    lines += ["| %s | %s |" % r for r in rows]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
