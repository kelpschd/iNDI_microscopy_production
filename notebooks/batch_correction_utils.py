"""Shared helpers for per-plate batch correction of iNDI HCS features.

Written for the uoe5 nucleus / organelle analyses. Every function takes the
feature list and column names as arguments, so the same code runs for any
structure (nucleus, lysosome, endosome, mitochondria).

Typical flow
------------
    df_t, report, tcols = auto_transform(df_roi, RAW_FEATURES)
    wells, WF = build_well_table(df_t, tcols, meta_cols=[...], count_col="organelle_count")
    WF = finalize_features(wells, WF)
    corr = robust_z(wells, wells[wells.is_parental], WF, pooled_scale=True)
    res  = eval_method(wells, lambda w, p, f: robust_z(w, p, f, pooled_scale=True), WF)
    eff  = compute_effects(corr, WF)
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest, skew

PLATE = "iNDI_Plate_ID"
WELL_KEYS = ["iNDI_Plate_ID", "Well_ID"]


# --------------------------------------------------------------------------
# 1. Transforms
# --------------------------------------------------------------------------
def auto_transform(df, features, eps=1e-3, skew_thresh=1.0):
    """Data-driven transform of ROI-level features.

    Rule per feature (computed on the data passed in):
      - fraction-like (min >= 0 and max <= 1) and |skew| > skew_thresh -> logit
      - non-negative and skew > skew_thresh                           -> log1p
      - otherwise                                                      -> unchanged
    New columns are named ``t_<feature>``; originals are left untouched.
    Returns (df_with_t_columns, report_dataframe, list_of_t_columns).
    """
    out = df.copy()
    rows, tcols = [], []
    for f in features:
        if f not in out.columns:
            rows.append(dict(feature=f, transform="MISSING"))
            continue
        x = pd.to_numeric(out[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        xv = x.dropna()
        if len(xv) < 3 or xv.nunique() < 2:
            rows.append(dict(feature=f, transform="CONSTANT/EMPTY", frac_nan=x.isna().mean()))
            continue
        lo, hi, s_raw = xv.min(), xv.max(), skew(xv)
        if lo >= 0 and hi <= 1 and abs(s_raw) > skew_thresh:
            xc = x.clip(eps, 1 - eps)
            y, t = np.log(xc / (1 - xc)), "logit"
        elif lo >= 0 and s_raw > skew_thresh:
            y, t = np.log1p(x), "log1p"
        else:
            y, t = x, "none"
        out[f"t_{f}"] = y
        tcols.append(f"t_{f}")
        rows.append(dict(feature=f, transform=t, skew_raw=round(s_raw, 2),
                         skew_t=round(skew(y.dropna()), 2),
                         frac_nan=round(x.isna().mean(), 3), min=lo, max=hi))
    return out, pd.DataFrame(rows).set_index("feature"), tcols


# --------------------------------------------------------------------------
# 2. Well-level table
# --------------------------------------------------------------------------
def build_well_table(df, tcols, well_keys=WELL_KEYS, meta_cols=None,
                     count_col=None, parental_col="Gene", parental_value="KOLF2.1J"):
    """ROI -> well: median and IQR of each transformed feature, ROI count,
    optionally fraction of ROIs with count_col == 0 (e.g. no organelle objects).
    Metadata columns are carried with .first() after an invariance check.
    Returns (wells, feature_columns).
    """
    meta_cols = [m for m in (meta_cols or []) if m in df.columns]
    g = df.groupby(well_keys)
    for c in meta_cols:
        bad = (g[c].nunique() > 1).sum()
        if bad:
            print(f"WARNING: {c} varies within {bad} well(s)")
    med = g[tcols].median().add_suffix("_med")
    iqr = (g[tcols].quantile(0.75) - g[tcols].quantile(0.25)).add_suffix("_iqr")
    parts = [g[meta_cols].first(), g.size().rename("n_rois"), med, iqr]
    if count_col and count_col in df.columns:
        parts.append(g[count_col].apply(lambda s: (s == 0).mean()).rename("frac_roi_zero"))
    wells = pd.concat(parts, axis=1).reset_index()
    wells["log_n_rois"] = np.log(wells["n_rois"])
    wells["is_parental"] = (wells[parental_col] == parental_value) if parental_col in wells else False
    feats = [c for c in wells.columns if c.endswith(("_med", "_iqr"))] + ["log_n_rois"]
    if "frac_roi_zero" in wells:
        feats.append("frac_roi_zero")
    return wells, feats


def finalize_features(wells, feats, max_nan=0.05, min_sd=1e-8, max_mode_frac=0.5):
    """Drop features that are mostly NaN, constant, or zero-inflated (more than
    `max_mode_frac` of wells share one value, so MAD ~ 0 and a robust z-score is
    meaningless); fill remaining NaN with the feature's median. Modifies `wells`
    in place; returns kept features. Set max_mode_frac=1 to keep everything."""
    keep = []
    for f in feats:
        nan_frac = wells[f].isna().mean()
        if nan_frac > max_nan or wells[f].std() < min_sd:
            print(f"dropped {f}: nan_frac={nan_frac:.3f}, sd={wells[f].std():.2g}")
            continue
        mode_frac = wells[f].round(10).value_counts(normalize=True).iloc[0]
        if mode_frac > max_mode_frac:
            print(f"dropped {f}: {mode_frac:.0%} of wells share one value (zero-inflated)")
            continue
        if nan_frac > 0:
            wells[f] = wells[f].fillna(wells[f].median())
        keep.append(f)
    return keep


def prune_correlated(wells, feats, thresh=0.9, plate_col=PLATE, mask=None):
    """Greedy |r| pruning on plate-centred wells (plate effects would otherwise
    inflate correlations). The first feature in `feats` wins each tie, so order
    them by preference. Returns (kept, dropped_pairs)."""
    d = wells if mask is None else wells[mask]
    x = d[feats] - d.groupby(plate_col)[feats].transform("median")
    corr = x.corr().abs()
    keep, dropped = [], []
    for f in feats:
        clash = [k for k in keep if corr.loc[f, k] > thresh]
        if clash:
            dropped.append((f, clash[0], round(corr.loc[f, clash[0]], 3)))
        else:
            keep.append(f)
    return keep, dropped


# --------------------------------------------------------------------------
# 3. Correction
# --------------------------------------------------------------------------
def robust_z(wells, par_fit, feats, plate_col=PLATE, pooled_scale=True, shrink=False, eps=1e-9):
    """Per-plate centring on the parental median; scale by MAD*1.4826.

    pooled_scale=True  (M1b): one scale per feature, from all plates' centred
                              parentals -> effects in common units across plates.
    pooled_scale=False (M1) : per-plate MAD.
    shrink=True : empirical-Bayes shrinkage of each plate's centre toward the
                  grand centre. A feature with no real plate effect gets
                  (almost) one global centre instead of 13 noisy plate centres,
                  which avoids ADDING variance to unaffected features. Features
                  with strong plate effects are barely shrunk.
    Fit only on `par_fit` rows; applied to every row of `wells`. Features whose
    parental MAD is ~0 (zero-inflated) fall back to the SD, then to 1.0.
    """
    grp = par_fit.groupby(plate_col)[feats]
    plate_med = grp.median()
    n_p = grp.size().to_numpy()[:, None].astype(float)
    grand = plate_med.median()
    centres = plate_med.copy()
    if shrink:
        resid = par_fit[feats].to_numpy() - plate_med.loc[par_fit[plate_col]].to_numpy()
        s = 1.4826 * np.median(np.abs(resid - np.median(resid, axis=0)), axis=0)
        se2 = (1.2533 * s) ** 2 / n_p                       # sampling variance of each plate median
        tau2 = np.maximum(0.0, plate_med.var(ddof=1).to_numpy() - se2.mean(axis=0))
        B = tau2 / (tau2 + se2 + 1e-12)                     # plates x feats shrink factors
        centres = pd.DataFrame(grand.to_numpy() + B * (plate_med.to_numpy() - grand.to_numpy()),
                               index=plate_med.index, columns=feats)
    # plates with no parental wells fall back to the grand centre
    all_plates = pd.Index(wells[plate_col].unique())
    missing = all_plates.difference(centres.index)
    if len(missing):
        print(f"WARNING: no parental wells on {list(missing)}; using the grand centre")
    centres = centres.reindex(all_plates)
    centres = centres.fillna(grand)

    plates = wells[plate_col]
    centred_all = wells[feats].to_numpy() - centres.loc[plates].to_numpy()
    cp = par_fit[feats].to_numpy() - centres.loc[par_fit[plate_col]].to_numpy()

    def _safe(scale, fallback):
        scale = np.where(scale > 1e-8, scale, fallback)
        return np.where(scale > 1e-8, scale, 1.0)

    if pooled_scale:
        mad = 1.4826 * np.median(np.abs(cp - np.median(cp, axis=0)), axis=0)
        sd = cp.std(axis=0)
        scale = np.broadcast_to(_safe(mad, sd), centred_all.shape)
    else:
        mad = par_fit.assign(**{f: cp[:, i] for i, f in enumerate(feats)}).groupby(plate_col)[feats].apply(
            lambda d: 1.4826 * (d - d.median()).abs().median())
        sd = par_fit.assign(**{f: cp[:, i] for i, f in enumerate(feats)}).groupby(plate_col)[feats].std()
        scale = _safe(mad.reindex(all_plates).loc[plates].to_numpy(),
                      sd.reindex(all_plates).loc[plates].to_numpy())
    out = wells.copy()
    out[feats] = centred_all / (scale + eps)
    return out


# --------------------------------------------------------------------------
# 4. Evaluation on held-out parentals
# --------------------------------------------------------------------------
def omega_sq(values, groups):
    """Bias-corrected one-way omega-squared."""
    d = pd.DataFrame({"v": np.asarray(values), "g": np.asarray(groups)}).dropna()
    k, n, gm = d["g"].nunique(), len(d), d["v"].mean()
    if k < 2 or n <= k:
        return np.nan
    ssb = d.groupby("g")["v"].apply(lambda s: len(s) * (s.mean() - gm) ** 2).sum()
    sst = ((d["v"] - gm) ** 2).sum()
    msw = (sst - ssb) / (n - k)
    denom = sst + msw
    if denom <= 0:          # constant feature
        return np.nan
    return max(0.0, (ssb - (k - 1) * msw) / denom)


def split_parentals(wells, seed=0, frac_fit=0.5, plate_col=PLATE):
    """Return parental rows with a 'half' column: 0 = fit, 1 = held-out eval."""
    rng = np.random.default_rng(seed)
    p = wells[wells["is_parental"]].copy()
    p["half"] = p.groupby(plate_col)[plate_col].transform(
        lambda s: (rng.permutation(len(s)) >= int(round(len(s) * frac_fit))).astype(int))
    return p


def eval_method(wells, correct_fn, feats, seeds=range(20), frac_fit=0.5, plate_col=PLATE):
    """Held-out plate omega^2 (obs) and permutation null, per feature and seed.
    correct_fn(wells, par_fit, feats) -> corrected wells (same index)."""
    rows = []
    for s in seeds:
        r = np.random.default_rng(1000 + s)
        p = split_parentals(wells, seed=s, frac_fit=frac_fit, plate_col=plate_col)
        corr = correct_fn(wells, p[p["half"] == 0], feats)
        e = corr.loc[p.index[p["half"] == 1]]
        obs = pd.Series({f: omega_sq(e[f], e[plate_col]) for f in feats})
        null = pd.Series({f: omega_sq(e[f], r.permutation(e[plate_col].to_numpy())) for f in feats})
        rows.append(pd.DataFrame({"obs": obs, "null": null, "seed": s}))
    return pd.concat(rows)


# --------------------------------------------------------------------------
# 5. Variant effects, revertant null, allelic-series test
# --------------------------------------------------------------------------
def compute_effects(corr, feats, group_col="Gene_Var_Type", plate_col=PLATE,
                    rev_pattern="REV/WT"):
    """Plate-matched effect = median(variant wells) - median(parental wells on
    the same plate), in corrected units. Adds Mann-Whitney p (well level; this
    ignores clone noise and is anti-conservative -- calibrate against REV)."""
    rows = []
    for (gvt, plate), v in corr[~corr["is_parental"]].groupby([group_col, plate_col]):
        p_ = corr[corr["is_parental"] & (corr[plate_col] == plate)]
        for f in feats:
            rows.append(dict(**{group_col: gvt}, plate=plate, feature=f,
                             effect=v[f].median() - p_[f].median(),
                             p=mannwhitneyu(v[f], p_[f]).pvalue,
                             n_var=len(v), n_par=len(p_)))
    eff = pd.DataFrame(rows)
    eff["is_rev"] = eff[group_col].str.contains(rev_pattern, regex=False)
    return eff


def add_rev_z(eff):
    """Scale each effect by the robust spread of revertant effects for that
    feature. Returns (eff_with_z_rev_and_abs_z, rev_sd, rev_p95)."""
    rev = eff[eff["is_rev"]]
    rev_sd = rev.groupby("feature")["effect"].apply(
        lambda s: 1.4826 * (s - s.median()).abs().median())
    rev_sd = rev_sd.where(rev_sd > 1e-8)          # zero spread -> z undefined (NaN), not inf
    rev_p95 = rev.groupby("feature")["effect"].apply(lambda s: s.abs().quantile(0.95))
    eff = eff.copy()
    eff["z_rev"] = eff["effect"] / eff["feature"].map(rev_sd)
    eff["abs_z"] = eff["z_rev"].abs()
    return eff, rev_sd, rev_p95


def series_monotone_test(eff, group_col="Gene_Var_Type"):
    """For each gene_variant series with REV/WT, SNV/WT and SNV/SNV, test whether
    the fraction of monotone (REV, het, hom) orderings exceeds the 1/3 expected
    by chance (2 of 6 orderings are monotone)."""
    e = eff.copy()
    e[["series", "geno"]] = e[group_col].str.rsplit("_", n=1, expand=True)
    w = e.pivot_table(index=["series", "feature"], columns="geno", values="effect")
    need = ["REV/WT", "SNV/WT", "SNV/SNV"]
    if not set(need) <= set(w.columns):
        raise ValueError(f"need genotypes {need}, found {list(w.columns)}")
    w = w.dropna(subset=need)
    up = (w["REV/WT"] < w["SNV/WT"]) & (w["SNV/WT"] < w["SNV/SNV"])
    down = (w["REV/WT"] > w["SNV/WT"]) & (w["SNV/WT"] > w["SNV/SNV"])
    w["mono"] = up | down
    rows = []
    for f, d in w.groupby(level="feature"):
        k, n = int(d["mono"].sum()), len(d)
        rows.append(dict(feature=f, n_series=n, n_monotone=k, frac=round(k / n, 2),
                         p_vs_1_3=round(binomtest(k, n, 1 / 3, alternative="greater").pvalue, 3)))
    return pd.DataFrame(rows).sort_values("p_vs_1_3").reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. Visual demonstration that normalization worked
# --------------------------------------------------------------------------
def _pc_scores(df, feats, n=4):
    """PCA via SVD on per-feature z-scored data (no sklearn needed).
    Returns (scores[:, :n], variance_explained[:n])."""
    X = df[feats].to_numpy(float)
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    return U[:, :n] * S[:n], var[:n]


def normalization_report(wells, feats, correct_fn=None, corrected=None, tag="",
                         outdir=None, seed=0, n_seeds=5, top_k=4,
                         plate_col=PLATE, show=True):
    """Four figures that show whether per-plate correction worked.

    1. PCA of HELD-OUT parental wells, coloured by plate, before vs after
       (correction fitted on the other half of each plate's parentals, so the
       'after' panel is not circular). Plate omega^2 of each plotted PC is in
       the panel title.
    2. PCA of ALL wells (final fit on all parentals), coloured by plate and by
       group (parental / REV/WT / variant): plates should mix, and variants
       should sit inside the parental cloud unless they truly differ.
    3. Per-plate boxplots of the most plate-dependent features, before vs after.
    4. Plate omega^2 per feature, before vs after, with the permutation null.

    PCA is computed separately on the before and after data (each z-scored), so
    the two panels show each dataset's own dominant structure.
    correct_fn(wells, par_fit, feats) defaults to robust_z(pooled_scale=True);
    pass `corrected=` to reuse an already-corrected table for figure 2.
    Returns {name: Figure}; saves PNGs when outdir is given.
    """
    import pathlib
    import matplotlib.pyplot as plt

    if correct_fn is None:
        correct_fn = lambda w, p, f: robust_z(w, p, f, pooled_scale=True)
    plates = sorted(wells[plate_col].unique())
    cmap = (plt.cm.tab20(np.linspace(0, 1, 20)) if len(plates) <= 20
            else plt.cm.hsv(np.linspace(0, 1, len(plates), endpoint=False)))
    pal = {p: cmap[i % len(cmap)] for i, p in enumerate(plates)}
    figs = {}

    # ---- held-out parental data --------------------------------------
    par = split_parentals(wells, seed=seed, plate_col=plate_col)
    fit, ev_idx = par[par["half"] == 0], par.index[par["half"] == 1]
    after_all = correct_fn(wells, fit, feats)
    b, a = wells.loc[ev_idx], after_all.loc[ev_idx]

    # ---- Fig 1: held-out parental PCA ---------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for j, (d, name) in enumerate([(b, "before"), (a, "after")]):
        sc, var = _pc_scores(d, feats, 4)
        lab = d[plate_col].to_numpy()
        for i, (x, y) in enumerate([(0, 1), (2, 3)]):
            ax = axes[i, j]
            for p in plates:
                m = lab == p
                ax.scatter(sc[m, x], sc[m, y], s=16, alpha=0.75, color=pal[p], label=p, linewidths=0)
            om = [omega_sq(sc[:, k], lab) for k in (x, y)]
            ax.set_title(f"{name}: PC{x+1} plate $\\omega^2$={om[0]:.2f} | PC{y+1} $\\omega^2$={om[1]:.2f}", fontsize=10)
            ax.set_xlabel(f"PC{x+1} ({var[x]:.0%} var)")
            ax.set_ylabel(f"PC{y+1} ({var[y]:.0%} var)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=7, markerscale=1.4, title="plate")
    fig.suptitle(f"{tag} held-out parental wells (n={len(b)}): coloured by plate", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    figs["heldout_parental_pca"] = fig

    # ---- Fig 2: all wells ---------------------------------------------
    allc = corrected if corrected is not None else correct_fn(wells, wells[wells["is_parental"]], feats)
    gvt = wells["Gene_Var_Type"].astype(str) if "Gene_Var_Type" in wells else pd.Series("", index=wells.index)
    group = np.where(wells["is_parental"], "parental",
                     np.where(gvt.str.contains("REV/WT", regex=False), "REV/WT", "variant"))
    gcol = {"variant": "#9aa0a6", "REV/WT": "#e8710a", "parental": "#1a73e8"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for j, (d, name) in enumerate([(wells, "before"), (allc, "after")]):
        sc, var = _pc_scores(d, feats, 2)
        lab = d[plate_col].to_numpy()
        for p in plates:
            m = lab == p
            axes[0, j].scatter(sc[m, 0], sc[m, 1], s=9, alpha=0.6, color=pal[p], label=p, linewidths=0)
        for g_ in ("variant", "REV/WT", "parental"):          # parentals drawn last, on top
            m = group == g_
            axes[1, j].scatter(sc[m, 0], sc[m, 1], s=9, alpha=0.55, color=gcol[g_], label=g_, linewidths=0)
        om = omega_sq(sc[:, 0], lab)
        axes[0, j].set_title(f"{name}: all wells by plate (PC1 plate $\\omega^2$={om:.2f})", fontsize=10)
        axes[1, j].set_title(f"{name}: all wells by group", fontsize=10)
        for i in (0, 1):
            axes[i, j].set_xlabel(f"PC1 ({var[0]:.0%} var)")
            axes[i, j].set_ylabel(f"PC2 ({var[1]:.0%} var)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", fontsize=7, markerscale=1.4, title="plate", bbox_to_anchor=(0.995, 0.93))
    axes[1, 1].legend(fontsize=8, markerscale=1.6, loc="best")
    fig.suptitle(f"{tag} all wells (correction fit on all parentals)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    figs["all_wells_pca"] = fig

    # ---- Fig 3: per-plate boxplots of the most plate-dependent features
    om0 = pd.Series({f: omega_sq(b[f], b[plate_col]) for f in feats}).dropna()
    top = om0.sort_values(ascending=False).index[:top_k]
    order = b.groupby(plate_col)[top[0]].median().sort_values().index.tolist()
    fig, axes = plt.subplots(2, len(top), figsize=(4.2 * len(top), 8.5), squeeze=False)
    for j, f in enumerate(top):
        for i, (d, name) in enumerate([(b, "before"), (a, "after")]):
            ax = axes[i, j]
            data = [d.loc[d[plate_col] == p, f].dropna().to_numpy() for p in order]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.7)
            for patch, p in zip(bp["boxes"], order):
                patch.set_facecolor(pal[p]); patch.set_alpha(0.85)
            ax.set_title(f"{f}\n{name} (plate $\\omega^2$={omega_sq(d[f], d[plate_col]):.2f})", fontsize=9)
            if i == 1:
                ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=90, fontsize=7)
                ax.axhline(0, color="k", lw=0.6, ls=":")
            else:
                ax.set_xticks([])
    fig.suptitle(f"{tag} held-out parental wells per plate (ordered by '{top[0]}' median)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figs["feature_boxplots"] = fig

    # ---- Fig 4: omega^2 bars ------------------------------------------
    r0 = eval_method(wells, lambda w, p, f: w, feats, seeds=range(n_seeds), plate_col=plate_col)
    r1 = eval_method(wells, correct_fn, feats, seeds=range(n_seeds), plate_col=plate_col)
    t = pd.DataFrame({"before": r0.groupby(level=0)["obs"].mean(),
                      "after": r1.groupby(level=0)["obs"].mean()}).sort_values("before")
    null = r0.groupby(level=0)["null"].mean().mean()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.3 * len(t) + 1.5)))
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["before"], 0.4, label="before", color="#d93025")
    ax.barh(y - 0.2, t["after"], 0.4, label="after", color="#188038")
    ax.axvline(null, color="k", ls="--", lw=0.9, label=f"permutation null ({null:.3f})")
    ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=7)
    ax.set_xlabel("plate $\\omega^2$ on held-out parental wells (mean over seeds)")
    ax.set_title(f"{tag} plate effect per feature: mean before {t['before'].mean():.2f}, after {t['after'].mean():.3f}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    figs["omega_bars"] = fig

    if outdir:
        outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        stem = (tag.strip().replace(" ", "_") + "_") if tag else ""
        for k, f_ in figs.items():
            f_.savefig(outdir / f"{stem}{k}.png", dpi=150, bbox_inches="tight")
        print("saved:", ", ".join(f"{stem}{k}.png" for k in figs))
    if show:
        plt.show()
    else:
        for f_ in figs.values():
            plt.close(f_)
    return figs
