"""
nifty_lab.py  --  Price-only quantitative forecasting lab for the Nifty 50.

Pipeline for one forecast (all inputs strictly dated <= origin day):
  1. Features  : 12-1m momentum, trend (log P / SMA200), drawdown depth  (vol-normalised)
  2. Drift     : long-run prior + ridge-shrunk tilt learned from past (feature -> fwd 6m return) pairs
  3. Volatility: GARCH(1,1) / GJR-GARCH(1,1) fitted on the last L days, long-run variance anchored
  4. Shocks    : filtered historical simulation (bootstrap of standardised residuals) or Gaussian
  5. Output    : N simulated paths -> quantile fan, P(up), drawdown odds ...
"""
import numpy as np, pandas as pd, warnings
from arch import arch_model
warnings.filterwarnings("ignore")

TD = 250
QS = np.array([.05, .10, .25, .50, .75, .90, .95])
H_ROLL = 125          # ~6 months of trading days for rolling backtests


class Lab:
    def __init__(self, path="nifty_daily_clean.csv"):
        d = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        self.d = d
        self.dates = d.index
        self.close = d["close"].values.astype(float)
        self.lp = np.log(self.close)
        self.r = np.r_[np.nan, np.diff(self.lp)]
        self.F = self._features()
        self.fits, self.shock_cache = {}, {}

    # ------------------------------------------------------------------ helpers
    def pos(self, date):
        """index of last trading day <= date"""
        return int(self.dates.searchsorted(pd.Timestamp(date), side="right") - 1)

    def _features(self):
        c = pd.Series(self.close, index=self.dates)
        lp = np.log(c)
        vol252 = lp.diff().rolling(252).std() * np.sqrt(TD)
        mom = (lp.shift(21) - lp.shift(252)) / (vol252 * np.sqrt(231 / TD))
        trend = np.log(c / c.rolling(200).mean()) / vol252
        dd = (c / c.rolling(252).max() - 1) / vol252
        return pd.DataFrame({"mom": mom.clip(-3, 3), "trend": trend.clip(-3, 3),
                             "dd": dd.clip(-3, 0)}, index=self.dates)

    # ------------------------------------------------------------------ drift tilt
    def tilt(self, o, H, spec):
        """Ridge-shrunk drift tilt (log-return over H days) using only pairs fully realised by day o."""
        if spec["drift"] == "prior":
            return 0.0
        if spec["drift"] == "bayes":        # prior blended with trailing 3y sample mean (precision-weighted)
            Lb = 750
            if o < Lb:
                return 0.0
            mu_s = (self.lp[o] - self.lp[o - Lb]) / (Lb / TD)
            sig = np.nanstd(self.r[o - Lb + 1:o + 1]) * np.sqrt(TD)
            se2 = sig ** 2 / (Lb / TD)
            tau2 = spec.get("tau", 0.04) ** 2
            w = (1 / tau2) / (1 / tau2 + 1 / se2)
            return float((w * spec["prior"] + (1 - w) * mu_s - spec["prior"]) * H / TD)
        cols = ["mom"] if spec["drift"] == "mom" else ["mom", "trend", "dd"]
        Fm = self.F[cols].values
        idx = np.arange(0, o - H + 1)
        X = Fm[idx]
        y = self.lp[idx + H] - self.lp[idx] - spec["prior"] * H / TD
        ok = ~np.isnan(X).any(1)
        X, y = X[ok], y[ok]
        n = len(y)
        if n < 250:
            return 0.0
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        yc = y - y.mean()                       # tilt is zero-mean: it only re-shapes, never re-levels
        lam = spec.get("lam", 10) * n
        beta = np.linalg.solve(Xs.T @ Xs + lam * np.eye(len(cols)), Xs.T @ yc)
        x_now = (Fm[o] - mu) / sd
        if np.isnan(x_now).any():
            return 0.0
        return float(np.clip(x_now @ beta, -0.06, 0.06))

    # ------------------------------------------------------------------ volatility
    def get_fit(self, o, vol, L):
        key = (o, vol, L)
        if key in self.fits:
            return self.fits[key]
        r100 = self.r[o - L + 1:o + 1] * 100
        assert not np.isnan(r100).any()
        am = arch_model(r100, mean="Zero", vol="GARCH", p=1, o=1 if vol == "gjr" else 0, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        p = res.params
        a, b = float(p["alpha[1]"]), float(p["beta[1]"])
        g = float(p["gamma[1]"]) if vol == "gjr" else 0.0
        om = float(p["omega"])
        pers = a + g / 2 + b
        if pers > 0.995:
            s = 0.995 / pers
            a, b, g = a * s, b * s, g * s
        hT = float(np.asarray(res.conditional_volatility)[-1] ** 2)
        rT = r100[-1]
        h1 = om + (a + g * (rT < 0)) * rT ** 2 + b * hT
        z = np.asarray(res.std_resid)
        z = z[np.isfinite(z)]
        z = (z - z.mean()) / z.std()
        out = dict(a=a, b=b, g=g, om=om, h1=h1, z=z, var_win=float(np.var(r100)), pers=a + g / 2 + b)
        self.fits[key] = out
        return out

    def long_var(self, o):
        return float(np.var(self.r[1:o + 1] * 100))

    def shocks(self, o, H, vol, anchor, tails, L, N, seed):
        key = (o, H, vol, anchor, tails, L, N, seed)
        if key in self.shock_cache:
            return self.shock_cache[key]
        f = self.get_fit(o, vol, L)
        a, b, g = f["a"], f["b"], f["g"]
        pers = a + g / 2 + b
        av = f["var_win"] if anchor == "window" else self.long_var(o)
        om = av * (1 - pers)
        rng = np.random.default_rng(seed)
        Z = f["z"][rng.integers(0, len(f["z"]), size=(N, H))] if tails == "fhs" else rng.standard_normal((N, H))
        h = np.full(N, f["h1"])
        R = np.empty((N, H)); hv = np.empty(H)
        for t in range(H):
            hv[t] = h.mean()
            r = np.sqrt(h) * Z[:, t]
            R[:, t] = r
            h = om + (a + g * (r < 0)) * r * r + b * h
        out = (R / 100, hv)
        if N <= 5000:
            self.shock_cache[key] = (R.sum(1) / 100, hv)   # rolling evaluation needs only terminal sums
            return self.shock_cache[key]
        return out

    # ------------------------------------------------------------------ one full forecast
    def forecast(self, o, H, spec, N=20000, seed=42):
        assert o < len(self.close)
        R, hv = self.shocks(o, H, spec["vol"], spec["anchor"], spec["tails"], spec["L"], N, seed)
        tl = self.tilt(o, H, spec)
        muH = spec["prior"] * H / TD + tl
        us = spec.get("volunc", 0.0)
        if us > 0:
            m = np.exp(us * np.random.default_rng(seed + 1).standard_normal(len(R)) - us ** 2 / 2)
            R = R * m[:, None]
        cum = spec["k"] * np.cumsum(R, axis=1) + muH * np.arange(1, H + 1) / H
        return dict(cum=cum, hv=hv, tilt=tl, muH=muH, o=o, H=H, spec=spec)

    def regime(self, o):
        c = self.close
        sma200 = c[o - 199:o + 1].mean()
        return dict(
            date=str(self.dates[o].date()), close=float(c[o]),
            vs_sma200_pct=float((c[o] / sma200 - 1) * 100),
            mom_12_1_z=float(self.F["mom"].iloc[o]), trend_z=float(self.F["trend"].iloc[o]),
            drawdown_pct=float((c[o] / c[o - 251:o + 1].max() - 1) * 100),
            vol63_ann_pct=float(np.nanstd(self.r[o - 62:o + 1]) * np.sqrt(TD) * 100),
            vol750_ann_pct=float(np.nanstd(self.r[o - 749:o + 1]) * np.sqrt(TD) * 100),
        )

    # ------------------------------------------------------------------ exam scoring
    @staticmethod
    def _pinball(y, q):
        e = y - q
        return float(np.maximum(QS * e, (QS - 1) * e).mean())

    def score_exam(self, fc, o_end):
        o, H = fc["o"], fc["H"]
        cum = fc["cum"]
        act = self.lp[o + 1:o_end + 1] - self.lp[o]
        assert len(act) == H
        term = cum[:, -1]
        ya = act[-1]
        q = np.quantile(term, QS)
        pit = float((term < ya).mean())
        band = lambda lo, hi: float(((act >= np.quantile(cum, lo, axis=0)) & (act <= np.quantile(cum, hi, axis=0))).mean())
        # drawdown / trough measures
        runmax = np.maximum.accumulate(np.c_[np.zeros(len(cum)), cum], axis=1)[:, 1:]
        mdd_sim = (cum - runmax).min(1)
        ar = np.maximum.accumulate(np.r_[0, act])[1:]
        mdd_act = float((act - ar).min())
        min_sim = cum.min(1); min_act = float(act.min())
        # vol
        vol_fc = float(np.sqrt(fc["hv"].mean()) * np.sqrt(TD) / 100 * fc["spec"]["k"])
        vol_act = float(np.std(np.diff(np.r_[0, act])) * np.sqrt(TD))
        # naive benchmarks
        sig = np.nanstd(self.r[o - 749:o + 1]) * np.sqrt(H)
        from scipy.stats import norm
        naive0 = norm.ppf(QS) * sig
        naive1 = fc["spec"]["prior"] * H / TD + norm.ppf(QS) * sig
        return dict(
            H=H, act_ret_pct=float((np.exp(ya) - 1) * 100), fc_median_pct=float((np.exp(np.median(term)) - 1) * 100),
            fc_mean_pct=float((np.exp(term).mean() - 1) * 100),
            q_pct=[float((np.exp(x) - 1) * 100) for x in q],
            err_vs_median_pp=float((np.exp(ya) - np.exp(np.median(term))) * 100),
            pit=pit, in50=bool(q[2] <= ya <= q[4]), in80=bool(q[1] <= ya <= q[5]), in90=bool(q[0] <= ya <= q[6]),
            p_up=float((term > 0).mean()), dir_hit=bool((term > 0).mean() > .5) == bool(ya > 0),
            path_cov50=band(.25, .75), path_cov80=band(.10, .90), path_cov90=band(.05, .95),
            min_act_pct=float((np.exp(min_act) - 1) * 100), min_pctile=float((min_sim < min_act).mean()),
            p_touch_m10=float((min_sim < np.log(.90)).mean()), p_touch_m5=float((min_sim < np.log(.95)).mean()),
            mdd_act_pct=float((np.exp(mdd_act) - 1) * 100), mdd_pctile=float((mdd_sim < mdd_act).mean()),
            mdd_median_pct=float((np.exp(np.median(mdd_sim)) - 1) * 100),
            vol_fc_pct=vol_fc * 100, vol_act_pct=vol_act * 100,
            pin_model=self._pinball(ya, q), pin_naive0=self._pinball(ya, naive0), pin_naive1=self._pinball(ya, naive1),
        )

    # ------------------------------------------------------------------ rolling-origin backtest
    def build_rolling(self, o_exam, H, N=4000, Ls=(500, 750), step=21):
        """Monthly origins anchored backwards from the exam origin, so the exam outcome is the last fold."""
        origins = sorted([o_exam - step * j for j in range(0, 400) if o_exam - step * j >= 750])
        y = np.array([self.lp[o + H] - self.lp[o] for o in origins])
        Q = {}
        for L in Ls:
            for vol in ("garch", "gjr"):
                for anchor in ("window", "long"):
                    for tails in ("gauss", "fhs"):
                        Ss = [self.shocks(o, H, vol, anchor, tails, L, N, seed=o)[0] for o in origins]
                        for us in (0.0, 0.25, 0.4):
                            rows = []
                            for o, S in zip(origins, Ss):
                                if us > 0:
                                    S = S * np.exp(us * np.random.default_rng(o + 7).standard_normal(len(S)) - us ** 2 / 2)
                                rows.append(np.quantile(S, QS))
                            Q[(L, vol, anchor, tails, us)] = np.array(rows)
        tilts = {d: np.array([self.tilt(o, H, dict(drift=d, prior=0.09, lam=10)) for o in origins])
                 for d in ("prior", "mom", "multi", "bayes")}
        sig = np.array([np.nanstd(self.r[o - 749:o + 1]) * np.sqrt(H) for o in origins])
        return dict(origins=origins, y=y, Q=Q, tilt=tilts, sig=sig, H=H)

    def evaluate(self, R, spec):
        key = (spec["L"], spec["vol"], spec["anchor"], spec["tails"], spec.get("volunc", 0.0))
        muH = spec["prior"] * R["H"] / TD + R["tilt"][spec["drift"]]
        F = spec["k"] * R["Q"][key] + muH[:, None]
        y = R["y"]
        e = y[:, None] - F
        pin = np.maximum(QS * e, (QS - 1) * e).mean(1)
        cov = dict(c50=float(((y >= F[:, 2]) & (y <= F[:, 4])).mean()),
                   c80=float(((y >= F[:, 1]) & (y <= F[:, 5])).mean()),
                   c90=float(((y >= F[:, 0]) & (y <= F[:, 6])).mean()))
        return pin, cov

    def naive_scores(self, R, prior=0.09):
        from scipy.stats import norm
        z = norm.ppf(QS)
        out = {}
        for nm, mu in (("naive_zero_drift", 0.0), ("naive_prior_drift", prior * R["H"] / TD)):
            F = mu + R["sig"][:, None] * z[None, :]
            e = R["y"][:, None] - F
            out[nm] = float(np.maximum(QS * e, (QS - 1) * e).mean())
        return out

    def greedy(self, R, incumbent, dims, thresh=0.02):
        cur, hist = dict(incumbent), []
        for dname in dims:
            cur.setdefault(dname, {"volunc": 0.0}.get(dname))
        while True:
            base = self.evaluate(R, cur)[0]
            best = None
            for dim, vals in dims.items():
                for v in vals:
                    if v == cur[dim]:
                        continue
                    cand = dict(cur, **{dim: v})
                    pin = self.evaluate(R, cand)[0]
                    imp = 1 - pin.mean() / base.mean()
                    th = np.array_split(np.arange(len(pin)), 3)
                    wins = sum(pin[t].mean() < base[t].mean() for t in th)
                    if imp > thresh and wins >= 2 and (best is None or imp > best[0]):
                        best = (imp, dim, v, cand, wins)
            if best is None:
                break
            cur = best[3]
            hist.append(dict(dim=best[1], to=best[2], gain_pct=round(best[0] * 100, 1), thirds_won=int(best[4])))
        return cur, hist

    # ------------------------------------------------------------------ hypothesis table
    def hypotheses(self, o, H=H_ROLL):
        idx = np.arange(252, o - H + 1)
        fwd = self.lp[idx + H] - self.lp[idx]
        n_eff = len(idx) / H
        out = {"n_overlapping_windows": int(len(idx)), "approx_independent_windows": round(n_eff, 1),
               "base_rate_up_6m_pct": round(float((fwd > 0).mean() * 100), 1),
               "mean_6m_ret_pct": round(float((np.exp(fwd) - 1).mean() * 100), 1)}
        for c in ("mom", "trend", "dd"):
            x = self.F[c].values[idx]
            ok = ~np.isnan(x)
            out[f"corr_{c}_vs_fwd6m"] = round(float(np.corrcoef(x[ok], fwd[ok])[0, 1]), 2)
        out["corr_needed_for_significance(~2/sqrt(n_eff))"] = round(2 / np.sqrt(n_eff), 2)
        r2 = (self.r[1:o + 1]) ** 2
        ac = lambda k: float(np.corrcoef(r2[:-k], r2[k:])[0, 1])
        out["ACF_of_squared_returns_lag1/5/21"] = [round(ac(1), 2), round(ac(5), 2), round(ac(21), 2)]
        # is the current 63d vol informative about the next 6m realised vol?
        v63 = pd.Series(self.r).rolling(63).std().values
        vf = np.array([np.nanstd(self.r[i + 1:i + H + 1]) for i in idx])
        out["corr_vol63_vs_next6m_vol"] = round(float(np.corrcoef(v63[idx], vf)[0, 1]), 2)
        return out
