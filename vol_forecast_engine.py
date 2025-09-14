#!/usr/bin/env python3
# vol_forecast_engine.py
# Note: quick-and-dirty research driver; not production infra.
# GARCH(1,1) + ML (RF/GBRT) rolling 1-day-ahead variance forecasts.

import argparse
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd

# Optional econometrics (arch) 
try:
    from arch import arch_model
    try:
        from arch.univariate.base import DataScaleWarning, StartingValueWarning, ConvergenceWarning
    except Exception:
        from arch.univariate.base import DataScaleWarning, StartingValueWarning
        class ConvergenceWarning(UserWarning):
            pass
    _HAVE_ARCH = True
except Exception:
    _HAVE_ARCH = False

# Classic ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def read_returns(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "daily_return" not in df.columns:
        if "price" not in df.columns:
            raise ValueError("Need 'daily_return' or 'price' to compute returns.")
        prev = df["price"].shift(1)
        df["daily_return"] = (df["price"] - prev) / prev

    mask = (df["date"] >= "2018-01-01") & (df["date"] <= "2025-12-31")
    df = df.loc[mask].dropna(subset=["daily_return"]).reset_index(drop=True)

    if "vol" not in df.columns and "volume" in df.columns:
        df["vol"] = df["volume"]

    df.set_index("date", inplace=True)
    return df


def realized_variance(r: pd.Series) -> pd.Series:
    return r.pow(2)


# GARCH(1,1) rolling -

def _fit_arch(am, last_params):
    try:
        if last_params is not None:
            return am.fit(
                update_freq=0,
                disp="off",
                starting_values=last_params,
                show_warning=False,
                options={"maxiter": 2000},
            )
        return am.fit(
            update_freq=0,
            disp="off",
            show_warning=False,
            options={"maxiter": 2000},
        )
    except Exception:
        return am.fit(update_freq=0, disp="off", show_warning=False)


def rolling_garch_forecast(
    returns: pd.Series,
    dist: str = "normal",
    start_idx: int = 350,
    refit_every: int = 40,
) -> pd.Series:
    if not _HAVE_ARCH:
        raise RuntimeError("arch is not installed. pip install arch")

    SCALE = 100.0  # pre-scale returns; disable arch internal rescale
    r = returns.astype(float).to_numpy()
    n = len(r)
    out = np.full(n, np.nan, dtype=float)
    last_params = None

    i = start_idx
    while i < n:
        train = r[:i] * SCALE
        am = arch_model(train, mean="constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)

        if (i - start_idx) % refit_every == 0 or last_params is None:
            res = _fit_arch(am, None)
            last_params = res.params
        else:
            res = _fit_arch(am, last_params)

        v_scaled = res.forecast(horizon=1).variance.values[-1, 0]
        out[i] = v_scaled / (SCALE ** 2)
        i += 1

    return pd.Series(out, index=returns.index, name="garch_n")


# ML features 

def build_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    r = df["daily_return"]

    for k in range(1, lags + 1):
        X[f"r_lag_{k}"] = r.shift(k)

    for k in (5, 10, 20):
        X[f"ma_{k}"] = r.rolling(k).mean().shift(1)

    for k in (5, 10, 20):
        X[f"vol_{k}"] = r.rolling(k).std(ddof=0).shift(1)

    if "vol" in df.columns:
        X["vol_raw"] = df["vol"].shift(1)
        X["vol_roll_z"] = (
            df["vol"].shift(1) - df["vol"].rolling(20).mean().shift(1)
        ) / (df["vol"].rolling(20).std(ddof=0).shift(1) + 1e-12)

    y = realized_variance(r).shift(-1).rename("y")

    return pd.concat([y, X], axis=1).dropna()


# Rolling ML (RF / GBRT) 

def rolling_ml_forecast(
    df_fx: pd.DataFrame,
    model_name: str = "rf",
    start_idx: int = 380,
    refit_every: int = 30,
    n_estimators: int = 400,
    max_depth: Optional[int] = None,
    lr: float = 0.05,
) -> pd.Series:
    X = df_fx.drop(columns=["y"]).to_numpy(dtype=float)
    y = df_fx["y"].to_numpy(dtype=float)
    n = len(df_fx)
    preds = np.full(n, np.nan, dtype=float)

    if model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=-1
        )
    elif model_name == "gbrt":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbrt", GradientBoostingRegressor(random_state=0, learning_rate=lr)),
        ])
    else:
        raise ValueError("model_name must be 'rf' or 'gbrt'")

    last_fit = None
    i = start_idx
    while i < n:
        if (last_fit is None) or ((i - start_idx) % refit_every == 0):
            model.fit(X[:i], y[:i])
            last_fit = i
        preds[i] = float(model.predict(X[i:i+1])[0])
        i += 1

    return pd.Series(preds, index=df_fx.index, name=model_name)


# Metrics 

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def _qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    f = np.clip(y_pred, eps, None)
    y = np.clip(y_true, eps, None)
    return float(np.mean(np.log(f) + y / f))

def evaluate_and_print_scores(forecasts: pd.DataFrame) -> None:
    df = forecasts.copy()
    df["naive"] = df["rv_next"].shift(1)

    cols = ["rv_next", "naive"] + [c for c in ["garch_n", "rf", "gbrt"] if c in df.columns]
    df = df.dropna(subset=cols)

    y = df["rv_next"].to_numpy()
    scores = {}
    for c in cols[1:]:
        f = df[c].to_numpy()
        scores[c] = {"MSE": _mse(y, f), "QLIKE": _qlike(y, f)}

    def fmt(x): 
        return f"{x:.4g}" if x > 1e-4 else f"{x:.2e}"

    print("\n=== Evaluation vs Realised Variance (lower is better) ===")
    print(f"{'Model':<12} {'MSE':>14} {'QLIKE':>14}")
    for name, sc in scores.items():
        print(f"{name:<12} {fmt(sc['MSE']):>14} {fmt(sc['QLIKE']):>14}")

    if "naive" in scores:
        mse_b = scores["naive"]["MSE"]
        ql_b  = scores["naive"]["QLIKE"]
        print("\n=== Improvement vs Naïve (positive % = better) ===")
        print(f"{'Model':<12} {'ΔMSE%':>10} {'ΔQLIKE%':>10}")
        for name, sc in scores.items():
            if name == "naive":
                continue
            d_mse = 100.0 * (mse_b - sc["MSE"]) / mse_b
            d_ql  = 100.0 * (ql_b  - sc["QLIKE"]) / ql_b
            print(f"{name:<12} {d_mse:>9.2f}% {d_ql:>10.2f}%")


# Main flow 

def main():
    # mirror mc_var_engine.py: argparse + defaults
    ap = argparse.ArgumentParser(description="Rolling vol forecasts: GARCH(1,1) + ML (RF/GBRT)")
    ap.add_argument("--csv", type=Path, default=Path(r"C:\Users\Haise\Downloads\FTSE_100_Cleaned.csv"))
    ap.add_argument("--out", type=Path, default=Path("vol_forecasts.csv"))
    ap.add_argument("--garch_start", type=int, default=350)
    ap.add_argument("--garch_refit", type=int, default=40)
    ap.add_argument("--ml_start", type=int, default=380)
    ap.add_argument("--ml_refit", type=int, default=30)
    ap.add_argument("--rf_trees", type=int, default=400)
    ap.add_argument("--rf_depth", type=int, default=None)
    ap.add_argument("--gbrt_lr", type=float, default=0.05)
    args = ap.parse_args()

    # quiet arch warnings if available
    try:
        warnings.filterwarnings("ignore", category=DataScaleWarning)
        warnings.filterwarnings("ignore", category=StartingValueWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass

    # load & prep
    df = read_returns(args.csv)
    rv_next = realized_variance(df["daily_return"]).shift(-1).rename("rv_next")

    results = {}

    # GARCH(1,1)
    if _HAVE_ARCH:
        print("Fitting GARCH(1,1) [normal] ...")
        results["garch_n"] = rolling_garch_forecast(
            df["daily_return"],
            dist="normal",
            start_idx=args.garch_start,
            refit_every=args.garch_refit,
        )
    else:
        print("WARNING: 'arch' not installed — skipping GARCH. pip install arch")

    # ML features + rolling forecasts
    fx = build_features(df, lags=5)

    print("Training Random Forest (rolling) ...")
    results["rf"] = rolling_ml_forecast(
        fx,
        model_name="rf",
        start_idx=args.ml_start,
        refit_every=args.ml_refit,
        n_estimators=args.rf_trees,
        max_depth=args.rf_depth,
    )

    print("Training Gradient Boosting (rolling) ...")
    results["gbrt"] = rolling_ml_forecast(
        fx,
        model_name="gbrt",
        start_idx=args.ml_start,
        refit_every=args.ml_refit,
        lr=args.gbrt_lr,
    )

    # save
    out = pd.concat([rv_next] + [s.rename(k) for k, s in results.items()], axis=1)
    out.to_csv(args.out, index=True)
    print(f"Wrote forecasts → {args.out.resolve()}")

    # metrics
    evaluate_and_print_scores(out)

if __name__ == "__main__":
    main()
