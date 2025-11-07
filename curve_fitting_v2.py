"""
Parametric Curve Fitting — VERSION 2 (Research Style, Enhanced)
Author: V DIVYA
Date: November 2025

Features:
- New variable naming (phi, growth_rate, x_offset)
- Dual optimizer support (Differential Evolution or Powell)
- Extra diagnostics (MAE, correlation)
- Robust CSV parsing and saved outputs (Kaggle-ready)
- Saves fit plot, residual plot and results text to /kaggle/working
Usage:
    python curve_fitting_v2.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "/kaggle/input/xy-data/xy_data (1).csv"   # change if needed
OUT_DIR = "/kaggle/working"                          # Kaggle working dir (or set local)
os.makedirs(OUT_DIR, exist_ok=True)

# Choose method: "de" for Differential Evolution (global), "powell" for local Powell
DEFAULT_METHOD = "de"
DE_SEED = 99
DE_MAXITER = 500
DE_POPSIZE = 10

# -----------------------
# DATA LOADING (robust)
# -----------------------
if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found at {CSV_PATH}")
    sys.exit(1)

# try read with/without header, keep first two columns
try:
    df = pd.read_csv(CSV_PATH)
    if df.shape[1] == 1:
        df = pd.read_csv(CSV_PATH, header=None)
except Exception:
    df = pd.read_csv(CSV_PATH, header=None)

if df.shape[1] >= 2:
    df = df.iloc[:, :2]
df.columns = ["x", "y"]

# coerce to numeric and drop invalid rows
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna().reset_index(drop=True)

obs_x = df["x"].values.astype(float)
obs_y = df["y"].values.astype(float)
n_points = len(obs_x)

print(f"Loaded {n_points} valid points. X range: [{obs_x.min():.3f}, {obs_x.max():.3f}]  Y range: [{obs_y.min():.3f}, {obs_y.max():.3f}]")

# -----------------------
# MODEL & LOSS
# -----------------------
class ResearchEstimator:
    """
    phi         -> angular orientation (radians)
    growth_rate -> exponential coefficient (M)
    x_offset    -> horizontal translation (X)
    tau         -> parameter array (6..60)
    """

    def __init__(self, x_vals, y_vals, t_start=6.0, t_end=60.0):
        self.x = np.array(x_vals, dtype=float)
        self.y = np.array(y_vals, dtype=float)
        self.tau = np.linspace(t_start, t_end, len(self.x))
        self.phi = None
        self.growth_rate = None
        self.x_offset = None
        self.final_loss = None

    def model_curve(self, phi, growth_rate, x_offset):
        t = self.tau
        x_pred = t * np.cos(phi) - np.exp(growth_rate * np.abs(t)) * np.sin(0.3 * t) * np.sin(phi) + x_offset
        y_pred = 42.0 + t * np.sin(phi) + np.exp(growth_rate * np.abs(t)) * np.sin(0.3 * t) * np.cos(phi)
        return x_pred, y_pred

    def l1_error(self, params):
        phi, growth_rate, x_offset = params
        xp, yp = self.model_curve(phi, growth_rate, x_offset)
        return np.sum(np.abs(xp - self.x)) + np.sum(np.abs(yp - self.y))

    def fit(self, method=DEFAULT_METHOD, seed=DE_SEED, de_maxiter=DE_MAXITER, de_popsize=DE_POPSIZE):
        bounds = [
            (np.radians(0.0), np.radians(50.0)),  # phi in radians
            (-0.05, 0.05),                         # growth_rate (M)
            (0.0, 100.0)                           # x_offset (X)
        ]

        if method == "de":
            print("Running Differential Evolution (global) ...")
            res_de = differential_evolution(
                func=lambda p: self.l1_error(p),
                bounds=bounds,
                seed=seed,
                maxiter=de_maxiter,
                popsize=de_popsize,
                tol=1e-6,
                strategy='best1bin'
            )
            x0 = res_de.x
            print(f"DE finished. Local refine with Powell ...")
            res_local = minimize(lambda p: self.l1_error(p), x0=x0, method="Powell", bounds=bounds,
                                 options={'xtol':1e-8, 'ftol':1e-8, 'maxiter':1000})
            res = res_local if res_local.success else res_de
        elif method == "powell":
            print("Running Powell local optimizer from default initial guess ...")
            res = minimize(lambda p: self.l1_error(p), x0=[np.radians(25), 0.0, 50.0],
                           method="Powell", bounds=bounds, options={'xtol':1e-8, 'ftol':1e-8, 'maxiter':1000})
        else:
            raise ValueError("Unknown method: choose 'de' or 'powell'")

        self.phi, self.growth_rate, self.x_offset = res.x
        self.final_loss = float(res.fun)
        return res

    def diagnostics(self):
        xp, yp = self.model_curve(self.phi, self.growth_rate, self.x_offset)
        res_x = xp - self.x
        res_y = yp - self.y
        mae = (np.mean(np.abs(res_x)) + np.mean(np.abs(res_y))) / 2.0
        corr_x = np.corrcoef(self.x, xp)[0, 1]
        return {"mae": mae, "corr_x": corr_x, "res_x": res_x, "res_y": res_y, "xp": xp, "yp": yp}

    def save_results(self, out_dir=OUT_DIR, tag="v5"):
        stats = self.diagnostics()
        xp, yp = stats["xp"], stats["yp"]

        # print summary
        print("\n" + "="*60)
        print("FINAL RESULTS (Version 5 - Research)")
        print("="*60)
        print(f"phi (radians): {self.phi:.8f}")
        print(f"phi (degrees): {np.degrees(self.phi):.6f}°")
        print(f"growth_rate (M): {self.growth_rate:.8f}")
        print(f"x_offset (X): {self.x_offset:.6f}")
        print(f"L1 Error: {self.final_loss:.6f}")
        print(f"Mean Abs Error: {stats['mae']:.6f}")
        print(f"Correlation (x vs x_pred): {stats['corr_x']:.6f}")
        print("="*60)

        # Desmos string
        desmos = (f"\\left(t*\\cos({self.phi:.6f})-e^{{{self.growth_rate:.6f}\\left|t\\right|}}"
                  f"\\cdot\\sin(0.3t)\\sin({self.phi:.6f})+{self.x_offset:.4f},"
                  f"42+t*\\sin({self.phi:.6f})+e^{{{self.growth_rate:.6f}\\left|t\\right|}}"
                  f"\\cdot\\sin(0.3t)\\cos({self.phi:.6f})\\right)")

        # save text
        out_txt = os.path.join(out_dir, f"results_{tag}.txt")
        with open(out_txt, "w") as f:
            f.write("Version 5 - Parametric Curve Estimation (Research)\n")
            f.write("="*60 + "\n")
            f.write(f"phi (radians): {self.phi:.8f}\n")
            f.write(f"phi (degrees): {np.degrees(self.phi):.6f}\n")
            f.write(f"growth_rate (M): {self.growth_rate:.8f}\n")
            f.write(f"x_offset (X): {self.x_offset:.8f}\n")
            f.write(f"L1 Error: {self.final_loss:.6f}\n")
            f.write(f"MAE: {stats['mae']:.6f}\n")
            f.write(f"Correlation (x vs x_pred): {stats['corr_x']:.6f}\n\n")
            f.write("Desmos Format:\n" + desmos + "\n")

        # save fit plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.x, self.y, s=1.5, alpha=0.6, label="Observed Data")
        plt.plot(xp, yp, 'r-', linewidth=1.5, label=f"Fitted Curve ({tag})")
        plt.axis('equal'); plt.grid(True)
        plt.xlabel("X"); plt.ylabel("Y"); plt.title(f"Parametric Curve Fit - {tag}")
        plt.legend(); plt.tight_layout()
        out_png = os.path.join(out_dir, f"fitting_results_{tag}.png")
        plt.savefig(out_png, dpi=300)
        plt.show()

        # residuals plot
        t = self.tau
        plt.figure(figsize=(10, 4))
        plt.plot(t, np.abs(stats['res_x']), '.', ms=2, label='|res_x|')
        plt.plot(t, np.abs(stats['res_y']), '.', ms=2, label='|res_y|')
        plt.xlabel("t"); plt.ylabel("Absolute residual"); plt.title(f"Residuals - {tag}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        out_res = os.path.join(out_dir, f"residuals_{tag}.png")
        plt.savefig(out_res, dpi=300)
        plt.show()

        print(f"Saved: {out_txt}, {out_png}, {out_res}")
        print("\nDesmos string (paste into Desmos; domain 6 ≤ t ≤ 60):")
        print(desmos)
        return {"results_file": out_txt, "fit_png": out_png, "residual_png": out_res, "desmos": desmos}

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    estimator = ResearchEstimator(obs_x, obs_y)
    # fit with default method (change to "powell" to run Powell directly)
    res = estimator.fit(method=DEFAULT_METHOD, seed=DE_SEED, de_maxiter=DE_MAXITER, de_popsize=DE_POPSIZE)
    estimator.save_results(tag="v5")
