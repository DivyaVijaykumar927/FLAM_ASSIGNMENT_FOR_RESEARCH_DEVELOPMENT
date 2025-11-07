"""
Parametric Curve Fitting — VERSION 1 (Engineering style, Kaggle-ready)
Author: V DIVYA
Date: November 2025

Robust CSV parsing, Differential Evolution (global) + L-BFGS-B (local),
visualization, Desmos string, saved outputs to /kaggle/working.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import os
import sys

# ---------------------------
# CONFIG
# ---------------------------
CSV_PATH = "/kaggle/input/xy-data/xy_data (1).csv"   # change if needed
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# LOAD + CLEAN DATA (robust)
# ---------------------------
if not os.path.exists(CSV_PATH):
    print(f"❌ Data file not found at: {CSV_PATH}")
    sys.exit(1)

print(f"✅ Loading dataset from: {CSV_PATH}")

# Try to read; handle header/no-header and extra columns
try:
    data = pd.read_csv(CSV_PATH)
    if data.shape[1] == 1:
        data = pd.read_csv(CSV_PATH, header=None)
except Exception:
    data = pd.read_csv(CSV_PATH, header=None)

# Keep first two columns, rename
if data.shape[1] >= 2:
    data = data.iloc[:, :2]
data.columns = ['x', 'y']

# Convert to numeric and drop invalid rows
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data = data.dropna().reset_index(drop=True)

obs_x = data['x'].values.astype(float)
obs_y = data['y'].values.astype(float)
n_points = len(obs_x)

print(f"✅ Loaded {n_points} clean points.")
print(f"   X range: {obs_x.min():.3f} -> {obs_x.max():.3f}")
print(f"   Y range: {obs_y.min():.3f} -> {obs_y.max():.3f}")

# ---------------------------
# SpiralEstimator class (Version 4)
# ---------------------------
class SpiralEstimator:
    def __init__(self, x_vals, y_vals, t_start=6.0, t_end=60.0):
        self.xs = np.array(x_vals, dtype=float)
        self.ys = np.array(y_vals, dtype=float)
        self.n = len(self.xs)
        self.time_param = np.linspace(t_start, t_end, self.n)

    def compute_curve(self, rotation_rad, exp_coeff, x_shift):
        t = self.time_param
        x_pred = t * np.cos(rotation_rad) - np.exp(exp_coeff * np.abs(t)) * np.sin(0.3 * t) * np.sin(rotation_rad) + x_shift
        y_pred = 42.0 + t * np.sin(rotation_rad) + np.exp(exp_coeff * np.abs(t)) * np.sin(0.3 * t) * np.cos(rotation_rad)
        return x_pred, y_pred

    def l1_loss(self, params):
        rotation_rad, exp_coeff, x_shift = params
        px, py = self.compute_curve(rotation_rad, exp_coeff, x_shift)
        return np.sum(np.abs(px - self.xs)) + np.sum(np.abs(py - self.ys))

    def estimate(self, seed=101, de_maxiter=600, de_popsize=12):
        bounds = [
            (np.radians(0.0), np.radians(50.0)),  # rotation_rad
            (-0.05, 0.05),                         # exp_coeff (M)
            (0.0, 100.0)                           # x_shift (X)
        ]

        print("\n🚀 Running global optimization (Differential Evolution)...")
        res_global = differential_evolution(
            func=lambda p: self.l1_loss(p),
            bounds=bounds,
            strategy='best1bin',
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-7,
            seed=seed,
            disp=False
        )
        print("✅ Global optimization complete. Refining locally (L-BFGS-B)...")

        res_local = minimize(
            fun=lambda p: self.l1_loss(p),
            x0=res_global.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        self.rotation_rad, self.exp_coeff, self.x_shift = res_local.x
        self.final_loss = float(res_local.fun)
        return res_local

    def save_and_plot(self, out_dir=OUT_DIR, prefix="v4"):
        theta = self.rotation_rad
        M = self.exp_coeff
        X = self.x_shift

        # Desmos-ready string (radians)
        desmos_str = (
            f"\\left(t*\\cos({theta:.6f})-e^{{{M:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta:.6f})+{X:.4f},"
            f"42+t*\\sin({theta:.6f})+e^{{{M:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta:.6f})\\right)"
        )

        print("\n" + "="*65)
        print("🎯 FINAL PARAMETER ESTIMATION RESULTS (Version 4 – Engineering)")
        print("="*65)
        print(f"θ (radians): {theta:.8f}")
        print(f"θ (degrees): {np.degrees(theta):.6f}°")
        print(f"M (exp_coeff): {M:.8f}")
        print(f"X (x_shift): {X:.6f}")
        print(f"L1 Error: {self.final_loss:.6f}")
        print("="*65)
        print("\n📄 Desmos Submission Format:")
        print(desmos_str)

        # Save textual results
        out_txt = os.path.join(out_dir, f"results_{prefix}.txt")
        with open(out_txt, "w") as f:
            f.write("Version 4 — Parametric Curve Estimation (Engineering)\n")
            f.write("="*60 + "\n")
            f.write(f"θ (radians): {theta:.8f}\n")
            f.write(f"θ (degrees): {np.degrees(theta):.6f}\n")
            f.write(f"M: {M:.8f}\n")
            f.write(f"X: {X:.6f}\n")
            f.write(f"L1 Error: {self.final_loss:.6f}\n\n")
            f.write("Desmos Format:\n")
            f.write(desmos_str + "\n")

        # Plot fit vs observed
        fit_x, fit_y = self.compute_curve(theta, M, X)
        plt.figure(figsize=(10, 8))
        plt.scatter(self.xs, self.ys, s=1.8, alpha=0.6, label='Observed Data')
        plt.plot(fit_x, fit_y, '-', linewidth=1.5, color='tab:red', label='Fitted Curve (v4)')
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Parametric Curve Fitting – Version 4 (Engineering)")
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"fitting_results_{prefix}.png")
        plt.savefig(out_png, dpi=300)
        plt.show()

        print(f"\n✅ Saved: {out_txt} and {out_png}")

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    estimator = SpiralEstimator(obs_x, obs_y)
    estimator.estimate(seed=101)      # change seed or DE params if you want variation
    estimator.save_and_plot(prefix="v4")
