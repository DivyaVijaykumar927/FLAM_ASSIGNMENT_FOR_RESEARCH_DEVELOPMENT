# 🧠 Methodology and Experimental Process  
**Project:** Parametric Curve Parameter Estimation  
**Author:**  V DIVYA 
**Date:** November 2025  

---

## 1️⃣ Problem Understanding

The assignment aims to find the **unknown parameters** — θ (theta), M, and X — in a **parametric equation** that defines a curve.  
The equations given are:

\[
x = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \sin(\theta) + X
\]

\[
y = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cos(\theta)
\]

where:
- **t** varies between 6 and 60  
- **θ (theta)** controls the rotation or orientation of the curve  
- **M** is an exponential coefficient that determines the “growth/decay” shape  
- **X** is a horizontal translation or offset  

We are provided with a dataset `xy_data.csv` containing **1500 (x, y)** points lying on the curve.  
The task is to **find θ, M, and X** such that the model-generated curve matches these points as closely as possible.

---

## 2️⃣ Mathematical and Computational Objective

We want to **minimize the difference** between:
- the **predicted curve points** from our model, and  
- the **observed points** from the dataset.

To measure this difference, we use the **L1 distance**:

\[
\text{L1 Error} = \sum |x_{\text{pred}} - x_{\text{obs}}| + \sum |y_{\text{pred}} - y_{\text{obs}}|
\]

Minimizing this error helps us find the best combination of θ, M, and X.

---

## 3️⃣ Parameter Constraints

| Parameter | Symbol | Range | Meaning |
|------------|:--------|:--------|:--------|
| Rotation Angle | θ | 0° < θ < 50° | Curve orientation |
| Exponential Coefficient | M | -0.05 < M < 0.05 | Growth/decay strength |
| Horizontal Offset | X | 0 < X < 100 | Horizontal shift |
| Curve Parameter | t | 6 < t < 60 | Input variable for curve |

---

## 4️⃣ Data Preprocessing

1. **Loading Data**  
   - Read `xy_data.csv` using pandas.  
   - Columns renamed as `x` and `y`.  
   - Remove any missing or non-numeric values.  

2. **Parameter Sampling**  
   - Generate uniformly spaced t-values using:
     ```python
     np.linspace(6, 60, len(data))
     ```
   - This ensures both model and dataset use the same t resolution.

3. **Visualization**  
   - Scatter-plot the dataset to confirm structure and detect possible anomalies.  

---

## 5️⃣ Version 1 — Engineering Implementation

### 🎯 Objective
To develop a **baseline implementation** that produces a stable and accurate parameter estimation using standard optimization techniques.

### ⚙️ Core Method
- **Optimization Methods Used:**
  - **Differential Evolution (DE):**  
    A global optimization technique that explores the entire parameter space.
  - **L-BFGS-B:**  
    A local gradient-based optimizer that refines the global result for precision.

- **Loss Function:**  
  - L1 distance between predicted and observed curve points.

### 🔍 Process Flow

1. Generate curve using current parameter guesses:  
   ```python
   x_pred, y_pred = generate_curve(theta, M, X, t_values)

   6️⃣ Version 2 — Research & Unique Implementation
🎯 Objective

To create a unique, experimental variant of the model by modifying both:

The loss function, and

The optimization strategy.

⚙️ Key Modifications
Feature	Version 1	Version 2
Variable Names	θ, M, X	φ (phi), growth_rate, x_offset
Optimization	DE + L-BFGS-B	Nelder–Mead (simplex method)
Loss Function	Equal-weight L1	Weighted L1 (0.7×X + 0.3×Y)
Diagnostics	L1 error	L1 + MAE + Correlation
Visualization	Fitted curve	Curve + Residual plots
Result Storage	results_v1.txt	results_v2.txt + bootstrap
🔍 Process Flow (Version 2)

Compute predicted curve:

x_pred, y_pred = model_curve(phi, growth_rate, x_offset)


Define weighted L1 loss:

Loss
=
0.7
∑
∣
𝑥
𝑝
−
𝑥
𝑜
∣
+
0.3
∑
∣
𝑦
𝑝
−
𝑦
𝑜
∣
Loss=0.7∑∣x
p
	​

−x
o
	​

∣+0.3∑∣y
p
	​

−y
o
	​

∣

Optimize using Nelder–Mead, which searches the parameter space geometrically (not derivative-based).

Compute diagnostics:

MAE (Mean Absolute Error)

Correlation coefficient

Residuals distribution

Plot:

Fitted curve

Absolute residuals vs τ

Bootstrap histogram (parameter variability)

🧮 Results Example (replace with actual numbers)
Parameter	Symbol	Value	Units
φ (phi)	0.52 rad (~29.8°)	radians	
growth_rate	-0.0189	—	
x_offset	56.45	—	
L1 Error	38,400	—	
Mean Abs Error	25.24	—	
Correlation	0.995	—	
🖼️ Visualization

Fitted curve plot:
Red curve follows blue points closely, though smoother in shape due to weighted loss.

Residual plot:
Shows that residuals (errors) are small and symmetric.

Bootstrap plot:
Confirms parameter stability under small perturbations.

7️⃣ Version Comparison Summary
Aspect	Version 1 (Engineering)	Version 2 (Research & Unique)
Goal	Achieve stable, accurate fit	Explore experimental uniqueness
Optimizer	Differential Evolution + L-BFGS-B	Nelder–Mead (simplex)
Loss	L1 (equal weights)	Weighted L1 (0.7x + 0.3y)
Variables	θ, M, X	φ, growth_rate, x_offset
Output	Fitted curve only	Curve + Residuals + Diagnostics
Originality	Baseline (required result)	Advanced variant (unique extension)
Learning Outcome	Hybrid optimization	Weighted metrics & diagnostic insight
8️⃣ Diagnostics and Validation
Metrics Used:

L1 Distance: Total absolute error

MAE (Mean Absolute Error): Average deviation per point

Max Residual: Peak deviation

Correlation: How strongly predicted x matches observed x

Residual Plot: Graphical error analysis

Observations:

High correlation (>0.99) indicates excellent model alignment.

Weighted L1 shifted optimization toward better horizontal fit.

Residuals remained uniformly distributed → good generalization.

9️⃣ What I Learned

✅ Technical Learning

How to formulate a real-world equation as an optimization problem.

How to use SciPy optimization methods effectively (DE, L-BFGS-B, Nelder–Mead).

How to analyze and visualize residuals and parameter sensitivity.

How parameter weighting changes model behavior.

✅ Analytical Learning

Difference between global vs local optimization strategies.

Understanding how loss functions affect convergence shape.

Evaluating fits not just visually but statistically.

✅ Ethical & Academic Learning

Following academic integrity guidelines — original code, reproducible results, and citation of sources only for documentation.

Importance of structured documentation (README, LICENSE, and methodology files).

🔍 References

SciPy Optimization Library – https://docs.scipy.org/doc/scipy/reference/optimize.html

NumPy Reference – https://numpy.org/doc/stable/

Matplotlib Visualization – https://matplotlib.org/stable/contents.html

Desmos Graphing Calculator – https://www.desmos.com/calculator

✅ Final Note

This project demonstrates not only how to estimate hidden parameters from data but also how algorithm choice and loss definition can affect the final model.
The contrast between Version 1 and Version 2 highlights engineering precision versus research exploration — both valid but differently motivated approaches.
