
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ============================================================
# DATA PREPARATION
# ============================================================

iris = load_iris()

X_raw = iris.data[:, :2]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X = X_scaled

# X = first feature
# Y = second feature

x = X[:, 0]
y = X[:, 1]

N = len(X)

print("=" * 60)
print("DATA PREPARATION - TASK B")
print("=" * 60)

print("Data shape:", X.shape)
print("Feature X:", iris.feature_names[0])
print("Feature Y:", iris.feature_names[1])
print("Number of samples:", N)


# ============================================================
# B1. MEAN VECTOR AND COVARIANCE MATRIX
# ============================================================

print("\n" + "=" * 60)
print("B1. MEAN VECTOR AND COVARIANCE MATRIX")
print("=" * 60)

mu = np.mean(X, axis=0)

# Assignment uses denominator N
Sigma = (1 / N) * (X - mu).T @ (X - mu)

print("Mean vector mu:")
print(mu)

print("\nCovariance matrix Sigma:")
print(Sigma)

print("\nCovariance calculated using denominator N.")


# ============================================================
# B2. MARGINAL DISTRIBUTION OF Y
# ============================================================

print("\n" + "=" * 60)
print("B2. MARGINAL DISTRIBUTION OF Y")
print("=" * 60)

mu_x = mu[0]
mu_y = mu[1]

sigma_xx = Sigma[0, 0]
sigma_yy = Sigma[1, 1]
sigma_xy = Sigma[0, 1]

print("Assume (X, Y) follows a bivariate Gaussian distribution.")

print("\nMarginal distribution of Y:")
print("Y ~ N(mu_y, sigma_yy)")

print("\nmu_y =", mu_y)
print("sigma_yy =", sigma_yy)


# ============================================================
# B3. CONDITIONAL DISTRIBUTION X | Y = y0
# ============================================================

print("\n" + "=" * 60)
print("B3. CONDITIONAL DISTRIBUTION X | Y = y0")
print("=" * 60)

# 75th percentile of Y
y0 = np.quantile(y, 0.75)

# Conditional mean
mu_cond = (
    mu_x
    + (sigma_xy / sigma_yy) * (y0 - mu_y)
)

# Conditional variance
var_cond = (
    sigma_xx
    - (sigma_xy ** 2 / sigma_yy)
)

print("75th percentile y0:")
print(y0)

print("\nConditional mean:")
print(mu_cond)

print("\nConditional variance:")
print(var_cond)

print("\nOriginal variance Sigma_xx:")
print(sigma_xx)

print("\nCheck conditional variance < Sigma_xx:")

if var_cond < sigma_xx:
    print("TRUE")
else:
    print("FALSE")

print("\nConditional distribution:")
print("X | Y = y0 ~ N(mu_cond, var_cond)")


# ============================================================
# B4. SCATTER PLOT AND GAUSSIAN CONTOUR
# ============================================================

print("\n" + "=" * 60)
print("B4. SCATTER PLOT AND GAUSSIAN CONTOUR")
print("=" * 60)

# Create grid
x_min, x_max = x.min() - 1, x.max() + 1
y_min, y_max = y.min() - 1, y.max() + 1

grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, 150),
    np.linspace(y_min, y_max, 150)
)

grid = np.stack(
    [grid_x, grid_y],
    axis=-1
)

# Gaussian density
Sigma_inv = np.linalg.inv(Sigma)

det_Sigma = np.linalg.det(Sigma)

diff = grid - mu

exponent = np.einsum(
    "...i,ij,...j->...",
    diff,
    Sigma_inv,
    diff
)

density = (
    1 / (2 * np.pi * np.sqrt(det_Sigma))
) * np.exp(-0.5 * exponent)

# Plot
plt.figure(figsize=(9, 7))

plt.scatter(
    x,
    y,
    alpha=0.6,
    label="Iris Data"
)

plt.contour(
    grid_x,
    grid_y,
    density,
    levels=8
)

plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")

plt.title(
    "Bivariate Gaussian Contours and Iris Data"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "Gaussian_Contour_B4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Gaussian contour saved as Gaussian_Contour_B4.png")


# ============================================================
# B5. MIXTURE OF TWO UNIVARIATE GAUSSIANS
# ============================================================

print("\n" + "=" * 60)
print("B5. MIXTURE OF TWO GAUSSIANS")
print("=" * 60)

# Choose mixture parameters
pi = 0.4

mu1 = 2.0
mu2 = 5.0

sigma1_sq = 1.0
sigma2_sq = 2.25

print("Mixture parameters:")

print("pi =", pi)
print("mu1 =", mu1)
print("sigma1^2 =", sigma1_sq)

print("mu2 =", mu2)
print("sigma2^2 =", sigma2_sq)


# ----------------------------
# Expected Value
# ----------------------------

E_X = (
    pi * mu1
    + (1 - pi) * mu2
)

print("\nExpected Value E[X]:")

print(
    "E[X] = pi*mu1 + (1-pi)*mu2"
)

print("E[X] =", E_X)


# ----------------------------
# Variance
# ----------------------------

Var_X = (
    pi * sigma1_sq
    + (1 - pi) * sigma2_sq
    + pi * (1 - pi) * (mu1 - mu2) ** 2
)

print("\nVariance V[X]:")

print("V[X] =", Var_X)


# ----------------------------
# Law of Total Variance
# ----------------------------

within_variance = (
    pi * sigma1_sq
    + (1 - pi) * sigma2_sq
)

between_variance = (
    pi * (1 - pi) * (mu1 - mu2) ** 2
)

total_variance = (
    within_variance
    + between_variance
)

print("\nLaw of Total Variance:")

print("Within-component variance:")
print(within_variance)

print("\nBetween-component variance:")
print(between_variance)

print("\nTotal variance:")
print(total_variance)

print("\nDirect variance:")
print(Var_X)

print("\nVerification:")

print(
    "Difference:",
    abs(Var_X - total_variance)
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("TASK B1-B5 COMPLETED")
print("=" * 60)
