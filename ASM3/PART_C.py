
# ============================================================
# MAI391 - ASSIGNMENT 3
# TASK C: SUPPORT VECTOR MACHINE (SVM)
# C1 - C6
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# C1. BINARY CLASSIFICATION DATA PREPARATION
# ============================================================

print("=" * 60)
print("C1. BINARY CLASSIFICATION DATA PREPARATION")
print("=" * 60)

iris = load_iris()

# Select two features
X_raw = iris.data[:, :2]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Binary labels:
# Setosa = +1
# Non-setosa = -1

y = np.where(iris.target == 0, 1, -1)

print("Data shape:", X.shape)
print("Selected features:", iris.feature_names[:2])

print("Binary labels:", np.unique(y))

print("Positive samples:", np.sum(y == 1))
print("Negative samples:", np.sum(y == -1))


# ============================================================
# C2. MAXIMUM MARGIN HYPERPLANE - ANALYTICAL SOLUTION
# ============================================================

print("\n" + "=" * 60)
print("C2. MAXIMUM MARGIN HYPERPLANE")
print("=" * 60)

# Given data
x1 = np.array([1.0, 1.0])
y1 = 1

x2 = np.array([-1.0, -1.0])
y2 = -1

# Analytical solution
w_C2 = np.array([0.5, 0.5])
b_C2 = 0.0

# Norm of w
norm_w = np.linalg.norm(w_C2)

# Margin width
margin_width = 2 / norm_w

# Distance from boundary to support vector
margin_distance = 1 / norm_w

print("w =", w_C2)
print("b =", b_C2)

print("\nDecision boundary:")
print("0.5*x1 + 0.5*x2 = 0")

print("\nNorm of w:")
print(norm_w)

print("\nMargin distance:")
print(margin_distance)

print("\nFull margin width:")
print(margin_width)

print("\nSupport vectors:")
print("x1 =", x1, "with y1 =", y1)
print("x2 =", x2, "with y2 =", y2)

print("\nVerification:")

print(
    "y1*(w.x1+b) =",
    y1 * (w_C2 @ x1 + b_C2)
)

print(
    "y2*(w.x2+b) =",
    y2 * (w_C2 @ x2 + b_C2)
)


# ============================================================
# C3. GEOMETRICAL INTERPRETATION OF w AND b
# ============================================================

print("\n" + "=" * 60)
print("C3. GEOMETRICAL INTERPRETATION")
print("=" * 60)

print("""
The vector w is perpendicular to the decision boundary.

Changing the direction of w changes the orientation
of the decision boundary.

Changing the magnitude of w changes the margin width.

The bias b shifts the decision boundary without changing
its orientation.

The decision boundary is defined by:
w.T @ x + b = 0

The margin boundaries are:
w.T @ x + b = +1
w.T @ x + b = -1
""")


# ============================================================
# C4. LINEAR SVM WITH DIFFERENT C VALUES
# ============================================================

print("\n" + "=" * 60)
print("C4. LINEAR SVM WITH DIFFERENT C VALUES")
print("=" * 60)

C_values = [0.01, 1, 100]

models = {}

# Create plotting grid
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1

y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for ax, C in zip(axes, C_values):

    model = SVC(
        kernel="linear",
        C=C
    )

    model.fit(X, y)

    models[C] = model

    # Decision function
    decision_values = model.decision_function(grid)

    decision_values = decision_values.reshape(
        xx.shape
    )

    # Plot samples
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        marker="o",
        label="Positive class"
    )

    ax.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        marker="x",
        label="Negative class"
    )

    # Decision boundary and margins
    ax.contour(
        xx,
        yy,
        decision_values,
        levels=[-1, 0, 1],
        linestyles=["dashed", "solid", "dashed"]
    )

    # Support vectors
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="black",
        label="Support vectors"
    )

    ax.set_title(f"Linear SVM (C={C})")

    ax.set_xlabel("Sepal Length (Standardized)")
    ax.set_ylabel("Sepal Width (Standardized)")

    ax.legend()
    ax.grid(True)


plt.suptitle(
    "Linear SVM Decision Boundaries and Margins"
)

plt.tight_layout()

plt.savefig(
    "SVM_C4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----------------------------
# Evaluate margin violations
# ----------------------------

print("\nMargin violation analysis:")

print(
    "{:<10} {:<20} {:<20}".format(
        "C",
        "Support Vectors",
        "Margin Violations"
    )
)

for C in C_values:

    model = models[C]

    decision = model.decision_function(X)

    margin_values = y * decision

    violations = np.sum(
        margin_values < 1 - 1e-8
    )

    print(
        "{:<10} {:<20} {:<20}".format(
            C,
            len(model.support_vectors_),
            violations
        )
    )

print("\nC4 figure saved as SVM_C4.png")


# ============================================================
# C5. XOR - NON-LINEARLY SEPARABLE DATA
# ============================================================

print("\n" + "=" * 60)
print("C5. XOR NON-LINEARLY SEPARABLE DATA")
print("=" * 60)

# XOR dataset
X_XOR = np.array([
    [1, 1],
    [-1, -1],
    [1, -1],
    [-1, 1]
], dtype=float)

# Positive class: same signs
# Negative class: opposite signs

y_XOR = np.array([
    1,
    1,
    -1,
    -1
])

print("XOR data:")
print(X_XOR)

print("\nXOR labels:")
print(y_XOR)

print("""
In the original two-dimensional space,
the convex hulls of the two classes intersect
at the origin.

Therefore, no linear separating hyperplane exists.
""")


# ----------------------------
# Feature Mapping
# ----------------------------

def feature_mapping(x):

    x1 = x[:, 0]
    x2 = x[:, 1]

    return np.column_stack([
        x1,
        x2,
        x1 * x2
    ])


X_XOR_mapped = feature_mapping(X_XOR)

print("\nMapped XOR data:")
print(X_XOR_mapped)

# Train linear SVM in mapped space
xor_model = SVC(
    kernel="linear",
    C=100
)

xor_model.fit(
    X_XOR_mapped,
    y_XOR
)

predictions = xor_model.predict(
    X_XOR_mapped
)

print("\nPredictions:")
print(predictions)

print("\nActual labels:")
print(y_XOR)

print("\nTraining accuracy:")
print(
    np.mean(predictions == y_XOR)
)


# ----------------------------
# Visualize XOR
# ----------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    X_XOR[y_XOR == 1, 0],
    X_XOR[y_XOR == 1, 1],
    marker="o",
    s=100,
    label="Positive class"
)

plt.scatter(
    X_XOR[y_XOR == -1, 0],
    X_XOR[y_XOR == -1, 1],
    marker="x",
    s=100,
    label="Negative class"
)

plt.xlabel("x1")
plt.ylabel("x2")

plt.title("XOR Dataset - Nonlinear Separability")

plt.legend()
plt.grid(True)

plt.savefig(
    "XOR_C5.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# C6. SOFT-MARGIN SVM DUAL PROBLEM
# ============================================================

print("\n" + "=" * 60)
print("C6. SOFT-MARGIN SVM DUAL PROBLEM")
print("=" * 60)

print("""
Primal problem:

minimize:
    1/2 * ||w||^2 + C * sum(xi_i)

subject to:
    y_i * (w.T @ x_i + b) >= 1 - xi_i

    xi_i >= 0


Dual problem:

maximize:

    sum(alpha_i)
    - 1/2 * sum_i sum_j
      alpha_i * alpha_j * y_i * y_j
      * (x_i.T @ x_j)

subject to:

    0 <= alpha_i <= C

    sum(alpha_i * y_i) = 0


The data appear only through inner products
x_i.T @ x_j.

Therefore, inner products can be replaced
with a kernel function K(x_i, x_j).

This allows SVM to learn nonlinear decision boundaries
without explicitly computing the feature mapping.
""")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("TASK C1-C6 COMPLETED")
print("=" * 60)
