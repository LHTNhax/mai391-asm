
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ============================================================
# DATA PREPARATION
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select first two features
X = X_scaled[:, :2]

N, D = X.shape

print("=" * 60)
print("DATA PREPARATION")
print("=" * 60)

print("Data shape:", X.shape)
print("Selected features:", iris.feature_names[:2])
print("Number of samples:", N)


# ============================================================
# C1. ANALYTICAL GRADIENT
# ============================================================

print("\n" + "=" * 60)
print("C1. ANALYTICAL GRADIENT")
print("=" * 60)

def loss_function(w):
    z = X @ w
    e = z - y
    return np.mean(e ** 2)

def analytical_gradient(w):
    return (2 / N) * X.T @ (X @ w - y)

w = np.array([0.2, -0.3])

L = loss_function(w)
grad = analytical_gradient(w)

print("Weight vector w:", w)
print("Loss L(w):", L)
print("Gradient:")
print(grad)
print("Gradient row vector:")
print(grad.reshape(1, -1))


# ============================================================
# C2. HESSIAN AND CONVEXITY
# ============================================================

print("\n" + "=" * 60)
print("C2. HESSIAN AND CONVEXITY")
print("=" * 60)

XTX = X.T @ X
H = (2 / N) * XTX

eigenvalues = np.linalg.eigvalsh(H)

print("X.T @ X:")
print(XTX)

print("\nHessian matrix:")
print(H)

print("\nEigenvalues:")
print(eigenvalues)

if np.all(eigenvalues > 0):
    print("\nThe Hessian is positive definite.")
    print("Therefore, the loss function is strictly convex.")
else:
    print("\nThe Hessian is not positive definite.")


# ============================================================
# C3. COMPUTATIONAL GRAPH AND REVERSE MODE
# ============================================================

print("\n" + "=" * 60)
print("C3. COMPUTATIONAL GRAPH AND REVERSE MODE")
print("=" * 60)

indices = [0, 50]

X_C3 = X[indices]
y_C3 = y[indices]

w_C3 = np.array([0.2, -0.3])

print("X_C3:")
print(X_C3)

print("y_C3:")
print(y_C3)

print("w_C3:")
print(w_C3)


# ----------------------------
# Forward Pass
# ----------------------------

z = X_C3 @ w_C3
e = z - y_C3
q = e ** 2

L_C3 = np.mean(q)

print("\n--- Forward Pass ---")

print("z:")
print(z)

print("e:")
print(e)

print("q:")
print(q)

print("Loss:")
print(L_C3)


# ----------------------------
# Reverse Mode
# ----------------------------

dL_dq = np.ones_like(q) / len(q)

dq_de = 2 * e

dL_dz = dL_dq * dq_de

grad_reverse = X_C3.T @ dL_dz

grad_analytical = (
    2 / len(y_C3)
) * X_C3.T @ (X_C3 @ w_C3 - y_C3)

print("\n--- Reverse Mode ---")

print("dL/dq:")
print(dL_dq)

print("dq/de:")
print(dq_de)

print("dL/dz:")
print(dL_dz)

print("Reverse Mode Gradient:")
print(grad_reverse)

print("\nAnalytical Gradient:")
print(grad_analytical)

print("Reverse Mode Gradient:")
print(grad_reverse)

error = np.linalg.norm(
    grad_reverse - grad_analytical
)

print("L2 Error:")
print(error)


# ----------------------------
# Computational Graph
# ----------------------------

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_title(
    "Computational Graph - C3",
    fontsize=16
)

ax.axis("off")

nodes = {
    "w1": (0.1, 0.85),
    "w2": (0.1, 0.65),
    "z1": (0.35, 0.75),
    "z2": (0.35, 0.45),
    "e1": (0.55, 0.75),
    "e2": (0.55, 0.45),
    "q1": (0.75, 0.75),
    "q2": (0.75, 0.45),
    "L": (0.9, 0.6)
}

for name, position in nodes.items():
    ax.text(
        position[0],
        position[1],
        name,
        fontsize=12,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round",
            facecolor="lightblue",
            edgecolor="black"
        )
    )

edges = [
    ("w1", "z1"),
    ("w2", "z1"),
    ("w1", "z2"),
    ("w2", "z2"),
    ("z1", "e1"),
    ("z2", "e2"),
    ("e1", "q1"),
    ("e2", "q2"),
    ("q1", "L"),
    ("q2", "L")
]

for start, end in edges:
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color="black"
        )
    )

plt.subplots_adjust(
    left=0.12,
    right=0.95,
    top=0.90,
    bottom=0.12
)

plt.savefig(
    "Computational_Graph_C3.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nComputational Graph saved as:")
print("Computational_Graph_C3.png")


# ============================================================
# C4. AUTOMATIC DIFFERENTIATION
# ============================================================

print("\n" + "=" * 60)
print("C4. AUTOMATIC DIFFERENTIATION")
print("=" * 60)

X_torch = torch.tensor(
    X,
    dtype=torch.float64
)

y_torch = torch.tensor(
    y,
    dtype=torch.float64
)

weight_points = [
    np.array([0.0, 0.0]),
    np.array([0.5, -0.5]),
    np.array([-1.0, 1.0])
]

for w_np in weight_points:

    w_torch = torch.tensor(
        w_np,
        dtype=torch.float64,
        requires_grad=True
    )

    z_torch = X_torch @ w_torch

    loss_torch = torch.mean(
        (z_torch - y_torch) ** 2
    )

    loss_torch.backward()

    grad_autograd = w_torch.grad.detach().numpy()

    grad_analytical = analytical_gradient(w_np)

    abs_error = np.abs(
        grad_autograd - grad_analytical
    )

    max_error = np.max(abs_error)

    print("\nWeight point:", w_np)

    print("Loss:")
    print(loss_torch.item())

    print("Autograd Gradient:")
    print(grad_autograd)

    print("Analytical Gradient:")
    print(grad_analytical)

    print("Absolute Error:")
    print(abs_error)

    print("Maximum Absolute Error:")
    print(max_error)


# ============================================================
# C5. SECOND-ORDER TAYLOR EXPANSION
# ============================================================

print("\n" + "=" * 60)
print("C5. SECOND-ORDER TAYLOR EXPANSION")
print("=" * 60)

w0 = np.zeros(D)

L0 = loss_function(w0)

grad0 = analytical_gradient(w0)

H0 = H

print("L(0,0):")
print(L0)

print("\nGradient at origin:")
print(grad0)

print("\nHessian at origin:")
print(H0)


def taylor_approximation(w):
    delta = w - w0

    return (
        L0
        + grad0 @ delta
        + 0.5 * delta.T @ H0 @ delta
    )


test_points = [
    np.array([0.2, -0.3]),
    np.array([0.5, -0.5]),
    np.array([-1.0, 1.0])
]

for w_test in test_points:

    actual_loss = loss_function(w_test)

    approx_loss = taylor_approximation(w_test)

    abs_error = abs(
        actual_loss - approx_loss
    )

    print("\nTest weight:")
    print(w_test)

    print("Actual Loss:")
    print(actual_loss)

    print("Taylor Approximation:")
    print(approx_loss)

    print("Absolute Error:")
    print(abs_error)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("PART C1-C5 COMPLETED")
print("=" * 60)
