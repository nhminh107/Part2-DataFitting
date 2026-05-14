import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from hat_matrix import hat_matrix, verify_hat_properties, fitted_values_from_hat
from ols_fit import ols_fit

# ============================================================
# UNIT TESTS CHO HAT MATRIX
# ============================================================

def test_hat_idempotent():
    """Test 1: H² = H (idempotent)."""
    print("=" * 55)
    print("TEST 1: Idempotent — H² = H")
    print("=" * 55)
    np.random.seed(42)
    X = np.random.randn(20, 3)
    res = hat_matrix(X)
    H = res['H']
    err = np.linalg.norm(H @ H - H, 'fro')
    print(f"  ‖H² − H‖_F = {err:.2e}")
    assert err < 1e-10
    print("  ✅ PASSED\n")


def test_hat_symmetric():
    """Test 2: Hᵀ = H (symmetric)."""
    print("=" * 55)
    print("TEST 2: Symmetric — Hᵀ = H")
    print("=" * 55)
    np.random.seed(42)
    X = np.random.randn(30, 2)
    H = hat_matrix(X)['H']
    err = np.linalg.norm(H.T - H, 'fro')
    print(f"  ‖Hᵀ − H‖_F = {err:.2e}")
    assert err < 1e-10
    print("  ✅ PASSED\n")


def test_hat_eigenvalues():
    """Test 3: Eigenvalues chỉ là 0 hoặc 1."""
    print("=" * 55)
    print("TEST 3: Eigenvalues ∈ {0, 1}")
    print("=" * 55)
    np.random.seed(42)
    X = np.random.randn(25, 4)
    eigvals = hat_matrix(X)['eigenvalues']
    ok = all(np.isclose(e, 0, atol=1e-10) or np.isclose(e, 1, atol=1e-10)
             for e in eigvals)
    n_ones = int(np.sum(np.isclose(eigvals, 1, atol=1e-10)))
    n_zeros = int(np.sum(np.isclose(eigvals, 0, atol=1e-10)))
    print(f"  #(λ=1) = {n_ones},  #(λ=0) = {n_zeros},  total = {len(eigvals)}")
    assert ok
    print("  ✅ PASSED\n")


def test_hat_rank():
    """Test 4: rank(H) = p + 1."""
    print("=" * 55)
    print("TEST 4: rank(H) = p + 1")
    print("=" * 55)
    np.random.seed(42)
    p = 3
    X = np.random.randn(40, p)
    res = hat_matrix(X)
    print(f"  rank(H) = {res['rank']},  expected = {p + 1}")
    assert res['rank'] == p + 1
    print("  ✅ PASSED\n")


def test_hat_fitted_values():
    """Test 5: Hy = Xβ̂ (fitted values khớp OLS)."""
    print("=" * 55)
    print("TEST 5: ŷ = Hy khớp với Xβ̂")
    print("=" * 55)
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = 1.0 + X @ np.array([2.0, -1.0]) + 0.5 * np.random.randn(50)

    H_res = hat_matrix(X)
    y_hat_H = H_res['H'] @ y

    ols_res = ols_fit(X, y)
    y_hat_ols = ols_res['y_hat']

    err = np.linalg.norm(y_hat_H - y_hat_ols)
    print(f"  ‖Hy − Xβ̂‖ = {err:.2e}")
    assert err < 1e-10
    print("  ✅ PASSED\n")


# ============================================================
# DEMO
# ============================================================

def demo_hat_matrix():
    """
    Minh họa Hat Matrix:
    1. Tạo dữ liệu synthetic
    2. Kiểm tra tất cả tính chất
    3. Vẽ heatmap H
    4. Vẽ leverage plot
    """
    print("\n" + "=" * 55)
    print("DEMO: Hat Matrix trên dữ liệu giả lập")
    print("=" * 55)

    np.random.seed(42)
    n = 30
    X_raw = np.random.randn(n, 2)
    y = 1.0 + X_raw @ np.array([2.0, -1.5]) + np.random.randn(n)

    res = hat_matrix(X_raw)
    H = res['H']
    X_design = res['X_design']

    # Kiểm tra tính chất
    print("\n--- Kiểm tra 5 tính chất ---")
    props = verify_hat_properties(H, X_design, y)
    for name, info in props.items():
        status = "✅" if info['passed'] else "❌"
        print(f"  {status} {name}: {info['desc']}")

    # Thống kê leverage
    lev = res['leverage']
    avg_lev = np.mean(lev)
    k = X_design.shape[1]
    print(f"\n--- Leverage ---")
    print(f"  Trung bình hᵢᵢ = {avg_lev:.4f}  (lý thuyết: (p+1)/n = {k/n:.4f})")
    print(f"  Max hᵢᵢ = {np.max(lev):.4f} tại i = {np.argmax(lev)}")

    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(H, cmap='RdBu_r', vmin=-0.2, vmax=0.5, aspect='auto')
    ax.set_title(f'Hat Matrix H ({n}×{n})')
    ax.set_xlabel('j'); ax.set_ylabel('i')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Leverage plot
    ax = axes[1]
    ax.bar(range(n), lev, color='steelblue', edgecolor='k', linewidth=0.5)
    ax.axhline(2*k/n, color='r', ls='--', lw=1.5,
               label=f'Ngưỡng 2(p+1)/n = {2*k/n:.3f}')
    ax.set_xlabel('Quan sát i')
    ax.set_ylabel('Leverage hᵢᵢ')
    ax.set_title('Leverage Values')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_hat_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  📊 Đã lưu biểu đồ: demo_hat_matrix.png")

    # Kiểm tra ŷ = Hy
    fv = fitted_values_from_hat(H, y)
    ols_res = ols_fit(X_raw, y)
    err = np.linalg.norm(fv['y_hat'] - ols_res['y_hat'])
    print(f"\n  Kiểm tra ŷ = Hy vs Xβ̂: ‖diff‖ = {err:.2e}  ", end="")
    print("✅" if err < 1e-10 else "❌")


if __name__ == "__main__":
    print("╔" + "═"*53 + "╗")
    print("║  HAT MATRIX TEST — Ma trận chiếu (from scratch)   ║")
    print("╚" + "═"*53 + "╝\n")
    test_hat_idempotent()
    test_hat_symmetric()
    test_hat_eigenvalues()
    test_hat_rank()
    test_hat_fitted_values()
    print("🎉 Tất cả tests HAT MATRIX PASSED!\n")
    demo_hat_matrix()
    print("\n✅ Hoàn tất.")
