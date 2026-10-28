import numpy as np

from operations.inverse import inverse_matrix
from operations.transpose import transpose_matrix
from operations.pseudo_inverse import pseudo_inverse 
from operations.least_squares import least_squares


def main():
    # 1) Transposta
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])

    print("Matrix \n", A)
    # 1) Transposta
    print("Transposta:\n", transpose_matrix(A))
    print("Confere com A.T?", np.array_equal(transpose_matrix(A), A.T))

    # 2) Inversa
    B = np.array([[4, 7],
                  [2, 6]])
    B_inv = inverse_matrix(B)
    print("\nInversa:\n", B_inv)
    print("B @ B_inv é a identidade?", np.allclose(B @ B_inv, np.eye(2)))

    # 3) Pseudoinversa
    x = np.array([1, 2, 3, 4, 5])
    X = np.column_stack([np.ones_like(x), x])  # coluna de 1s = constante
    print("\nPseudoinversa:\n", pseudo_inverse(X))
    print("Confere com np.linalg.pinv?", np.allclose(pseudo_inverse(X), np.linalg.pinv(X)))

    # 4) Mínimos quadrados: dados gerados a partir de y = 1 + 2x
    y = 1 + 2 * x
    a, b = least_squares(X, y)
    print(f"\nAjuste exato: y = {a:.4f} + {b:.4f}x  (esperado: 1 + 2x)")

    # Agora com ruído
    rng = np.random.default_rng(42)
    y_ruido = y + rng.normal(0, 0.3, size=x.size)
    beta = least_squares(X, y_ruido)
    beta_np = np.linalg.lstsq(X, y_ruido, rcond=None)[0]
    print(f"Ajuste com ruído: y = {beta[0]:.4f} + {beta[1]:.4f}x")
    print("Confere com np.linalg.lstsq?", np.allclose(beta, beta_np))


if __name__ == "__main__":
    main()