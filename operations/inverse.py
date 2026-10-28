import numpy as np

def inverse_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A matriz precisa ser quadrada.")
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("A matriz é singular (não tem inversa).")