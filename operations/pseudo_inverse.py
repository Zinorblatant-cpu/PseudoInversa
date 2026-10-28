from .inverse import inverse_matrix
from .transpose import transpose_matrix

def pseudo_inverse(X):
    Xt = transpose_matrix(X)
    return inverse_matrix(Xt @ X) @ Xt