from .pseudo_inverse import pseudo_inverse

def least_squares(X, y):
    return pseudo_inverse(X) @ y