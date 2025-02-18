import numpy as np

def beta_divergence(mat1, mat2, beta):
    """Compute the beta divergence between two matrices."""
    eps = np.finfo(mat1.dtype).eps  # Machine epsilon for numerical stability
    
    vec = lambda X: X.flatten()
    
    if beta == 2:
        return 0.5 * np.linalg.norm(mat1 - mat2, 'fro')**2
    elif beta == 1:
        mask = mat1 > eps
        return (np.sum(mat1[mask] * np.log(mat1[mask] / mat2[mask]) - mat1[mask] + mat2[mask])
                + np.sum(mat2[~mask]))
    elif beta == 0:
        return np.sum(vec(mat1) / vec(mat2) - np.log(vec(mat1) / vec(mat2))) - len(vec(mat1))
    else:
        return np.sum(vec(mat1)**beta + (beta-1) * vec(mat2)**beta - beta * vec(mat1) * vec(mat2)**(beta-1)) / (beta*(beta-1))

def L21_norm(mat):
    """Compute the L_{2,1} norm of a matrix."""
    return np.sum(np.sqrt(np.sum(mat**2, axis=0)))

def kr_bcd(matrices, skip_idx):
    """Khatri-Rao product in block coordinate descent."""
    matrices = [mat for i, mat in enumerate(matrices) if i != skip_idx]
    return khatri_rao_product(matrices[::-1])

def khatri_rao_product(matrices):
    """Khatri-Rao product of a list of matrices."""
    if len(matrices) < 2:
        raise ValueError("At least two matrices are required.")
    
    n_col = matrices[0].shape[1]
    krp = matrices[0]
    
    for matrix in matrices[1:]:
        if krp.shape[1] != matrix.shape[1]:
            raise ValueError("All matrices must have the same number of columns.")
        
        krp = (krp[:, None, :] * matrix[None, :, :]).reshape(-1, n_col)
    
    return krp
