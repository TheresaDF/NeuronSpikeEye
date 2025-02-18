from src.model.robust_ntf.robust_ntf import robust_ntf
import numpy as np 

def clean_scalograms(scalograms: np.ndarray) -> np.ndarray:
    # Define parameters 
    rank = 91
    beta = 1
    reg_val = 30
    tol = 1e-3

    # Run robust NTF
    factors, _, _ = robust_ntf(scalograms, rank=rank, beta=beta, reg_val=reg_val, tol=tol)

    # Reconstruct
    rntf_recon = np.zeros(scalograms.shape)
    for i in range(rank):
        rntf_recon += np.outer(factors[0][:, i], np.outer(factors[1][:, i], factors[2][:, i]))

    return rntf_recon
