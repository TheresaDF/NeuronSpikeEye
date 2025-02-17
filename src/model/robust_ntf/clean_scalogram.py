from src.model.robust_ntf.robust_ntf import robust_ntf
from tensorly.tenalg import outer
import numpy as np 
import torch 

torch.set_default_tensor_type(torch.cuda.DoubleTensor)


def clean_scalograms(scalograms : np.ndarray) -> np.ndarray:
    # convert to torch tensor
    scalograms_torch = torch.tensor(scalograms).cuda()

    # define parameters 
    rank = 91 
    beta = 1
    reg_val = 30
    tol = 1e-3

    # run robust NTF
    factors, _, _ = robust_ntf(scalograms_torch, rank=rank, beta=beta, reg_val=reg_val, tol=tol)

    # reconstruct 
    rntf_recon = torch.zeros(scalograms.shape)
    for i in range(rank):
        rntf_recon = rntf_recon + outer([factors[0][:,i],
                                        factors[1][:,i],
                                        factors[2][:,i]])

    rntf_recon = rntf_recon.cpu().numpy()

    return rntf_recon