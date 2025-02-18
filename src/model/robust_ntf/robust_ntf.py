import numpy as np
import tensorly as tl
from .foldings import folder, unfolder
from .matrix_utils import kr_bcd, beta_divergence, L21_norm
tl.set_backend('numpy')


def robust_ntf(data, rank, beta, reg_val, tol, init='random', max_iter=1000,
               print_every=10, user_prov=None, verbose=False):
    """Robust Non-negative Tensor Factorization (rNTF)

    This function decomposes an input non-negative tensor into the sum of
    component rank-1 non-negative tensors (returned as a series of factor
    matrices), and a group-sparse non-negative outlier tensor that does not
    fit within a low-rank multi-linear model.

    The objective function is a weighted sum of the beta-divergence and L2,1
    norm, which allows for flexible noise modeling and imposing sparsity on the
    outliers. Missing values can be optionally handled via Expectation-
    Maximization. However, the model will no longer be identifiable.

    Parameters
    ----------
    data : np.ndarray
        An n-dimensional non-negative tensor. Missing values should be NaNs.

    rank : int
        Rank of the factorization/number of components.

    beta : float, range [0, 2]
        Float parameterizing the beta divergence.
        Values at certain limits:
            beta = 2: Squared Euclidean distance (Gaussian noise assumption)
            beta = 1: Kullback-Leibler divergence (Poisson noise assumption)
            beta = 0: Itakura-Saito divergence (multiplicative gamma noise
            assumption)
        Float values in between these integers interpolate between assumptions.

    reg_val : float
        Weight for the L2,1 penalty on the outlier tensor. Needs tuning
        specific to the range of the data. Start high and work your way down.

    tol : float
        tolerance on the iterative optimization.

    init : str, {'random' (default), 'user'}
        Initialization strategy.

    max_iter : int
        Maximum number of iterations to compute rNTF.

    print_every : int
        Print optimization progress every 'print_every' iterations.

    user_prov : None | dict
        Only relevant if init == 'user', i.e., you provide your own
        initialization. If so, provide a dictionary with the format:
        user_prov['factors'], user_prov['outlier'].

    Returns
    -------
    matrices : list
        A list of factor matrices retrieved from the decomposition.

    outlier : np.ndarray
        The outlier tensor retrieved from the decomposition.

    obj : np.ndarray, shape (n_iterations,)
        The history of the optimization.
    """
    
    # Utilities:
    eps = np.finfo(float).eps  # Small value to protect against division by zero

    # Initialize rNTF:
    matrices, outlier = initialize_rntf(data, rank, init, user_prov)

    # Set up for the algorithm:
    # Initial approximation of the reconstruction:
    data_approx = matrices[0] @ kr_bcd(matrices, 0).T
    data_approx = folder(data_approx, data, 0) + outlier + eps

    # EM step:
    ind = np.ones_like(data)
    ind[np.isnan(data)] = 0

    data_n = np.copy(data)
    data_n[ind == 0] = 0
    data_imp = data_n + (1 - ind) * data_approx

    fit = np.zeros(max_iter + 1)
    obj = np.zeros(max_iter + 1)

    # Monitoring convergence:
    fit[0] = beta_divergence(data_imp, data_approx, beta)
    obj[0] = fit[0] + reg_val * L21_norm(unfolder(outlier, 0))

    # Print initial iteration:
    if verbose:
        print(f'Iter = 0; Obj = {obj[0]}')

    for iter in range(max_iter):

        # EM step:
        data_imp = data_n + (1 - ind) * data_approx

        # Block coordinate descent/loop through modes:
        for mode in range(len(data_n.shape)):

            # Khatri-Rao product of the matrices being held constant:
            kr_term = kr_bcd(matrices, mode).T

            # Update factor matrix in mode of interest:
            matrices[mode] = update_factor(unfolder(data_imp, mode),
                                           unfolder(data_approx, mode),
                                           beta,
                                           matrices[mode],
                                           kr_term)

            # Update reconstruction:
            data_approx = (folder(matrices[mode] @ kr_term, data_n, mode)
                           + outlier
                           + eps)

            # Update outlier tensor:
            outlier = (folder(update_outlier(unfolder(data_imp, mode),
                                             unfolder(data_approx, mode),
                                             unfolder(outlier, mode),
                                             beta, reg_val),
                              data_n, mode))

            # Update reconstruction:
            data_approx = (folder(matrices[mode] @ kr_term, data_n, mode)
                           + outlier
                           + eps)

        # Monitor optimization:
        fit[iter+1] = beta_divergence(unfolder(data_imp, 0),
                                      unfolder(data_approx, 0),
                                      beta)
        obj[iter+1] = fit[iter+1] + reg_val * L21_norm(unfolder(outlier, 0))

        if (iter % print_every == 0) & verbose:  # print progress
            print(f'Iter = {iter+1}; Obj = {obj[iter+1]}; Err = {np.abs((obj[iter] - obj[iter+1]) / obj[iter])}')

        # Termination criterion:
        if np.abs((obj[iter] - obj[iter+1]) / obj[iter]) <= tol:
            print('Algorithm converged as per defined tolerance')
            break

        if iter == (max_iter - 1):
            print('Maximum number of iterations achieved')

    # In case the algorithm terminated early:
    obj = obj[:iter]
    fit = fit[:iter]

    return matrices, outlier, obj


def initialize_rntf(data, rank, alg, user_prov=None):
    """Intialize Robust Non-negative Tensor Factorization."""
    eps = np.finfo(float).eps  # Small value to protect against division by zero

    # Initialize outliers with uniform random values:
    outlier = (np.random.rand(*data.shape) + eps)

    # Initialize basis and coefficients:
    if alg == 'random':
        print('Initializing rNTF with uniform noise.')
        matrices = [np.random.rand(data.shape[idx], rank) + eps for idx in range(len(data.shape))]
        return matrices, outlier

    elif alg == 'user':
        print('Initializing rNTF with user input.')
        matrices = user_prov['factors']
        outlier = user_prov['outlier']
        return matrices, outlier

    else:
        raise ValueError(f'Invalid algorithm: {alg}')


def update_factor(data, data_approx, beta, factor, krp):
    """Update factor matrix."""
    return factor * ((data * (data_approx ** (beta - 2))) @ krp.T) / ((data_approx ** (beta - 1)) @ krp.T)


def update_outlier(data, data_approx, outlier, beta, reg_val):
    """Update matricized outlier tensor."""
    bet1 = lambda X: X ** (beta - 1)
    bet2 = lambda X: X ** (beta - 2)

    eps = 3e-16

    # Normalize the outlier matrix (L2 norm along the columns)
    norm_outlier = np.linalg.norm(outlier, axis=0, keepdims=True)
    norm_outlier = np.maximum(norm_outlier, eps)  # to avoid division by zero

    return outlier * ((data * bet2(data_approx)) / (bet1(data_approx) + reg_val * (outlier / norm_outlier)))

