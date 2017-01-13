# Created by ay27 at 17/1/13
import tensorflow as tf
import numpy as np
import src.ops as ops


def kruskal_to_tensor(factors):
    """Turns the Khatri-product of matrices into a full tensor

        ``factor_matrices = [|U_1, ... U_n|]`` becomes
        a tensor shape ``(U[1].shape[0], U[2].shape[0], ... U[-1].shape[0])``

    Parameters
    ----------
    factors : ndarray list
        list of factor matrices, all with the same number of columns
        i.e. for all matrix U in factor_matrices:
        U has shape ``(s_i, R)``, where R is fixed and s_i varies with i

    Returns
    -------
    ndarray
        full tensor of shape ``(U[1].shape[0], ... U[-1].shape[0])``

    Notes
    -----
    This version works by first computing the mode-0 unfolding of the tensor
    and then refolding it.
    There are other possible and equivalent alternate implementation.

    Version slower but closer to the mathematical definition
    of a tensor decomposition:

    >>> from functools import reduce
    >>> def kt_to_tensor(factors):
    ...     for r in range(factors[0].shape[1]):
    ...         vecs = np.ix_(*[u[:, r] for u in factors])
    ...         if r:
    ...             res += reduce(np.multiply, vecs)
    ...         else:
    ...             res = reduce(np.multiply, vecs)
    ...     return res

    """
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], ops.khatri(factors[1:]).T)
    return ops.fold(full_tensor, 0, shape)


def parafac(tensor, rank, n_iter_max=100, init='random', tol=10e-7,
            verbose=False):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)

        Computes a rank-`rank` decomposition of `tensor` [1]_ such that:
        ``tensor = [| factors[0], ..., factors[-1] |]``

    Parameters
    ----------
    tensor : ndarray
    rank  : int
            number of components
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity

    Returns
    -------
    factors : ndarray list
            list of factors of the CP decomposition
            element `i` is of shape (tensor.shape[i], rank)

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    tensor = tensor.astype(np.float)
    rng = np.random
    if init is 'random':
        factors = [rng.random_sample((tensor.shape[i], rank)) for i in range(tensor.ndim)]

    rec_errors = []
    norm_tensor = np.sqrt(np.sum(tensor**2))

    for iteration in range(n_iter_max):
        for mode in range(tensor.ndim):
            pseudo_inverse = np.ones((rank, rank))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse *= np.dot(factor.T, factor)
            factor = np.dot(ops.unfold(tensor, mode), ops.khatri(factors, skip_matrices_index=[mode]))
            factor = np.linalg.solve(pseudo_inverse.T, factor.T).T
            factors[mode] = factor

        # if verbose or tol:
        rec_error = np.sqrt(np.sum((tensor - kruskal_to_tensor(factors), 2)**2)) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print('reconsturction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break

    return factors
