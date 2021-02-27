"""Functions to carry out SPCA.

This module contains functions to find sparse loading vectors and SPCA
principal components for a particular data set.
"""
import numpy as np

def soft_thresh(x, l):
    """Soft-thresholding function.

    Method to reduce absolute values of vector entries by a specific quantity.

    Args:
        x (ndarray): 1D vector
        l (float): Positive quantity by which to reduce absolute value of
            each entry in x.

    Returns:

        ndarray of adjusted input vector.

    Raises:
        ValueError: If thresholding quantity is not positive.
    """
    if l <= 0:
        raise ValueError("Thresholding quantity must be positive.")
    x_ = x.copy()
    x_[np.bitwise_and(x < l, x > -l).flatten()] = 0
    # If greater than l, subtract l. If less than -l, add l
    x_[(x > l).flatten()] -= l
    x_[(x < -l).flatten()] += l
    return x_

def rank_n(X, t, tol=10e-4, rank=2, ran=False):
    """Generates n sparse loading vectors and SPCA principal components.

    Args:
        X (ndarray): 2D numpy array with features along columns and samples
            along rows. All columns should be centered to have zero mean.
        t (float): l1 norm constraint that determines level of sparsity.
        tol (float): Measure of SPCA algorithm convergence. Defaults to 10e-4.
        rank (int): Number of sparse loading vectors and SPCA principal 
            component vectors to return. Defaults to two.
        ran (bool): Intialises v in SPCA algorithm to equal density unit vector
            if False, and random unit vector if True. Defaults to False.
        
    Returns:

        Dict with sparse loading vector matrix and SPCA principal components
        matrix::

            {
                'V': 2D numpy array with sparse loading vectors as columns,
                'XV': 2D numpy array with SPCA principal components as columns,
                'U': 2D numpy array with SPCA u vectors as columns
            }
    """
    X_ = X.copy()
    Vh = []
    U = []
    X_r = []
    for i in range(rank):
        v, u = _sparse(X_, t, tol, ran=ran)
        U.append(u)
        Vh.append(v.T[0])
        X_r.append(np.dot(X_, v).T[0])
        X_ = X_ - np.dot(np.dot(u.T, X_),v) * np.dot(u, v.T)
    Uh = []
    for u in U:
        Uh.append(u.T[0])
    return {'V':np.array(Vh).T, 'XV':np.array(X_r).T, 'U':np.array(Uh).T}

def rank_n_o(X, t, tol=10e-4, rank=2, ran=False):
    """Generates n sparse loading vectors and orthogonal SPCA principal components.

    Args:
        X (ndarray): 2D numpy array with features along columns and samples
            along rows. All columns should be centered to have zero mean.
        t (float): l1 norm constraint that determines level of sparsity.
        tol (float): Measure of SPCA algorithm convergence. Defaults to 10e-4.
        rank (int): Number of sparse loading vectors and SPCA principal 
            component vectors to return. Defaults to two.
        ran (bool): Intialises v in SPCA algorithm to equal density unit vector
            if False, and random unit vector if True. Defaults to False.
        
    Returns:

        Dict with sparse loading vector matrix and orthogonal SPCA principal 
        components matrix::

            {
                'V': 2D numpy array with sparse loading vectors as columns,
                'XV': 2D numpy array with SPCA principal components as columns,
                'U': 2D numpy array with SPCA u vectors as columns
            }
    """
    X_ = X.copy()
    Vh = []
    U = []
    X_r = []
    for i in range(rank):
        P = np.eye(X_.shape[0])
        for j in range(i):
            P -= np.dot(U[j],U[j].T)
        v, u = _sparse_o(X_, t, P, tol, ran=ran)
        U.append(u)
        Vh.append(v.T[0])
        X_r.append(np.dot(X_, v).T[0])
        X_ = X_ - np.dot(np.dot(u.T, X_),v) * np.dot(u, v.T)
    Uh = []
    for u in U:
        Uh.append(u.T[0])
    return {'V':np.array(Vh).T, 'XV':np.array(X_r).T, 'U':np.array(Uh).T}

def _opt_thresh(x, t, tol):
    # norm = np.linalg.norm(x, ord=1)
    norm = np.linalg.norm(x/np.linalg.norm(x, ord=2), ord=1)
    if norm < t:
        return x
    l_u = np.max(np.abs(x))
    l_l = 0
    x_l = x.copy()
    count = 0
    while abs(norm - t) > tol:
        count += 1
        l_mid = (l_u + l_l)/2
        x_l = soft_thresh(x, l_mid)
        # norm = np.linalg.norm(x_l, ord=1)
        norm = np.linalg.norm(x_l/np.linalg.norm(x_l, ord=2), ord=1)
        if norm > t:
            l_l = l_mid
        else:
            l_u = l_mid
        if count > 25:
            break
    return x_l

def _sparse(X, t, tol, ran=False):
    if ran:
        random = np.random.RandomState(seed=0)
        v = np.array([[random.rand() for i in range(X.shape[1])]]).T
    else:
        v = np.array([[1.0 for i in range(X.shape[1])]]).T
    v = v/(np.linalg.norm(v, ord=2) + 0.00000001)
    
    count = 0
    while count < 100:
        Xv = np.dot(X, v)
        u = Xv / (np.linalg.norm(Xv, ord=2) + 0.00000001)
        S_XTu = _opt_thresh(np.dot(X.T,u), t, tol)
        v_n = S_XTu / (np.linalg.norm(S_XTu, ord=2) + 0.00000001)
        v_diff = np.linalg.norm(v - v_n, ord=1)
        if v_diff < tol:
            break
        v = v_n
        count += 1
    return v, u

def _sparse_o(X, t, P, tol, ran=False):
    if ran:
        random = np.random.RandomState(seed=0)
        v = np.array([[random.rand() for i in range(X.shape[1])]]).T
    else:
        v = np.array([[1.0 for i in range(X.shape[1])]]).T
    v = v/(np.linalg.norm(v, ord=2) + 0.00000001)
    
    count = 0
    while count < 100:
        PXv = np.dot(P,np.dot(X, v))
        u = PXv / (np.linalg.norm(PXv, ord=2) + 0.00000001)
        S_XTu = _opt_thresh(np.dot(X.T,u), t, tol)
        v_n = S_XTu / (np.linalg.norm(S_XTu, ord=2) + 0.00000001)
        v_diff = np.linalg.norm(v - v_n, ord=1)
        if v_diff < tol:
            break
        v = v_n
        count += 1
    return v, u

