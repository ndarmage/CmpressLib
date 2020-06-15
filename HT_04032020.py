#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:Author: Tomatis D.
:Date: 29/01/2018
:Company: CEA/DEN/DANS/DM2S/SERMA/LPEC

Implementation of the Hotelling transform, which is the
discrete version of the Karhunen-Loeve transform, according
to chap. 4 of [1].

References
----------

[1] SUN, Huifang et SHI, Yun Q. Image and Video Compression
    for Multimedia Engineering: Fundamentals, Algorithms, 
    and standards. CRC press, 2008.
"""

import numpy as np
import warnings

# set the machine precision to compare to zero
numerical_zero=np.finfo(np.float32).eps


def HTI(Y, m, E):
    """Inverse Hotelling transform.
    
    :param Y: real transformed matrix
    :param m: real array with original average values (per row)
    :param E: matrix with eigenvectors
    :type Y: ndarray, 2D array of `float` type
    :type m: ndarray, 1D array of `float` type
    :type E: ndarray, 2D array of `float` type
    :return Zp: anti-transformed matrix
    :rtype Zp: ndarray, 2D array of `float` type
    """
    mtype, n = type(m[0]), Y.shape[1]
    return (np.dot(E.T, Y) + np.outer(m, np.ones((n), dtype=mtype)))


def HT(Z, t=-.01, c=None, chk=False, d=None, force_SVD=False):
    """Apply the direct Hotelling transform to the input
    matrix Z. Note that eigenvectors and the corresponding
    eigenvectors (arranged per column) are set in decreasing
    order. The threshold for the allowed minimum mean squared
    reconstruction error (MSRE) is the input t, and it is equal
    to the sum of the discarded variances, i.e. the eigenvalues);
    it is given here as a fraction to 1, thus all variances are
    compared to their sum. Alternatively, an input fixed number
    of terms can be selected.
    
    This transform can also achieve the PCA, and it is a particular
    case of SVD.
    
    .. note:: t=0. disables the truncation after the transform.
    
    :param Z: input real 2D matrixM
    :param t: threshold for the MSRE
    :param c: number of coding terms
    :param chk: verify that the sum of squared error equals the 
    sum of discarded variances
    :param d: number of decimals used to check the property
    :param force_SVD: force the use of SVD 
    :type Z: ndarray, 2D array of `float` type
    :type t: float
    :type c: int
    :type chk: bool
    :type d: int
    :type force_SVD: bool
    :return Y: transformed matrix (with column-wise vectors),
    :return m: vector of average values per row
    :return Et: truncated matrix of eigenvectors
    :rtype Y: ndarray, 2D array of `float` type
    :rtype m: ndarray, 1D array of `float` type
    :rtype Ep: ndarray, 2D array of `float` type
    """
    n1, n2 = Z.shape
    nm = min(n1, n2)
    if c is not None:
        if type(c) != int:
            raise ValueError("input c must be integer")
        if not (0 < c <= nm):
            raise ValueError("invalid number of coding terms")
        if (t > 0.):
            warnings.warn(("Found both threshold and fixed coding"+
                          ". Fixed coding with {:d} terms will be"+
                          "used.").format(c))
    
    mtype = type(Z[0, 0])
    # compute the average values per row
    m = np.mean(Z, axis=1)
    M = Z - np.outer(m, np.ones((n2), dtype=mtype))
    
    # eigen-decomposition on the symmetric and positive semi-definite
    # matrix np.dot(M, M.T). This can be performed by SVD on the matrix
    # M itself. The eigenvalues in L will then be the squared singular
    # values of M, and E will contain the left-singular vectors.
    if n1 > n2 or force_SVD:
        E, S, V = np.linalg.svd(M, full_matrices=False)
        L = S**2; del V
    else:
    # the division of the matrix np.dot(M, M.T) by n1 is neglected here
    # note that the eigenvectors are already ortho-normalized,
    # i.e. np.dot(E.T, E) == np.diag(n1)
        L, E = np.linalg.eigh( np.dot(M, M.T) )
        
        # C = np.dot(M, M.T)  # covariance matrix (times n1)
        # np.testing.assert_array_almost_equal(C, 
        #     np.dot(Z, Z.T) - np.outer(m, m) * n1,
        #     err_msg="property of covariance matrix not verified")
    
    # sort the eigenvalues and corresponding eigenvectors
    idx = L.argsort()[::-1]
    # and reorder the eigenpairs
    L, E = L[idx], E[:,idx]
    # remove possible rounding-off error
    #if np.any( L < 0.): L = np.where( np.isclose(L, 0.), 0., L )
    # note that L contains variances, and it is expected as a
    # positive array. Any negative element must be very small.
    if len(L) != nm:
        raise RuntimeError("unexpected nb. of eigenvalues")
    
    Y = lambda E, M: np.dot(E, M)
    
    if c is None:
        # threshold coding when outputing
        Lr = L / np.sum(L)
        # print('check:', Lr); print(np.cumsum(Lr[::-1]))
        s, c = 0., nm - 1
        while (s <= t) and (c > -1):
            # the following abs is redundant if the negative-value-fix
            # above is enabled.
            s += abs(Lr[c])
            c -= 1
        c += 2
    #else apply the input fixed coding by truncation
    Ep = E[:,:c].T
    
    c_hat = (n1 * n2 - n1) // (n1 + n2)
    # if ((t > 0.) and (c is not None)) and (c > c_hat):
    if c > c_hat:
        warnings.warn("Negative savings, L=%d, I=%d." % (c, nm),
            RuntimeWarning)
    
    if chk:
        err = Z - HTI(Y(Ep, M), m, Ep)
        sse = np.linalg.norm(err, ord='fro')**2
        if d is None:
            d = np.finfo(type(sse)).precision - 1
        elif not isinstance(d, int):
            raise ValueError("input d must be integer")
        sum_discarded_vars = np.sum(L[c:])
        #if n1 != n2:  # mistaken
        #    sum_discarded_vars *= n2 / n1
        np.testing.assert_almost_equal(sse, sum_discarded_vars, decimal=d,
            err_msg="The sum of squared errors must be equal "+
                    "to the sum of the discared variances (eigs)")

    return Y(Ep, M), m, Ep

  
if __name__ == "__main__":
    
    dim = 10
    A = np.random.rand(dim, dim)
    A = np.load('../yama_uox/Z.npy').astype(np.float64)
    
    # test the HT without threshold coding (or truncation of terms)
    np.testing.assert_array_almost_equal(A, HTI( *list(HT(A, chk=True)) ), \
        err_msg="consistency check of direct-inverse Hotelling Transform")
    
    # verify threshold coding
    assert np.linalg.norm(A - HTI( *list(HT(A, t=0.10)) ), ord='fro') >= \
           np.linalg.norm(A - HTI( *list(HT(A, t=0.01)) ), ord='fro'), \
           "Higher threshold provides better accuracy after reconstruction?!?"
    
    try:
        Y, m, Ep = HT(A, t=0.01, chk=True, d=6)
    except AssertionError as e:
        print(e)

    A = np.load('../yama_uox/Z.npy').astype(np.float64)
    
    try:
        Y, m, Ep = HT(A, t=0.0005, chk=True, d=6)
    except AssertionError as e:
        print("Test large rectangular matrix failed.")
        print(e)
