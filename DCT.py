#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:Author: Tomatis D.
:Date: 01/07/2018
:Company: CEA/DEN/DANS/DM2S/SERMA/LPEC

Implementation of the discrete cosine transform, according to 
chap. 4 of [1] and the FFTPACK implementation of SciPy v0.14.0
( docs.scipy.org/doc/scipy-0.14.0/reference/generated/
scipy.fftpack.dct.html ).

References
----------

[1] SUN, Huifang et SHI, Yun Q. Image and Video Compression
    for Multimedia Engineering: Fundamentals, Algorithms, 
    and standards. CRC press, 2008.
"""

import numpy as np
from scipy.fftpack import dct, idct
from numpy import random, testing
import warnings


def dct2D(A, norm=None, ttype=2):
    """Apply the Discrete Cosine Tranform to the element of the
    2D input matrix A. Default normalization option is `ortho` to
    have a orthonormal DCT-II transform.
    
    :param A: input real 2D matrix
    :param norm: type of normalization
    :type A: ndarray, 2D array of `float` type
    :type norm: str
    :return B: transformed matrix (with column-wise vectors),
    :rtype B: ndarray, 2D array of `float` type
    """
    
    B = dct( dct( A, axis=0, norm=norm, type=ttype ),
                     axis=1, norm=norm, type=ttype )
    return B


def idctII2D(A, norm='ortho'):
    """Apply the Inverse Transform of the orthonormalized DCT-II to
    the 2D input matrix A. That is the orthonormalized DCT-III.
    
    :param A: input real 2D matrix
    :param norm: type of normalization
    :type A: ndarray, 2D array of `float` type
    :type norm: str
    :return B: transformed matrix (with column-wise vectors),
    :rtype B: ndarray, 2D array of `float` type
    """
    
    B = dct( dct( A, axis=0, norm=norm, type=3 ),
                     axis=1, norm=norm, type=3 )
    return B


def idctI2D(A):
    """Apply the Inverse Transform of the orthonormalized DCT-I to
    the 2D input matrix A. That is the unnormalized DCT-I up to the
    factor c.
    
    :param A: input real 2D matrix
    :param norm: type of normalization
    :type A: ndarray, 2D array of `float` type
    :type norm: str
    :return B: transformed matrix (with column-wise vectors),
    :rtype B: ndarray, 2D array of `float` type
    """
    I, J = A.shape
    cI, cJ = .5 / float(I - 1), .5 / float(J - 1)
    B = dct( dct( A, axis=0, norm=None, type=1 ),
                     axis=1, norm=None, type=1 )
    return B*(cI*cJ)


def idctIII2D(A):
    """Apply the Inverse Transform of the orthonormalized DCT-III to
    the 2D input matrix A. That is the unnormalized DCT-II up to the
    factor c.
    
    :param A: input real 2D matrix
    :param norm: type of normalization
    :type A: ndarray, 2D array of `float` type
    :type norm: str
    :return B: transformed matrix (with column-wise vectors),
    :rtype B: ndarray, 2D array of `float` type
    """
    I, J = A.shape
    cI, cJ = .5 / float(I), .5 / float(J)
    B = dct( dct( A, axis=0, norm=None, type=2 ),
                     axis=1, norm=None, type=2 )
    return B*(cI*cJ)


def dctnD(A, norm=None, ttype=2):
    """Apply the Discrete Cosine Tranform to the element of the input
    multi-dimensional array A. Default option is no normalization and
    DCT-II transform.
    
    :param A: input real ndarray
    :param norm: type of normalization
    :type A: ndarray of `float` type
    :type norm: str
    :return B: ndarray of transformed coefficients,
    :rtype B: ndarray of `float` type
    """
    dims = A.shape
    nb_dims, nb_els = len(dims), np.prod(dims)
    B = np.array(A, copy=True)
    
    for di, d in enumerate(dims):
      B = dct( B.reshape(d, nb_els // d), \
               norm=norm, type=ttype, axis=0 ).T
    
    return B.reshape(dims)


def idctnD(A, norm=None, ttype=2):
    """Apply the Inverse Transform of the selected DCT-(ttype) to the
    input multi-dimensional n-D array A.
    
    :param A: input real n-D array
    :param norm: type of requested normalization
    :type A: ndarray of `float` type
    :type norm: str
    :return B: ndarray of transformed coefficients
    :rtype B: ndarray of `float` type
    """
    if ttype == 2: ttype = 3
    elif ttype == 3: ttype = 2
    elif ttype == 1: pass
    else:
        raise ValueError('Not yet implemented')

    return dctnD(A, norm, ttype)

  
if __name__ == "__main__":
    
    dim = 4
    A = random.rand(dim,dim+2)
    I, J = A.shape
    
    # --- test 1 ---
    # get the orthonorm.zed DCT-II coefficients
    C = dct2D(A, norm='ortho')
    
    # test the inverse transform
    testing.assert_array_almost_equal(A, idctII2D(C),
      err_msg="IDCT of ortho. DCT-II failed!")

    # --- test 2 ---
    # get the unnorm.zed DCT-II coefficients
    C = dct2D(A)
    
    # test the inverse transform
    testing.assert_array_almost_equal(A,
      idctII2D(C, norm=None)/(4.*I*J),
      err_msg="IDCT of unnorm. DCT-II failed!")
    
    # --- test 3 ---
    # get the unnorm.zed DCT-I coefficients
    C = dct2D(A, ttype=1)
    
    # test the inverse transform
    testing.assert_array_almost_equal(A, idctI2D(C),
      err_msg="IDCT of norm. DCT-I failed!")
    
    # --- test 4 ---
    # get the unnorm.zed DCT-III coefficients
    C = dct2D(A, ttype=3)
    
    # test the inverse transform
    testing.assert_array_almost_equal(A, idctIII2D(C),
      err_msg="IDCT of norm. DCT-III failed!")
    
    # --- test 5 ---
    # create a multi-dimensional array
    A = random.rand(2,3,4)
    nb_els, nb_dims, dims = A.size, A.ndim, A.shape
    
    # 5.1 multi-dimensional orthonorm.zed DCT-II
    C = dctnD(A, norm='ortho')
    testing.assert_array_almost_equal(A, idctnD(C, norm='ortho'),
      err_msg="IDCT of norm. n-D DCT-II failed!")

    # 5.2 multi-dimensional unnorm.zed DCT-II
    C, c = dctnD(A), 1. / (2**nb_dims * nb_els)
    testing.assert_array_almost_equal(A, idctnD(C, norm=None)*c,
      err_msg="IDCT of unnorm. n-D DCT-II failed!")

    # 5.3 multi-dimensional unnorm.zed DCT-I
    C = dctnD(A, ttype=1)
    c = 1. / (2**nb_dims * np.prod(np.array(dims) - 1))
    testing.assert_array_almost_equal(A, idctnD(C, ttype=1)*c,
      err_msg="IDCT of unnorm. n-D DCT-II failed!")

    # 5.4 multi-dimensional unnorm.zed DCT-III
    C = dctnD(A, ttype=3)
    c = 1. / (2**nb_dims * nb_els)
    testing.assert_array_almost_equal(A, idctnD(C, ttype=3)*c,
      err_msg="IDCT of unnorm. n-D DCT-II failed!")
