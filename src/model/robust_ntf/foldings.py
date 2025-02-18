#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains functions to matricize (unfold) and tensorize (fold).

This mainly exists to undo the indexing style of PyTorch/TensorLy and revert to
the indexing style of Kolda and Bader, without having to rewrite several
functions in both PyTorch and TensorLy.

For background, PyTorch and TensorLy both assume C-style indexing, whereas I
originally developed my code based on Kolda & Bader's formulation which uses
Fortran-style indexing, which is slower. If you would like to rewrite rNTF to
use the faster C-style indexing, please feel free to open a PR.

An excellent blog post about this: http://jeankossaifi.com/blog/unfolding.html

Contents
--------
    permutation_generator() : generates the permutation order for the tensor
                              such that the correct operation will be applied
                              by folding and unfolding.
    folder() : tensorizes a matrix using the permutation order given by
               permutation_generator() and reshaping.
    unfolder() : matricizes a tensor using the permutation order given by
                 permutation_generator() and reshaping.

If you find bugs and/or limitations, please email neel DOT dey AT nyu DOT edu.

Created March 2019, refactored September 2019.
"""

import numpy as np
import tensorly as tl



def permutation_generator(dims, mode):
    """Generates the permutation order for the tensor such that the correct
    operation will be applied by folding and unfolding.

    Parameters
    ----------
    dims : tuple
        Shape of original tensor.
    mode : int
        Mode to perform folding/unfolding on.

    Returns
    -------
    ordering : list
        Permutation order for the tensor for correct operations.
    """
    ordering = list(range(len(dims)))
    ordering.remove(mode)
    ordering.reverse()
    ordering.insert(mode, mode)
    return ordering



def folder(mat, ten, mode):
    """Tensorizes a matrix using the permutation order given by
    permutation_generator() and then by using tl.fold().

    Parameters
    ----------
    mat : np.ndarray
        Matrix to be rearranged into a tensor.
    ten : np.ndarray
        Original data tensor, used here for a reference shape.
    mode : int
        Mode along which to rearrange.

    Returns
    -------
    np.ndarray
        Tensorization of the input matrix.
    """

    ten_shape = ten.shape  # Using .shape instead of .size() for consistency with numpy arrays
    permute_order = permutation_generator(ten_shape, mode)

    # Compute the size after permutation
    new_size = np.array(ten.shape)
    new_size[mode] = mat.shape[0]  # Adjust the mode dimension to match the mat's first dimension

    # Perform fold and permute operations
    return np.transpose(tl.fold(mat, mode, np.transpose(ten, permute_order).shape), permute_order)





def unfolder(ten, mode):
    """Matricizes a tensor using the permutation order given by
    permutation_generator() and reshaping.

    Parameters
    ----------
    ten : np.ndarray
        Tensor to be matricized.
    mode : int
        Mode along which to rearrange.

    Returns
    -------
    np.ndarray
        Matricized form of the input tensor.
    """
    ten_shape = ten.shape
    permute_order = permutation_generator(ten_shape, mode)

    # Permute tensor
    permuted_ten = np.transpose(ten, axes=permute_order)

    # Reshape to 2D matrix
    new_shape = (ten_shape[mode], -1)  
    return permuted_ten.reshape(new_shape)
