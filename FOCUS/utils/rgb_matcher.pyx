"""Extract ROIs from an image."""

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: language_level=3
# cython: cdivision=True

import cython
cimport numpy as np
import numpy as np

from numpy.math cimport INFINITY
from libc.stdlib cimport malloc, free

from libc.stdint cimport int32_t

ctypedef np.float32_t DTYPE32_t
ctypedef np.uint8_t DTYPE8_t
ctypedef np.int32_t DTYPE32i_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE32_t, ndim=4] _rgb_match(np.ndarray[DTYPE32_t, ndim=3] image, np.ndarray[DTYPE32i_t, ndim=2] corner, int width, int height, np.ndarray[DTYPE32_t, ndim=2] target_color):
    """Extract N=len(corner) best matches from image ROIs from image.
    
    Return relative to [N x 3] baseline.
    """
    cdef int N = corner.shape[0]

    # In flat space.
    cdef np.ndarray[DTYPE32i_t, ndim=1] best_match = np.zeros((N), dtype=np.int32)
    cdef int[:] _best_match = best_match

    cdef float[:, :, :] img = image
    cdef float[:, :] _color = target_color
    cdef float[:] this_target_color

    cdef int x, y, i, j, k, c
    cdef float diff
    cdef float sq_sum
    cdef float best_sq_sum
    for i in range(N):
        x = corner[i, 0]
        y = corner[i, 1]
        this_target_color = _color[i]
        best_sq_sum = INFINITY
        for j in range(height):
            for k in range(width):
                sq_sum = 0
                for c in range(3):  # Loop over color channels
                    diff = img[y + j, x + k, c] - this_target_color[c]
                    sq_sum += diff * diff


                if sq_sum < best_sq_sum:
                    best_sq_sum = sq_sum
                    this_best_match = j * width + k

        _best_match[i] = this_best_match

    return best_match


def rgb_match(image, corner, width, height, target_color):
    """Extract len(corner) ROIs from image."""
    return _rgb_match(image, corner, width, height, target_color)
