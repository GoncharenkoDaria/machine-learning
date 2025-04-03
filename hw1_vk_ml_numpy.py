import numpy as np


def product_of_diagonal_elements_vectorized(matrix: np.array):
    diagonal = np.diag(matrix)
    non_zero_elements = diagonal[diagonal != 0]

    return np.prod(non_zero_elements)


def are_equal_multisets_vectorized(x: np.array, y: np.array):

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    return np.array_equal(x_sorted, y_sorted)


def max_before_zero_vectorized(x: np.array):

    zero_indices = np.where(x[:-1] == 0)[0]

    elements_after_zero = x[zero_indices + 1]

    return (np.max(elements_after_zero))


def add_weighted_channels_vectorized(image: np.array):
    weights = np.array([0.299, 0.587, 0.114])
    return image@weights


def run_length_encoding_vectorized(x: np.array):
    change_indices = np.flatnonzero(np.concatenate(([True], x[1:] != x[:-1])))
    lengths = np.diff(np.concatenate((change_indices, [x.size])))
    values = x[change_indices]
    return values, lengths
