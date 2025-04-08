"""1. Подсчитать произведение ненулевых элементов на диагонали прямоугольной матрицы.
Для X = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 4, 4]]) ответ 3.

2. Даны два вектора x и y. Проверить, задают ли они одно и то же мультимножество.
Для x = np.array([1, 2, 2, 4]), y = np.array([4, 2, 1, 2]) ответ True.

3. Найти максимальный элемент в векторе x среди элементов, перед которыми стоит нулевой.
Для x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]) ответ 5.

4. Операции с изображением.
Дан трёхмерный массив, содержащий изображение, размера (height, width, numChannels), а также вектор длины numChannels. Сложить каналы изображения с указанными весами, и вернуть результат в виде матрицы размера (height, width). Считать реальное изображение можно при помощи функции scipy.misc.imread (если изображение не в формате png, установите пакет pillow: conda install pillow). Преобразуйте цветное изображение в оттенки серого, использовав коэффициенты np.array([0.299, 0.587, 0.114]).

5. Реализовать кодирование длин серий (Run-length encoding). Дан вектор x. Необходимо вернуть кортеж из двух векторов одинаковой длины. Первый содержит числа, а второй - сколько раз их нужно повторить.
Пример: x = np.array([2, 2, 2, 3, 3, 3, 5, 2, 2]). Ответ: (np.array([2, 3, 5, 2]), np.array([3, 3, 1, 2]))."""

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
