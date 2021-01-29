#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .base import ScoreEstimator


class SpectralScoreEstimator(ScoreEstimator):
    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=None):
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold
        super(SpectralScoreEstimator, self).__init__()

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        # samples: [..., M, x_dim]
        # x: [..., N, x_dim]
        # eigen_vectors: [..., M, n_eigen]
        # eigen_values: [..., n_eigen]
        # return: [..., N, n_eigen], by default n_eigen=M.
        M = tf.shape(samples)[-2]
        # Kxq: [..., N, M]
        # grad_Kx: [..., N, M, x_dim]
        # grad_Kq: [..., N, M, x_dim]
        Kxq = self.gram(x, samples, kernel_width)
        # Kxq = tf.Print(Kxq, [tf.shape(Kxq)], message="Kxq:")
        # ret: [..., N, n_eigen]
        ret = tf.sqrt(tf.cast(M, tf.float64)) * tf.matmul(tf.cast(Kxq, tf.float64), tf.cast(eigen_vectors, tf.float64))
        ret *= 1. / tf.expand_dims(tf.cast(eigen_values, tf.float64), axis=-2)
        return ret

    def compute_gradients(self, samples, x=None):
        # samples: [..., M, x_dim]
        # x: [..., N, x_dim]
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            # TODO: Simplify computation
            x = samples
        else:
            # _samples: [..., N + M, x_dim]
            _samples = tf.concat([samples, x], axis=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = tf.shape(samples)[-2]
        # Kq: [..., M, M]
        # grad_K1: [..., M, M, x_dim]
        # grad_K2: [..., M, M, x_dim]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * tf.cast(tf.eye(M), tf.float64)
        # eigen_vectors: [..., M, M]
        # eigen_values: [..., M]
        with tf.device("/cpu:0"):
            eigen_values, eigen_vectors = tf.self_adjoint_eig(Kq)
        # eigen_vectors = tf.matrix_inverse(Kq)
        # eigen_values = tf.reduce_sum(Kq, -1)
        # eigen_values = tf.Print(eigen_values, [eigen_values],
        #                         message="eigen_values:", summarize=20)
        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = tf.reduce_mean(
                tf.reshape(eigen_values, [-1, M]), axis=0)
            eigen_arr = tf.reverse(eigen_arr, axis=[-1])
            eigen_arr /= tf.reduce_sum(eigen_arr)
            eigen_cum = tf.cumsum(eigen_arr, axis=-1)
            self._n_eigen = tf.reduce_sum(
                tf.to_int32(tf.less(eigen_cum, self._n_eigen_threshold)))
            # self._n_eigen = tf.Print(self._n_eigen, [self._n_eigen],
            #                          message="n_eigen:")
        if self._n_eigen is not None:
            # eigen_values: [..., n_eigen]
            # eigen_vectors: [..., M, n_eigen]
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        # eigen_ext: [..., N, n_eigen]
        eigen_ext = self.nystrom_ext(
            samples, x, eigen_vectors, eigen_values, kernel_width)
        # grad_K1_avg = [..., M, x_dim]
        grad_K1_avg = tf.reduce_mean(grad_K1, axis=-3)
        # beta: [..., n_eigen, x_dim]
        beta = -tf.sqrt(tf.cast(M, tf.float64)) * tf.matmul(
            tf.cast(eigen_vectors, tf.float64), tf.cast(grad_K1_avg, tf.float64), transpose_a=True) / tf.expand_dims(
            tf.cast(eigen_values, tf.float64), -1)
        # grads: [..., N, x_dim]
        grads = tf.matmul(eigen_ext, beta)
        return grads
