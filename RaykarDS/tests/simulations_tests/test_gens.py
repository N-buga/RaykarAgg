import unittest

import sys

from em_DSraykar import EM_DS_Raykar, sigmoid
from generate_data import generate_model_points, generate_points, generate_DS_points, generate_AB_points
import numpy as np


class TestGens(unittest.TestCase):
    def check_generation(self, x, y, w, alpha, beta, y_real, l, max_steps, acc, mu_exact_const, mu_ans_const):
        em_ds_raykar = EM_DS_Raykar(x, y, y_real=None, l=l, max_steps=max_steps)
        res_alpha, res_beta, res_w, res_mu = em_ds_raykar.em_algorithm()

        if w is not None:
            assert ((np.abs(w / w[0] - res_w / res_w[0]) < acc).all())
        if y_real is not None:
            assert ((np.abs(y_real - res_mu) < acc).sum() > mu_exact_const * x.shape[0])
            assert ((y_real == (res_mu > 0.5).astype(int)).sum() > mu_ans_const * x.shape[0])
        if alpha is not None:
            assert ((np.abs(alpha - res_alpha) < acc).all())
        if beta is not None:
            assert ((np.abs(beta - res_beta) < acc).all())

    def test_raykar_gen(self):
        x, y, y_real, y_all, alpha, beta, w = generate_model_points(50000, 5, 6,
                                                                    alpha=np.array([0.6, 0.7, 0.7, 0.8, 0.9]),
                                                                    beta=np.array([0.6, 0.7, 0.7, 0.8, 0.9]))
        self.check_generation(x, y, w, alpha, beta, y_real, 1, max_steps=200, acc=1e-2, mu_exact_const=0.95,
                              mu_ans_const=0.99)

    def test_raykar_gen2(self):
        x, y, y_real, y_all, alpha, beta, w = generate_points(50000, 5, 6, 1,
                                                              alpha=np.array([0.6, 0.7, 0.7, 0.8, 0.9]),
                                                              beta=np.array([0.6, 0.7, 0.7, 0.8, 0.9]))

        self.check_generation(x, y, w, alpha, beta, y_real, 1, max_steps=200, acc=1e-2, mu_exact_const=0.95,
                              mu_ans_const=0.99)

    def test_raykar_genAB(self):
        x, y, y_real, y_all, alpha, beta, w = generate_AB_points(50000, 5, 6, 1,
                                                                 alpha=np.array([0.6, 0.7, 0.7, 0.8, 0.9]),
                                                                 beta=np.array([0.6, 0.7, 0.7, 0.8, 0.9]))

        self.check_generation(x, y, w, alpha, beta, y_real, 1, max_steps=150, acc=1e-2, mu_exact_const=0.95,
                              mu_ans_const=0.99)

    def test_DS_gen(self):
        x, y, y_real, y_all, alpha, beta, w = generate_DS_points(100000, 6, 6,
                                                                 alpha=np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]),
                                                                 beta=np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]))
        self.check_generation(x, y, None, alpha, beta, y_real, 0, max_steps=200, acc=2e-2, mu_exact_const=0.75,
                              mu_ans_const=0.95)

    def test_DS_gen2(self):
        x, y, y_real, y_all, alpha, beta, w = generate_points(100000, 6, 6, 0,
                                                              alpha=np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]),
                                                              beta=np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]))

        self.check_generation(x, y, None, alpha, beta, y_real, 0, max_steps=200, acc=2e-2, mu_exact_const=0.75,
                              mu_ans_const=0.95)

    def test_DS_genAB(self):
        x, y, y_real, y_all, alpha, beta, w = generate_AB_points(100000, 6, 6, 0,
                                                                 alpha=np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]),
                                                                 beta=np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]))

        self.check_generation(x, y, None, alpha, beta, y_real, 0, max_steps=150, acc=2e-2, mu_exact_const=0.75,
                              mu_ans_const=0.95)

