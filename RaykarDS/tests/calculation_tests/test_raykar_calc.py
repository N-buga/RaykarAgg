import unittest
import numpy as np

from em_DSraykar import EM_DS_Raykar, sigmoid


class TestRaykarCalculations(unittest.TestCase):
    def test_grad_w(self):
        x = np.array([[1, 2, 1], [0, 2, 2]])

        em_dsraykar = EM_DS_Raykar(y=np.array([[1, 0], [0, 1]]), x=x, l=1, y_real=None)

        mu = np.array([0.3, 0.2])
        w = np.array([1, 2, 3])

        ans = np.zeros((x.shape[1],))
        for i in range(x.shape[0]):
            ans += x[i]*(mu[i] - sigmoid(np.sum(w*x[i])))

        self.assertTrue(np.allclose(ans, em_dsraykar.grad_w(w, mu)))

# def test_hess_w(self):
#     x = np.array([[1, 2, 1], [0, 2, 2]])
#
#     em_dsraykar = EM_DS_Raykar(y=None, x=x, l=None)
#
#     w = np.array([1, 2, 3])
#
#     ans = np.zeros((x.shape[1], x.shape[1]))
#     for i in range(x.shape[0]):
#         ans -= np.matmul(np.transpose(x[i, None]), x[i, None])*(1 - sigmoid(np.sum(w*x[i])))*sigmoid(np.sum(w*x[i]))
#
#     self.assertTrue(np.allclose(ans, em_dsraykar.hess_w(w)))
