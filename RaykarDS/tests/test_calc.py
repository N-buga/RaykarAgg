import unittest
import numpy as np

from em import EM_DS_Raykar, sigmoid


class TestCalculations(unittest.TestCase):

    def test_a(self):
        y = np.array([[1, 0, 1], [1, 1, 1]])

        em_dsraykar = EM_DS_Raykar(y=y, x=None, l=None)

        alpha = np.array([1, 0.6, 0.2])
        assert(np.allclose(em_dsraykar.a(alpha), np.array([0.08, 0.12])))

    def test_b(self):
        y = np.array([[1, 0, 1], [1, 1, 1]])

        em_dsraykar = EM_DS_Raykar(y=y, x=None, l=None)

        beta = np.array([0.7, 0.6, 0.2])
        assert(np.allclose(em_dsraykar.b(beta), np.array([0.144, 0.096])))

    def test_initialize_values(self):
        x = np.array([[1, 1, 1], [2, 2, 2]])
        y = np.array([[0, 1, 0, 1], [1, 1, 1, 1]])

        em_dsraykar = EM_DS_Raykar(y=y, x=x, l=None)

        alpha, beta, w, mu = em_dsraykar.initialize_values()
        assert(np.allclose(alpha, np.array([0.5, 0.5, 0.5, 0.5])))
        assert(np.allclose(beta, np.array([0.5, 0.5, 0.5, 0.5])))
        assert(np.allclose(mu, np.array([0.5, 1])))
        assert(np.allclose(w, np.array([0, 0, 0])))

    def test_grad_w(self):
        x = np.array([[1, 2, 1], [0, 2, 2]])

        em_dsraykar = EM_DS_Raykar(y=None, x=x, l=None)

        mu = np.array([0.3, 0.2])
        w = np.array([1, 2, 3])

        ans = np.zeros((x.shape[1],))
        for i in range(x.shape[0]):
            ans += x[i]*(mu[i] - sigmoid(np.sum(w*x[i])))

        self.assertTrue(np.allclose(ans, em_dsraykar.grad_w(w, mu)))

    def test_hess_w(self):
        x = np.array([[1, 2, 1], [0, 2, 2]])

        em_dsraykar = EM_DS_Raykar(y=None, x=x, l=None)

        w = np.array([1, 2, 3])

        ans = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            ans -= np.matmul(np.transpose(x[i, None]), x[i, None])*(1 - sigmoid(np.sum(w*x[i])))*sigmoid(np.sum(w*x[i]))

        self.assertTrue(np.allclose(ans, em_dsraykar.hess_w(w)))

    def test_e_loglikelihood(self):
        x = np.array([[1, 2], [2, 3], [0, 1]])
        y = np.array([[0, 1], [1, 1], [0, 1]])

        em_dsraykar = EM_DS_Raykar(y=y, x=x, l=0.3)

        w = np.array([0.5, 1])
        alpha = np.array([0.6, 0.9])
        beta = np.array([0.7, 0.2])

        ps = em_dsraykar.p_y1(w)
        ans_p = [0.92414, 0.98201, 0.73106]
        for i, p in enumerate(ps):
            assert(abs(p - ans_p[i]) < 1e-4)

        a, b = em_dsraykar.a(alpha), em_dsraykar.b(beta)
        a_ans = np.array([0.36, 0.54, 0.36])
        b_ans = np.array([0.56, 0.24, 0.56])
        assert(np.allclose(a, a_ans))
        assert(np.allclose(b, b_ans))

        mus = em_dsraykar.update_mu(a, b, w)
        ans_mu = [0.5399433, 0.997577, 0.464722]
        for i, mu in enumerate(mus):
            assert(abs(mu - ans_mu[i]) < 1e-4)

        e_log = em_dsraykar.e_loglikelihood(a, b, w, mus)
        e_log_ans = -2.5613497
        assert(abs(e_log - e_log_ans) < 1e-4)

if __name__ == '__main__':
    unittest.main()