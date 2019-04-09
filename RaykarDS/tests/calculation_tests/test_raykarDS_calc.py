import unittest
import numpy as np

from em_DSraykar import EM_DS_Raykar
from models import sigmoid, LogisticRegressionModel


class TestCalculations(unittest.TestCase):

    def test_a(self):
        y = np.array([[1, 0, 1], [1, 1, 1]])

        em_dsraykar = EM_DS_Raykar(y=y, x=None, model=None, y_real=None, l=None)

        alpha = np.array([1, 0.6, 0.2])
        assert(np.allclose(em_dsraykar.a(alpha), np.array([0.08, 0.12])))

    def test_b(self):
        y = np.array([[1, 0, 1], [1, 1, 1]])

        em_dsraykar = EM_DS_Raykar(y=y, x=None, model=None, y_real=None, l=None)

        beta = np.array([0.7, 0.6, 0.2])
        assert(np.allclose(em_dsraykar.b(beta), np.array([0.144, 0.096])))


class TestLogRegressionCalculations(unittest.TestCase):
    def test_initialize_values(self):
        x = np.array([[1, 1, 1], [2, 2, 2]])
        y = np.array([[0, 1, 0, 1], [1, 1, 1, 1]])

        em_dsraykar = EM_DS_Raykar(y=y, x=x, model=LogisticRegressionModel(), y_real=None, l=None)

        alpha, beta, mu, l = em_dsraykar.initialize_values()
        assert(np.allclose(alpha, np.array([0.5, 0.5, 0.5, 0.5])))
        assert(np.allclose(beta, np.array([0.5, 0.5, 0.5, 0.5])))
        assert(np.allclose(mu, np.array([0.5, 1])))
        assert(np.allclose(em_dsraykar.model.get_w(), np.array([0, 0, 0])))

    def test_grad_w(self):
        x = np.array([[1, 2, 1], [0, 2, 2]])
        y = np.array([[0, 1, 0, 1], [0, 1, 1, 1]])
        l = 0.6

        em_dsraykar = EM_DS_Raykar(y=y, x=x, model=LogisticRegressionModel(), y_real=None, l=l)
        em_dsraykar.initialize_values()

        mu = np.array([0.3, 0.2])
        w = np.array([1, 2, 3])

        ans = np.zeros((x.shape[1],))
        for i in range(x.shape[0]):
            cur_p = sigmoid(np.sum(w*x[i]))
            cur_q = y.mean(axis=1)[i]
            ans += x[i]*cur_p*(1 - cur_p)*(mu[i]*(l/(cur_p*l + (1 - l)*cur_q)) + (1 - mu[i])*(-l/((1 - cur_p)*l + (1 - l)*(1 - cur_q))))

        self.assertTrue(np.allclose(ans, em_dsraykar.model.grad_w(em_dsraykar.q(), w, mu, l)))

    def test_w(self):
        x = np.array([[1, 2, 1], [0, 2, 2]])
        y = np.array([[0, 1, 0, 1], [0, 1, 1, 1]])
        l = 0.6

        em_dsraykar = EM_DS_Raykar(y=y, x=x, model=LogisticRegressionModel(), y_real=None, l=l)
        em_dsraykar.initialize_values()

        w = np.array([1, 2, 3])
        mu = np.array([0.3, 0.2])

        ans = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            cur_p = sigmoid(np.sum(w*x[i]))
            cur_q = y.mean(axis=1)[i]
            ans += np.matmul(np.transpose(x[i, None]), x[i, None])*\
                   (mu[i]*l*cur_p*(1 - cur_p)*(1 - 2*cur_p)/(cur_p*l + cur_q*(1 - l)) -
                    mu[i]*l*l*cur_p*cur_p*(1 - cur_p)*(1 - cur_p)/((cur_p*l + cur_q*(1 - l))*(cur_p*l + cur_q*(1 - l))) -
                    (1 - mu[i])*l*cur_p*(1 - cur_p)*(1 - 2*cur_p)/((1 - cur_p)*l + (1 - l)*(1 - cur_q)) -
                    (1 - mu[i])*l*l*cur_p*cur_p*(1 - cur_p)*(1 - cur_p)/(((1 - cur_p)*l + (1 - l)*(1 - cur_q))*((1 - cur_p)*l + (1 - l)*(1 - cur_q))))

        self.assertTrue(np.allclose(ans, em_dsraykar.model.hess_w(em_dsraykar.q(), w, mu, l)))

    def test_e_loglikelihood(self):
        x = np.array([[1, 2], [2, 3], [0, 1]])
        y = np.array([[0, 1], [1, 1], [0, 1]])
        l = 0.3

        em_dsraykar = EM_DS_Raykar(y=y, x=x, model=LogisticRegressionModel(), l=0.3, y_real=None)
        em_dsraykar.initialize_values()

        w = np.array([0.5, 1])
        em_dsraykar.model.w = w
        alpha = np.array([0.6, 0.9])
        beta = np.array([0.7, 0.2])

        ps = em_dsraykar.model._p_w(w)
        ans_p = [0.92414, 0.98201, 0.73106]
        for i, p in enumerate(ps):
            assert(abs(p - ans_p[i]) < 1e-4)

        a, b = em_dsraykar.a(alpha), em_dsraykar.b(beta)
        a_ans = np.array([0.36, 0.54, 0.36])
        b_ans = np.array([0.56, 0.24, 0.56])
        assert(np.allclose(a, a_ans))
        assert(np.allclose(b, b_ans))

        mus = em_dsraykar.update_mu(alpha, beta, l)
        ans_mu = [0.5399433, 0.997577, 0.464722]
        for i, mu in enumerate(mus):
            assert(abs(mu - ans_mu[i]) < 1e-4)

        e_log = em_dsraykar.e_loglikelihood(alpha, beta, em_dsraykar.model.p(), mus, l)
        e_log_ans = -3.65826624
        assert(abs(e_log - e_log_ans) < 1e-4)


if __name__ == '__main__':
    unittest.main()