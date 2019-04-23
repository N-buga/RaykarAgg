import numpy as np
import scipy.special
from scipy import sparse

from em_DSraykar import EPS

from func_optimizers import GradientDescentOptimizer


#TODO: DOCSTRINGS!

class Model:
    def __init__(self):
        pass

    def init_model(self, x, y):
        self.x = x
        self.y = y
        pass

    def p(self):
        pass

    def get_w(self):
        pass

    def update_w(self, a, b, q, e_loglikelihood, mu, l):
        pass


class LogisticRegressionModel(Model):
    def __init__(self, optimizer=GradientDescentOptimizer(), reg_type='ridge', reg_coeff=0):
        super().__init__()
        self.optimizer = optimizer
        self.reg_type = reg_type
        self.reg_coeff = reg_coeff

    def init_model(self, x, y):
        self.x = x
        self.sparse_x = sparse.csc_matrix(self.x)
        self.y = y
        self.w = np.zeros((self.x.shape[1]))

    def get_w(self):
        return self.w

    def p(self):
        """
        Calculate p using model weights.
        :return:
        """
        return self._p_w(self.w)

    def _p_w(self, w):
        """
        Calculate p using given weights. Compatibility with optimizer.
        :param w:
        :return:
        """
        p = scipy.special.expit(self.sparse_x.dot(w))
        p[p < EPS] = EPS
        p[p + EPS > 1] = 1 - EPS
        return p

    def grad_w(self, q, w, mu, lambda_):
        """
        Gradient of w.
        :param w: Weights in linear regression.
        :param mu:
        :param l:
        :return:
        """
        p = self._p_w(w)

        koeff = p * (1 - p) * (mu * lambda_ / (p * lambda_ + q * (1 - lambda_)) - (1 - mu) * lambda_ / (1 - p * lambda_ - q * (1 - lambda_)))

        if self.reg_type == 'ridge':
            reg = 2*self.reg_coeff*w
        elif self.reg_type == 'lasso':
            reg = self.reg_coeff * np.where(w > 0, np.ones(w.shape[0]), -np.ones(w.shape[0]))
            reg[w == 0] = 0
        else:
            raise AttributeError("Wrong reg type")

        # dense_res = np.squeeze(np.matmul(koeff[None, :], self.x)) - self.x.shape[0]*reg
        sparse_res = np.squeeze(self.sparse_x.transpose().dot(koeff[None, :].transpose()).transpose()) - reg*self.x.shape[0]

        # pass
        # assert (sparse_res == reg).all()
        # assert np.allclose(dense_res, sparse_res)

        return sparse_res

    def hess_w(self, q, w, mu, lambda_):
        """
        Hessian of w.
        :param w: Weights in linear regression.
        :param mu:
        :param lambda_:
        :return: (d x d) np array.
        """

        trans_x = np.transpose(self.x)
        p = self._p_w(w)
        denom1 = (p * lambda_ + q * (1 - lambda_))
        denom2 = (1 - p * lambda_ - q * (1 - lambda_))
        numer1 = lambda_ * p * (1 - p) * (1 - 2 * p)
        numer2 = lambda_ * lambda_ * p * p * (1 - p) * (1 - p)
        koeff = mu * numer1 / denom1 - mu * numer2 / (denom1 * denom1) - \
                (1 - mu) * numer1 / denom2 - (1 - mu) * numer2 / (denom2 * denom2)
        x_with_values = np.transpose(koeff * trans_x)
        ans = np.dot(trans_x[None, :, :], x_with_values[None, :, :])
        ans_sq = np.squeeze(ans)

        if self.reg_type == 'ridge':
            reg = 2*self.reg_coeff
        elif self.reg_type == 'lasso':
            reg = 0
        else:
            raise AttributeError("Wrong reg type")

        return ans_sq - self.x.shape[0]*reg

    def update_w(self, a, b, q, e_loglikelihood, mu, lambda_):
        new_w = self.optimizer.optimize(
            self.w,
            func=lambda var: e_loglikelihood(self._p_w(var)),
            grad_func=lambda var: self.grad_w(q=q, w=var, mu=mu, lambda_=lambda_),
            hess_func=lambda var: self.hess_w(q=q, w=var, mu=mu, lambda_=lambda_)
        )

        self.w = new_w


# class SVRModel(Model):
#     def __init__(self,
#                  optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
#                  eps=0.1):
#         super().__init__()
#         self.optimizer = optimizer
#         self.eps = eps
#
#     def init_model(self, x, y):
#         self.x = x
#         self.y = y
#         self.w = np.zeros((self.x.shape[1]))
#
#     def _p_w(self, w):
#         pass
#
#     def p(self):
#         pass
#
#     def get_w(self):
#         pass
#
#     def grad_w(self, q, w, mu, l):
#         pass
#
#     def update_w(self, a, b, q, e_loglikelihood, mu, l):
#         new_w = self.optimizer.optimize(
#             self.w,
#             func=lambda var: e_loglikelihood(self._p_w(var)),
#             grad_func=lambda var: self.grad_w(q=q, w=var, mu=mu, l=l)
#         )
#
#         self.w = new_w
