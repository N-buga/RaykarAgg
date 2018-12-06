import numpy as np
from optimizers import GradientDescentOptimizer

# N -- tasks, M -- max count of marks
# y_i^j --> (N, M);
# x --> (N, D);
# w --> (D,)
# alpha^j --> (M,)
# beta^j --> (M,)
# mu --> (N,)
import sys

EPS = 1e-8


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# TODO: add l to docstring

class EM_DS_Raykar:
    def __init__(self, x, y, y_real, l=None, max_steps=200, verbose=False, optimizer=None):
        """
        :param x: Features (N x D).
        :param y: Worker answers (N x M).
        :param y_real: Provide y_real for debug purposes.
        :param l: l that will be used. If no l are provided, l will be evaluated from other data.
        :param max_steps: Max count of EM-steps.
        :param verbose:
        :param optimizer: Optimizer for w.
        """
        self.x = x
        self.y = y
        self.l = l
        self.max_steps = max_steps
        self.y_real = y_real
        self.verbose = verbose
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = GradientDescentOptimizer()

    def a(self, alpha):
        """
        Calculate the parameter a.
        :param alpha:
        :return: The a.
        """
        a = np.prod(np.power(alpha, self.y), axis=1) * np.prod(np.power(1 - alpha, 1 - self.y), axis=1)
        return a

    def b(self, beta):
        """
        Calculate the parameter b.
        :param beta:
        :return: The b.
        """
        a = np.prod(np.power(beta, 1 - self.y), axis=1) * np.prod(np.power(1 - beta, self.y), axis=1)
        return a

    def p(self, w):
        return sigmoid(np.matmul(self.x, w))

    def q(self):
        q = self.y.mean(axis=1)
        q[q < EPS] = EPS
        q[q > 1 - EPS] = 1 - EPS
        return q

    def initialize_values(self):
        """
        Initialization values of alpha, beta, mu, w, l.
        :return: (alpha, beta, mu, w, l)
        """
        mu = np.mean(self.y, axis=1)
        alpha = 0.5 * np.ones((self.y.shape[1],))
        beta = 0.5 * np.ones((self.y.shape[1],))
        w = np.zeros(self.x.shape[1])
        l = 0.5 * np.ones((self.y.shape[0]))
        return alpha, beta, w, mu, l

    def grad_w(self, w, mu, l):
        """
        Gradient of w.
        :param w: Weights in linear regression.
        :param mu:
        :param l:
        :return:
        """
        p = self.p(w)
        p[p < EPS] = EPS
        p[p + EPS > 1] = 1 - EPS
        q = self.q()
        q[q < EPS] = EPS
        q[q + EPS > 1] = 1 - EPS

        if self.l is None:
            ll = l
        else:
            ll = self.l

        koeff = p * (1 - p) * (mu * ll / (p * ll + q * (1 - ll)) - (1 - mu) * ll / (1 - p * ll - q * (1 - ll)))
        return np.squeeze(np.matmul(koeff[None, :], self.x))

    def hess_w(self, w, mu, l):
        """
        Hessian of w.
        :param w: Weights in linear regression.
        :param mu:
        :param l:
        :return: (d x d) np array.
        """

        if self.l is not None:
            ll = self.l
        else:
            ll = l

        trans_x = np.transpose(self.x)
        p = self.p(w)
        q = self.q()
        denom1 = (p * ll + q * (1 - ll))
        denom2 = (1 - p * ll - q * (1 - ll))
        numer1 = ll * p * (1 - p) * (1 - 2 * p)
        numer2 = ll * ll * p * p * (1 - p) * (1 - p)
        koeff = mu * numer1 / denom1 - mu * numer2 / (denom1 * denom1) - \
                (1 - mu) * numer1 / denom2 - (1 - mu) * numer2 / (denom2 * denom2)
        x_with_values = np.transpose(koeff * trans_x)
        ans = np.dot(trans_x[None, :, :], x_with_values[None, :, :])
        ans_sq = np.squeeze(ans)
        return ans_sq

    def update_vars(self, w, mu, l):
        new_alpha = np.matmul(np.transpose(self.y), mu) / (mu.sum())
        new_beta = np.matmul((1 - np.transpose(self.y)), (1 - mu)) / ((1 - mu).sum())

        new_w = self.optimizer.optimize(
            w,
            func=lambda var: self.e_loglikelihood(alpha=new_alpha, beta=new_beta, w=var, mu=mu, l=l),
            grad_func=lambda var: self.grad_w(w=var, mu=mu, l=l),
            hess_func=lambda var: self.hess_w(w=var, mu=mu, l=l)
        )
        q = self.q()
        p = self.p(new_w)

        ind_change = abs(q - p) > EPS

        new_l = l.copy()
        new_l[ind_change] = (q[ind_change] - mu[ind_change]) / (q[ind_change] - p[ind_change])

        wrong_l = np.logical_or(new_l < 0, new_l > 1)
        new_l[wrong_l] = l[wrong_l]

        return new_alpha, new_beta, new_w, new_l

    def e_loglikelihood(self, alpha, beta, w, mu, l):
        a = self.a(alpha)
        b = self.b(beta)
        a[a < EPS] = EPS
        b[b < EPS] = EPS
        p = self.p(w)
        p[p < EPS] = EPS
        p[p + EPS > 1] = 1 - EPS
        q = self.q()
        q[q < EPS] = EPS
        q[q + EPS > 1] = 1 - EPS

        if self.l is not None:
            ll = self.l
        else:
            ll = l

        return (mu * np.log(a) + mu * np.log(p * ll + q * (1 - ll)) + (1 - mu) * np.log(b) + \
                (1 - mu) * np.log(1 - p * ll - q * (1 - ll))).sum()

    # def e_loglikelihoodRaykar(self, alpha, beta, w, mu):
    #     a = self.a(alpha)
    #     b = self.b(beta)
    #     a[a == 0] = EPS
    #     b[b == 0] = EPS
    #     p = self.p(w)
    #
    #     return (mu*np.log(a) + mu*np.log(p*self.l) + (1 - mu)*np.log(b) + \
    #            (1 - mu)*np.log((1 - p)*self.l)).sum()

    def update_mu(self, alpha, beta, w, l):
        a = self.a(alpha)
        b = self.b(beta)
        p = self.p(w)
        q = self.q()

        if self.l is not None:
            ll = self.l
        else:
            ll = l

        return a * p / (a * p + b * (1 - p)) * ll + a * q / (a * q + b * (1 - q)) * (1 - ll)

    def em_step(self, alpha, beta, w, mu, l):
        new_mu = self.update_mu(alpha, beta, w, l)
        new_alpha, new_beta, new_w, new_l = self.update_vars(w, new_mu, l)
        return new_alpha, new_beta, new_w, new_mu, new_l

    def em_algorithm(self):
        alpha, beta, w, mu, l = self.initialize_values()
        alpha, beta, w, l = self.update_vars(w, mu, l)
        exp_old = self.e_loglikelihood(alpha, beta, w, mu, l)
        alpha, beta, w, mu, l = self.em_step(alpha, beta, w, mu, l)
        exp_new = self.e_loglikelihood(alpha, beta, w, mu, l)

        step = 0
        if self.verbose:
            self.out(step, alpha, beta, w, mu, exp_new)
        while exp_new - exp_old > 0:
            if self.verbose:
                print("\nDiff = {}\n".format(exp_new - exp_old))
            alpha, beta, w, mu, l = self.em_step(alpha, beta, w, mu, l)
            exp_old = exp_new
            exp_new = self.e_loglikelihood(alpha, beta, w, mu, l)
            step += 1
            if self.verbose:
                self.out(step, alpha, beta, w, mu, exp_new)

            if step > self.max_steps:
                break

        return alpha, beta, w, mu, l

    @staticmethod
    def out(step, alpha, beta, w, mu, exp_new):
        print("--------------------\nStep={}\nlog E={}\n"
              .format(step, np.exp(exp_new / mu.shape[0])), sys.stderr)
