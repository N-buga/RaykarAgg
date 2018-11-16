import numpy as np
from functools import partial

# N -- tasks, M -- max count of marks
# y_i^j --> (N, M);
# x --> (N, D);
# w --> (D,)
# alpha^j --> (M,)
# beta^j --> (M,)
# mu --> (N,)

EPS = 1e-8


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def newton(w, grad_func, hess_func, step_counts=100, step=0.01):
    for i in range(step_counts):
        w -= step * np.matmul(np.linalg.inv(hess_func(w)), grad_func(w))
    return w


def grad_descent(w, grad_func, step_counts=100, step=0.001):
    for i in range(step_counts):
        w += step * grad_func(w)
    return w


class EM_DS_Raykar:
    def __init__(self, x, y, l, verbose=False):
        self.x = x
        self.y = y
        self.l = l
        self.verbose = verbose

    def a(self, alpha):
        """
        Calculate the parameter a.
        :param alpha:
        :return: The a.
        """
        a = np.prod(np.power(alpha, self.y), axis=1)*np.prod(np.power(1 - alpha, 1 - self.y), axis=1)
        return a

    def b(self, beta):
        """
        Calculate the parameter b.
        :param beta:
        :return: The b.
        """
        a = np.prod(np.power(beta, 1 - self.y), axis=1)*np.prod(np.power(1 - beta, self.y), axis=1)
        return a

    def p(self, w):
        return sigmoid(np.matmul(self.x, w))

    def q(self):
        return self.y.mean(axis=1)

    def initialize_values(self):
        """
        Initialization values of alpha, beta, mu, w.
        :param x: Description of each task.
        :param y: Array of worker answers on each task.
        :return: (alpha, beta, mu, w)
        """
        mu = np.mean(self.y, axis=1)
        alpha = 0.5*np.ones((self.y.shape[1],))
        beta = 0.5*np.ones((self.y.shape[1],))
        w = np.zeros(self.x.shape[1])
        return alpha, beta, w, mu

    def grad_w(self, w, mu):
        """
        Gradient of w.
        :param w: Weights in linear regression.
        :param mu:
        :return:
        """
        p = self.p(w)
        q = self.q()
        koeff = p*(1 - p)*(mu*self.l/(p*self.l + q*(1 - self.l)) - (1 - mu)*self.l/(1 - p*self.l - q*(1 - self.l)))
        return np.squeeze(np.matmul(koeff[None, :], self.x))

    # def hess_w(self, w):
    #     """
    #     Hessian of w.
    #     :param w: Weights in linear regression.
    #     :return:
    #     """

    def update_vars(self, w, mu):
        new_alpha = np.matmul(np.transpose(self.y), mu)/(mu.sum())
        new_beta = np.matmul((1 - np.transpose(self.y)), (1 - mu))/((1 - mu).sum())
        new_w = grad_descent(w, partial(self.grad_w, mu=mu), 100)
        return new_alpha, new_beta, new_w

    def e_loglikelihood(self, alpha, beta, w, mu):
        a = self.a(alpha)
        b = self.b(beta)
        a[a == 0] = EPS
        b[b == 0] = EPS
        p = self.p(w)
        q = self.q()
        return (mu*np.log(a) + mu*np.log(p*self.l + q*(1 - self.l)) + (1 - mu)*np.log(b) + \
               (1 - mu)*np.log(1 - p*self.l - q*(1 - self.l))).sum()

    def update_mu(self, alpha, beta, w):
        a = self.a(alpha)
        b = self.b(beta)
        p = self.p(w)
        q = self.q()
        return a*p/(a*p + b*(1 - p))*self.l + a*q/(a*q + b*(1 - q))*(1 - self.l)

    def em_step(self, alpha, beta, w, mu):
        new_mu = self.update_mu(alpha, beta, w)
        new_alpha, new_beta, new_w = self.update_vars(w, mu)
        return new_alpha, new_beta, new_w, new_mu

    def em_algorithm(self):
        alpha, beta, w, mu = self.initialize_values()
        alpha, beta, w = self.update_vars(w, mu)
        exp_old = self.e_loglikelihood(alpha, beta, w, mu)
        alpha, beta, w, mu = self.em_step(alpha, beta, w, mu)
        exp_new = self.e_loglikelihood(alpha, beta, w, mu)

        step = 0
        self.out(step, alpha, beta, w, mu, exp_new)
        while exp_new - exp_old > 0:
            if self.verbose:
                print("\nDiff = {}\n".format(exp_new - exp_old))
            alpha, beta, w, mu = self.em_step(alpha, beta, w, mu)
            exp_old = exp_new
            exp_new = self.e_loglikelihood(alpha, beta, w, mu)
            step += 1
            if self.verbose:
                self.out(step, alpha, beta, w, mu, exp_new)

            if step > 1000:
                break

        return alpha, beta, w, mu

    @staticmethod
    def out(step, alpha, beta, w, mu, exp_new):
        print("--------------------\nStep={}\nlog E={}\n"
              .format(step, exp_new))
