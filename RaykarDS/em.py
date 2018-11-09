import numpy as np
from functools import partial

# N -- tasks, M -- max count of marks
# y_i^j --> (N, M);
# x --> (N, D);
# w --> (D,)
# alpha^j --> (M,)
# beta^j --> (M,)
# mu --> (N,)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def newton(w, grad_func, hess_func, step_counts=100, step=0.01):
    for i in range(step_counts):
        w -= step * np.matmul(np.linalg.inv(hess_func(w)), grad_func(w))
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

    def p_y1(self, w=None):
        if w is not None:
            return sigmoid(np.matmul(self.x, w))
        else:
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
        return ((mu - sigmoid(np.matmul(self.x, w)))*np.transpose(self.x)).sum(axis=1)

    def hess_w(self, w):
        """
        Hessian of w.
        :param w: Weights in linear regression.
        :return:
        """
        trans_x = np.transpose(self.x)
        x_with_values = np.transpose((1 - sigmoid(np.matmul(self.x, w)))*sigmoid(np.matmul(self.x, w))*trans_x)
        ans = -np.dot(trans_x[None, :, :], x_with_values[None, :, :])
        ans_sq = np.squeeze(ans)
        return ans_sq

    def update_vars(self, w, mu):
        new_alpha = np.matmul(np.transpose(self.y), mu)/(mu.sum())
        new_beta = np.matmul((1 - np.transpose(self.y)), (1 - mu))/((1 - mu).sum())
        new_w = newton(w, partial(self.grad_w, mu=mu), self.hess_w, 100)
        return new_alpha, new_beta, new_w

    def e_loglikelihood(self, a, b, w, mu):
        return (mu*np.log(a) + mu*np.log(self.p_y1(w)*self.l + 1 - self.l) + (1 - mu)*np.log(b) + \
               (1 - mu)*np.log((1 - self.p_y1(w))*self.l + 1 - self.l)).sum()

    def update_mu(self, a, b, w):
        pA = self.p_y1(w)
        pB = self.p_y1()
        return a*pA/(a*pA + b*(1 - pA))*self.l + a*pB/(a*pB + b*(1 - pB))*(1 - self.l)

    def em_step(self, alpha, beta, w, mu):
        new_mu = self.update_mu(self.a(alpha), self.b(beta), w)
        new_alpha, new_beta, new_w = self.update_vars(w, mu)
        return new_alpha, new_beta, new_w, new_mu

    def em_algorithm(self):
        alpha, beta, w, mu = self.initialize_values()
        alpha, beta, w = self.update_vars(w, mu)
        exp_old = self.e_loglikelihood(self.a(alpha), self.b(beta), w, mu)
        alpha, beta, w, mu = self.em_step(alpha, beta, w, mu)
        exp_new = self.e_loglikelihood(self.a(alpha), self.b(beta), w, mu)

        step = 0
        self.out(step, alpha, beta, w, mu, exp_new)
        while exp_new - exp_old > 0:
            if self.verbose:
                print("\nDiff = {}\n".format(exp_new - exp_old))
            alpha, beta, w, mu = self.em_step(alpha, beta, w, mu)
            exp_old = exp_new
            exp_new = self.e_loglikelihood(self.a(alpha), self.b(beta), w, mu)
            step += 1
            if self.verbose:
                self.out(step, alpha, beta, w, mu, exp_new)

        return alpha, beta, w, mu

    @staticmethod
    def out(step, alpha, beta, w, mu, exp_new):
        print("--------------------\nStep={}\nalpha={}\nbeta={}\nw={}\nmu={}\nlog E={}\n"
              .format(step, alpha, beta, w, mu, exp_new))
