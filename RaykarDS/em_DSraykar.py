import numpy as np
from func_optimizers import GradientDescentOptimizer

# N -- tasks, M -- max count of marks
# y_i^j --> (N, M);
# x --> (N, D);
# w --> (D,)
# alpha^j --> (M,)
# beta^j --> (M,)
# mu --> (N,)
import sys

EPS = 1e-8


# TODO: add l to docstring

class EM_DS_Raykar:
    def __init__(self, x, y, y_real, model, l=None, max_steps=200, verbose=False):
        """
        :param x: Features (N x D).
        :param y: Worker answers (N x M).
        :param y_real: Provide y_real for debug purposes.
        :param model: Probability classification model. Contains hidden optimization parameter w.
        :param l: l that will be used. If no l are provided, l will be evaluated from other data.
        :param max_steps: Max count of EM-steps.
        :param verbose:
        """
        self.x = x
        self.y = y
        self.model = model
        self.l = l
        self.max_steps = max_steps
        self.y_real = y_real
        self.verbose = verbose

    def a(self, alpha):
        """
        Calculate the parameter a.
        :param alpha:
        :return: The a.
        """
        a = np.prod(np.power(alpha, self.y), axis=1) * np.prod(np.power(1 - alpha, 1 - self.y), axis=1)

        a[a < EPS] = EPS
        a[a > 1 - EPS] = 1 - EPS

        return a

    def b(self, beta):
        """
        Calculate the parameter b.
        :param beta:
        :return: The b.
        """
        b = np.prod(np.power(beta, 1 - self.y), axis=1) * np.prod(np.power(1 - beta, self.y), axis=1)

        b[b < EPS] = EPS
        b[b > 1 - EPS] = 1 - EPS

        return b

    def q(self):
        q = self.y.mean(axis=1)
        q[q < EPS] = EPS
        q[q > 1 - EPS] = 1 - EPS
        return q

    def p1(self, l):
        if self.l is not None:
            ll = self.l
        else:
            ll = l

        return ll * self.model.p() + (1 - ll) * self.q()

    def initialize_values(self):
        """
        Initialization values of alpha, beta, mu, w, l.
        :return: (alpha, beta, mu, w, l)
        """
        mu = self.y.mean(axis=1)
        mu[mu < 0.5] = 1 - mu[mu < 0.5]
        mu[mu > 1 - EPS] = 1 - EPS
        alpha = 0.5 * np.ones((self.y.shape[1],))
        beta = 0.5 * np.ones((self.y.shape[1],))
        self.model.init_model(x=self.x, y=self.y)
        l = 0.5 * np.ones((self.y.shape[0]))
        return alpha, beta, mu, l

    def update_vars(self, mu, l):
        if self.l is not None:
            ll = self.l
        else:
            ll = l

        new_alpha = np.matmul(np.transpose(self.y), mu) / (mu.sum())
        new_beta = np.matmul((1 - np.transpose(self.y)), (1 - mu)) / ((1 - mu).sum())
        self.model.update_w(
            self.a(new_alpha),
            self.b(new_beta),
            q=self.q(),
            e_loglikelihood=lambda p: self.e_loglikelihood(
                alpha=new_alpha,
                beta=new_beta,
                p=p,
                mu=mu,
                l=ll
            ),
            mu=mu,
            l=ll
        )

        q = self.q()
        p = self.model.p()

        ind_change = abs(q - p) > EPS

        new_l = l.copy()
        new_l[ind_change] = (q[ind_change] - mu[ind_change]) / (q[ind_change] - p[ind_change])

        # wrong_l = np.logical_or(new_l < 0, new_l > 1)
        # new_l[wrong_l] = l[wrong_l]

        l_less0 = new_l <= EPS
        l_more1 = new_l >= 1 - EPS
        new_l[l_less0] = EPS
        new_l[l_more1] = 1 - EPS

        return new_alpha, new_beta, new_l

    def e_loglikelihood(self, alpha, beta, p, mu, l):
        a = self.a(alpha)
        b = self.b(beta)
        q = self.q()

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

    def update_mu(self, alpha, beta, l):
        a = self.a(alpha)
        b = self.b(beta)
        p = self.model.p()
        q = self.q()

        if self.l is not None:
            ll = self.l
        else:
            ll = l

        new_mu = a * p / (a * p + b * (1 - p)) * ll + a * q / (a * q + b * (1 - q)) * (1 - ll)

        new_mu[new_mu < EPS] = EPS
        new_mu[new_mu > 1 - EPS] = 1 - EPS

        return new_mu

    def em_step(self, alpha, beta, mu, l):
        new_mu = self.update_mu(alpha, beta, l)
        new_alpha, new_beta, new_l = self.update_vars(new_mu, l)
        return new_alpha, new_beta, new_mu, new_l

    def em_algorithm(self):
        alpha, beta, mu, l = self.initialize_values()
        alpha, beta, l = self.update_vars(mu, l)
        exp_old = self.e_loglikelihood(alpha, beta, self.model.p(), mu, l)
        alpha, beta, mu, l = self.em_step(alpha, beta, mu, l)
        exp_new = self.e_loglikelihood(alpha, beta, self.model.p(), mu, l)

        step = 0
        if self.verbose:
            self.out(step, alpha, beta, mu, exp_new)
        while exp_new - exp_old > 0:
            if self.verbose:
                print("\nDiff = {}\n".format(exp_new - exp_old))
            alpha, beta, mu, l = self.em_step(alpha, beta, mu, l)
            exp_old = exp_new
            exp_new = self.e_loglikelihood(alpha, beta, self.model.p(), mu, l)
            step += 1
            if self.verbose:
                self.out(step, alpha, beta, mu, exp_new)

            if step > self.max_steps:
                break

        return alpha, beta, self.model.get_w(), self.p1(l), l

    @staticmethod
    def out(step, alpha, beta, mu, exp_new):
        print("--------------------\nStep={}\nlog E={}\n"
              .format(step, np.exp(exp_new / mu.shape[0])), sys.stderr)
