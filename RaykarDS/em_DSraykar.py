import numpy as np
import scipy.special

# N -- tasks, M -- max count of marks
# y_i^j --> (N, M);
# x --> (N, D);
# w --> (D,)
# alpha^j --> (M,)
# beta^j --> (M,)
# mu --> (N,)
import sys

from func_optimizers import AdaGradOptimizer

EPS = 1e-8


# TODO: add l to docstring

class EM_DS_Raykar:
    def __init__(self, x, y, model, l=None, max_steps=200, verbose=False):
        """
        :param x: Features (N x D).
        :param y: Worker answers (N x M).
        :param model: Probability classification model. Contains hidden optimization parameter w.
        :param l: l that will be used. If no l are provided, l will be evaluated from other datasets.
        :param max_steps: Max count of EM-steps.
        :param verbose:
        """
        self.x = x
        self.y = y

        self.y_nan0 = np.nan_to_num(self.y)
        self.y_nan1 = np.where(np.isnan(self.y), 1, self.y)

        self.model = model
        self.l = l
        self.max_steps = max_steps
        self.verbose = verbose

    def a(self, alpha):
        """
        Calculate the parameter a.
        :param alpha:
        :return: The a.
        """

        a = np.prod(np.power(alpha, self.y_nan0), axis=1) * np.prod(np.power(1 - alpha, 1 - self.y_nan1), axis=1)

        a[a < EPS] = EPS
        a[a > 1 - EPS] = 1 - EPS

        return a

    def b(self, beta):
        """
        Calculate the parameter b.
        :param beta:
        :return: The b.
        """

        b = np.prod(np.power(beta, 1 - self.y_nan1), axis=1) * np.prod(np.power(1 - beta, self.y_nan0), axis=1)

        b[b < EPS] = EPS
        b[b > 1 - EPS] = 1 - EPS

        return b

    def q(self):
        q = np.nanmean(self.y, axis=1)
        q[q < EPS] = EPS
        q[q > 1 - EPS] = 1 - EPS
        return q

    def p1(self, l):
        if self.l is not None:
            lambda_ = scipy.special.expit(self.l)
        else:
            lambda_ = scipy.special.expit(l)

        return lambda_ * self.model.p() + (1 - lambda_) * self.q()

    def initialize_values(self):
        """
        Initialization values of alpha, beta, mu, w, l.
        :return: (alpha, beta, mu, w, l)
        """
        # mu = np.nanmean(self.y, axis=1)
        # mu[mu < 0.5] = 1 - mu[mu < 0.5]
        # mu[mu > 1 - EPS] = 1 - EPS

        alpha = 0.6 * np.ones((self.y.shape[1],))
        beta = 0.6 * np.ones((self.y.shape[1],))
        self.model.init_model(x=self.x, y=self.y)
        l = 0

        mu = self.update_mu(alpha, beta, l)

        return alpha, beta, mu, l

    def update_vars(self, mu, l):
        if self.l is not None:
            l = self.l

        mu_nan0 = np.where(np.transpose(~np.isnan(self.y)), mu, 0)
        mu_nan1 = np.where(np.transpose(~np.isnan(self.y)), mu, 1)

        new_alpha = np.matmul(np.transpose(self.y_nan0), mu) / (mu_nan0.sum(axis=1))
        new_beta = np.matmul((1 - np.transpose(self.y_nan1)), (1 - mu)) / ((1 - mu_nan1).sum(axis=1))
        self.model.update_w(
            self.a(new_alpha),
            self.b(new_beta),
            q=self.q(),
            e_loglikelihood=lambda p: self.e_loglikelihood(
                alpha=new_alpha,
                beta=new_beta,
                p=p,
                mu=mu,
                l=l
            ),
            mu=mu,
            lambda_=scipy.special.expit(l)
        )

        q = self.q()
        p = self.model.p()

        new_l = AdaGradOptimizer().optimize(
            l,
            func=lambda l: self.e_loglikelihood(
                alpha=new_alpha,
                beta=new_beta,
                p=p,
                mu=mu,
                l=l
            ),
            grad_func=lambda var: np.sum(scipy.special.expit(var) * (1 - scipy.special.expit(var)) *
                                  (mu * (p - q) / (scipy.special.expit(var) * p + q * (1 - scipy.special.expit(var))) +
                                   ((1 - mu) * (q - p) / (
                                           1 - p * scipy.special.expit(var) - q * (1 - scipy.special.expit(var)))))),
            hess_func=None
        )

        return new_alpha, new_beta, new_l

    def e_loglikelihood(self, alpha, beta, p, mu, l):
        if self.l is not None:
            l = self.l
        lambda_ = scipy.special.expit(l)

        a = self.a(alpha)
        b = self.b(beta)
        q = self.q()

        return (mu * np.log(a) + mu * np.log(p * lambda_ + q * (1 - lambda_)) + (1 - mu) * np.log(b) + \
                (1 - mu) * np.log(1 - p * lambda_ - q * (1 - lambda_))).sum()

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
        if self.l is not None:
            l = self.l
        lambda_ = scipy.special.expit(l)

        a = self.a(alpha)
        b = self.b(beta)
        p = self.model.p()
        q = self.q()

        new_mu = a * p / (a * p + b * (1 - p)) * lambda_ + a * q / (a * q + b * (1 - q)) * (1 - lambda_)

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

        return alpha, beta, self.model.get_w(), self.p1(l), l, exp_old

    @staticmethod
    def out(step, alpha, beta, mu, exp_new):
        print("--------------------\nStep={}\nlog E={}\n"
              .format(step, np.exp(exp_new / mu.shape[0])), sys.stderr)
