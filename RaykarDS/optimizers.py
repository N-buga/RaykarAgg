import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Base class.
    """
    def optimize(self, var, func, grad_func, hess_func):
        """
        Maximise the function with var start point.
        :param var: start value
        :param func: Function which we maximise.
        :param grad_func: Gradient function.
        :param hess_func: The hessian of function.
        :return: optimal value.
        """
        raise NotImplementedError('Base class')

    def description(self):
        """
        Return string description of optimizer
        :return:
        """
        raise NotImplementedError('Base class')


class GradientDescentOptimizer(Optimizer):
    """
    Gradient descent optimization.
    """
    def __init__(self, step=0.001, steps_count=100):
        """
        Set parameters for gradient descent.
        :param step: The value of step.
        :param steps_count:
        """
        self.step = step
        self.steps_count = steps_count

    def description(self):
        """
        Return string description of optimizer
        :return:
        """
        return "GradientDescentOptimizer step={}; steps_count={}".format(self.step, self.steps_count)

    def optimize(self, var, func, grad_func, hess_func=None):
        """
        Maximise the function with var start point.
        :param var: start value
        :param func: Function which we maximise.
        :param grad_func: Gradient function.
        :param hess_func: The hessian of function. Isn't necessary to provide.
        :return: optimal value.
        """
        new_var = var.copy()
        old_var = new_var
        old_func_value = func(old_var)
        for i in range(self.steps_count):
            new_var += self.step * grad_func(new_var)
            new_func_value = func(new_var)
            if new_func_value < old_func_value:
                return old_var
            else:
                old_var = new_var
                old_func_value = new_func_value
        return new_var


class NewtonOptimizer(Optimizer):
    """
    Newton-Raphson optimization.
    """
    def __init__(self, step=0.01, steps_count=50):
        """
        Set parameters for Newton-Raphson optimization.
        :param step: The value of step.
        :param steps_count:
        """
        self.step = step
        self.steps_count = steps_count

    def description(self):
        """
        Return string description of optimizer
        :return:
        """
        return "NewtonOptimizer step={}; steps_count={}".format(self.step, self.steps_count)

    def optimize(self, var, func, grad_func, hess_func):
        """
        Maximise the function with var start point.
        :param var: start value
        :param func: Function which we maximise.
        :param grad_func: Gradient function.
        :param hess_func: The hessian of function.
        :return: optimal value.
        """
        new_var = var.copy()
        # old_var = new_var
        # old_func_value = func(old_var)
        for i in range(self.steps_count):
            try:
                inv_hess_func = np.linalg.inv(hess_func(new_var))
                if np.isnan(inv_hess_func).any():
                    new_var += self.step * self.step * grad_func(new_var)
                else:
                    new_var -= self.step * np.matmul(np.linalg.inv(hess_func(new_var)), grad_func(new_var))
            except np.linalg.LinAlgError:
                new_var += self.step * self.step * grad_func(new_var)
            # new_func_value = func(new_var)
            # if new_func_value < old_func_value:
            #     return old_var
            # else:
            #     old_var = new_var
            #     old_func_value = new_func_value
        return new_var


class ScipyOptimizer(Optimizer):
    """
    Optimizers that are defined in scipy.optimize.minimize.
    """

    def __init__(self, method, options=None):
        """
        Set parameters for scipy.optimize.minimize function.
        :param method: The method to use.
        :param options: Options for method.
        """
        self.method = method
        if options:
            self.options = options
        else:
            self.options = {'maxiter': 100}

    def description(self):
        return 'Scipy optimizer with method {} and options {}'.format(self.method, self.options)

    def optimize(self, var, func, grad_func, hess_func):
        """
        Maximise the function with var start point.
        :param var: start value
        :param func: Function which we maximise.
        :param grad_func: Gradient function.
        :param hess_func: The hessian of function.
        :return: optimal value.
        """
        res = minimize(fun=func, x0=var, method=self.method, jac=grad_func, hess=hess_func, options=self.options)
        return res.x
