# import numpy as np
# from functools import partial
#
# # N -- tasks, M -- max count of marks
# # y_i^j --> (N, M);
# # x --> (N, D);
# # w --> (D,)
# # alpha^j --> (M,)
# # beta^j --> (M,)
# # mu --> (N,)
#
# EPS = 1e-8
#
# def sigmoid(x):
#   return 1 / (1 + np.exp(-x))
#
#
# def newton(w, grad_func, hess_func, step_counts=100, step=0.01):
#     for i in range(step_counts):
#         print(i)
#         w -= step * np.matmul(np.linalg.inv(hess_func(w)), grad_func(w))
#     return w
#
#
# class EM_DS_Raykar:
#     def grad_w(self, w, mu):
#         """
#         Gradient of w.
#         :param w: Weights in linear regression.
#         :param mu:
#         :return:
#         """
#         return ((mu - sigmoid(np.matmul(self.x, w)))*np.transpose(self.x)).sum(axis=1)
#
#     def hess_w(self, w):
#         """
#         Hessian of w.
#         :param w: Weights in linear regression.
#         :return:
#         """
#         trans_x = np.transpose(self.x)
#         x_with_values = np.transpose((1 - sigmoid(np.matmul(self.x, w)))*sigmoid(np.matmul(self.x, w))*trans_x)
#         ans = -np.dot(trans_x[None, :, :], x_with_values[None, :, :])
#         ans_sq = np.squeeze(ans)
#         return ans_sq
#
#     @staticmethod
#     def out(step, alpha, beta, w, mu, exp_new):
#         print("--------------------\nStep={}\nalpha={}\nbeta={}\nw={}\nmu={}\nlog E={}\n"
#               .format(step, alpha, beta, w, mu, exp_new))
