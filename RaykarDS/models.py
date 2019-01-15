import numpy as np
from keras.optimizers import SGD

from keras import backend as K
from em_DSraykar import EPS
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from func_optimizers import GradientDescentOptimizer


#TODO: DOCSTRINGS!

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    def __init__(self, optimizer=GradientDescentOptimizer()):
        super().__init__()
        self.optimizer = optimizer

    def init_model(self, x, y):
        self.x = x
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
        p = sigmoid(np.matmul(self.x, w))
        p[p < EPS] = EPS
        p[p + EPS > 1] = 1 - EPS
        return p

    def grad_w(self, q, w, mu, l):
        """
        Gradient of w.
        :param w: Weights in linear regression.
        :param mu:
        :param l:
        :return:
        """
        p = self._p_w(w)

        koeff = p * (1 - p) * (mu * l / (p * l + q * (1 - l)) - (1 - mu) * l / (1 - p * l - q * (1 - l)))
        return np.squeeze(np.matmul(koeff[None, :], self.x))

    def hess_w(self, q, w, mu, l):
        """
        Hessian of w.
        :param w: Weights in linear regression.
        :param mu:
        :param l:
        :return: (d x d) np array.
        """

        trans_x = np.transpose(self.x)
        p = self._p_w(w)
        denom1 = (p * l + q * (1 - l))
        denom2 = (1 - p * l - q * (1 - l))
        numer1 = l * p * (1 - p) * (1 - 2 * p)
        numer2 = l * l * p * p * (1 - p) * (1 - p)
        koeff = mu * numer1 / denom1 - mu * numer2 / (denom1 * denom1) - \
                (1 - mu) * numer1 / denom2 - (1 - mu) * numer2 / (denom2 * denom2)
        x_with_values = np.transpose(koeff * trans_x)
        ans = np.dot(trans_x[None, :, :], x_with_values[None, :, :])
        ans_sq = np.squeeze(ans)
        return ans_sq

    def update_w(self, a, b, q, e_loglikelihood, mu, l):
        new_w = self.optimizer.optimize(
            self.w,
            func=lambda var: e_loglikelihood(self._p_w(var)),
            grad_func=lambda var: self.grad_w(q=q, w=var, mu=mu, l=l),
            hess_func=lambda var: self.hess_w(q=q, w=var, mu=mu, l=l)
        )

        self.w = new_w


class VGGModel(Model):
    def __init__(self,
                 optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                 weights_path=None,
                 nbatch=32):
        super().__init__()
        self.optimizer = optimizer
        self.weights_path = weights_path
        self.nbatch = nbatch

    def init_model(self, x, y):
        self.model = self.create_VGG_model()
        self.model.compile(optimizer=self.optimizer, loss=VGGModel.RaykarDS_loss)
        self.w_change = True

    def create_VGG_model(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        if self.weights_path:
            model.load_weights(self.weights_path)

        return model

    @staticmethod
    def RaykarDS_loss(params, p):
        a, b, q, mu, l = params

        loss = -K.sum((mu * K.log(a) + mu * K.log(p * l + q * (1 - l)) + (1 - mu) * K.log(b) + \
                 (1 - mu) * K.log(1 - p * l - q * (1 - l))))
        return loss

    def p(self):
        if self.w_change:
            self.p = self.model.predict(self.x)
            self.w_change = False
        return self.p

    def update_w(self, a, b, q, e_loglikelihood, mu, l):
        self.w_change = True
        self.model.fit([self.x], [a, b, q, mu, l], batch_size=self.nbatch, epochs=20)
