import numpy as np
import pandas as pd
from models import sigmoid


def generate_AB_points(n, m, d, l_parts,
                       alpha, beta,
                       random_state=0):
    np.random.seed(random_state)
    x = np.random.normal(size=(n, d))
    w = np.random.normal(size=(d,))

    model = np.random.binomial(size=n, n=1, p=l_parts)
    y_real_A = np.where(sigmoid(np.matmul(x, w)) > 0.5, np.ones((n,)), np.zeros((n,)))
    y_real_B = np.random.binomial(size=n, n=1, p=0.5)
    y_real = np.where(model == 1, y_real_A, y_real_B)
    y_all = np.zeros((n*m, 4))
    y_workers = np.zeros((n, m))

    for i in range(n):
        y_all[i*m:(i + 1)*m, 0] = i
        y_all[i*m:(i + 1)*m, 1] = list(range(m))
        y_all[i*m:(i + 1)*m, 3] = y_real[i]
        if y_real[i] == 1:
            y_all[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=alpha)
            y_workers[i] = y_all[i*m:(i + 1)*m, 2]
        else:
            y_all[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=1-beta)
            y_workers[i] = y_all[i*m:(i + 1)*m, 2]

    l = (y_real == y_real_A).mean()

    return x, y_workers, y_real, y_all, alpha, beta, w, l


def generate_model_points(n, m, d,
                    alpha, beta):
    np.random.seed(0)
    w = np.random.normal(loc=0, scale=1, size=(d,))
    x = np.random.normal(loc=0, scale=1, size=(n, d))
    y_real = (sigmoid(np.matmul(x, w)) > 0.5).astype(int)
    y_all = np.zeros((n*m, 4))
    y_workers = np.zeros((n, m))

    for i in range(n):
        y_all[i*m:(i + 1)*m, 0] = i
        y_all[i*m:(i + 1)*m, 1] = list(range(m))
        y_all[i*m:(i + 1)*m, 3] = y_real[i]
        if y_real[i] == 1:
            y_all[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=alpha)
            y_workers[i] = y_all[i*m:(i + 1)*m, 2]
        else:
            y_all[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=1-beta)
            y_workers[i] = y_all[i*m:(i + 1)*m, 2]

    return x, y_workers, y_real, y_all, alpha, beta, w, 1


def generate_DS_points(n, m, d,
                    alpha, beta):
    np.random.seed(0)
    x = np.random.normal(size=(n, d))
    y_real = np.random.binomial(size=n, n=1, p=0.5)
    y_all = np.zeros((n*m, 4))
    y_workers = np.zeros((n, m))

    for i in range(n):
        y_all[i*m:(i + 1)*m, 0] = i
        y_all[i*m:(i + 1)*m, 1] = list(range(m))
        y_all[i*m:(i + 1)*m, 3] = y_real[i]
        if y_real[i] == 1:
            y_all[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=alpha)
            y_workers[i] = y_all[i*m:(i + 1)*m, 2]
        else:
            y_all[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=1-beta)
            y_workers[i] = y_all[i*m:(i + 1)*m, 2]

    return x, y_workers, y_real, y_all, alpha, beta, None, 0


def generate_points(n, m, d, l,
                    alpha, beta,
                    random_state=0):
    np.random.seed(random_state)

    w = np.random.normal(size=(d,))

    x = np.random.normal(size=(n, d))
    y_all = np.zeros((n*m, 4))

    model_true = np.random.binomial(size=n, n=1, p=l)
    model_answer = (np.squeeze(sigmoid(np.matmul(x, w[:, None]))) > 0.5).astype(int)
    y_real = np.where(model_true, model_answer, 1 - model_answer)

    y_workers = np.zeros((n, m))

    for i in range(n):
        y_all[i*m:(i + 1)*m, 0] = i
        y_all[i*m:(i + 1)*m, 1] = list(range(m))
        y_all[i*m:(i + 1)*m, 3] = y_real[i]
        if y_real[i] == 1:
            y_all[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=alpha)
            y_workers[i] = y_all[i*m:(i + 1)*m, 2]
        else:
            y_all[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=1-beta)
            y_workers[i] = y_all[i*m:(i + 1)*m, 2]

    return x, y_workers, y_real, y_all, alpha, beta, w, l


def create_dfs(x, y_real, y_workers,
                worker_column='worker_id', task_column='task_id', worker_ans_column='response', gold_ans_column='gold'):
    task_pd = pd.DataFrame(np.hstack(
        (np.array(list(range(x.shape[0]))).reshape((x.shape[0], 1)), y_real.reshape((x.shape[0], 1)), x)),
                             columns=[task_column] + [gold_ans_column] + ['{}_feature'.format(i) for i in range(x.shape[1])])
    crowds_pd = pd.DataFrame(y_workers, columns=[task_column, worker_column, worker_ans_column, gold_ans_column])

    return task_pd, crowds_pd
