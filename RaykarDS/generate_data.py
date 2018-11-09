import os

import numpy as np
import pandas as pd
from em import sigmoid


def generate_A_points(n, m, d,
                    alpha, beta, w, l,
                    low=-20, high=20):
    np.random.seed(0)
    x = np.random.uniform(low, high, (n, d))
    y_real = np.where(np.matmul(x, w) > 0.5, np.ones((n,)), np.zeros((n,)))
    y_workers = np.zeros((n*m, 4))

    for i in range(n):
        y_workers[i*m:(i + 1)*m, 0] = i
        y_workers[i*m:(i + 1)*m, 1] = list(range(m))
        y_workers[i*m:(i + 1)*m, 3] = y_real[i]
        if y_real[i] == 1:
            y_workers[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=alpha)
        else:
            y_workers[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=1-beta)

    return x, y_real, y_workers


def generate_B_points(n, m, d,
                    alpha, beta, w, l,
                    low=-20, high=20):
    np.random.seed(0)
    x = np.random.uniform(low, high, (n, d))
    y_real = np.random.binomial(size=n, n=1, p=0.4)
    y_workers = np.zeros((n*m, 4))

    for i in range(n):
        y_workers[i*m:(i + 1)*m, 0] = i
        y_workers[i*m:(i + 1)*m, 1] = list(range(m))
        y_workers[i*m:(i + 1)*m, 3] = y_real[i]
        if y_real[i] == 1:
            y_workers[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=alpha)
        else:
            y_workers[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=1-beta)

    return x, y_real, y_workers


def generate_AB_points(n, m, d,
                    alpha, beta, w, l,
                    low=-20, high=20):
    np.random.seed(0)
    x = np.random.uniform(low, high, (n, d))
    model = np.random.binomial(size=n, n=1, p=l)
    y_real_A = np.where(np.matmul(x, w) > 0.5, np.ones((n,)), np.zeros((n,)))
    y_real_B = np.random.binomial(size=n, n=1, p=0.4)
    y_real = np.where(model == 1, y_real_A, y_real_B)
    y_workers = np.zeros((n*m, 4))

    for i in range(n):
        y_workers[i*m:(i + 1)*m, 0] = i
        y_workers[i*m:(i + 1)*m, 1] = list(range(m))
        y_workers[i*m:(i + 1)*m, 3] = y_real[i]
        if y_real[i] == 1:
            y_workers[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=alpha)
        else:
            y_workers[i*m:(i + 1)*m, 2] = np.random.binomial(size=m, n=1, p=1-beta)

    return x, y_real, y_workers


def generate_points(n, m, d, l):
    np.random.seed(0)

    u = np.random.normal(size=(m, d))
    w = np.random.normal(size=(d,))
    print('w={}'.format(w))

    x = np.random.normal(size=(n, d))
    y_workers = np.zeros((n*m, 4))

    model_true = np.random.binomial(size=n, n=1, p=l)
    model_answer = np.where(model_true, (np.squeeze(sigmoid(np.matmul(x, w[:, None]))) > 0.5).astype(int),
                            (np.squeeze(sigmoid(np.matmul(x, w[:, None]))) < 0.5).astype(int))
    workers_true = np.random.binomial(size=(n, m), n=1, p=sigmoid(np.matmul(x, np.transpose(u))))

    workers_labels = np.transpose(np.where(np.logical_xor(model_true, np.transpose(workers_true)),
                              1 - model_answer, model_answer))
    y_workers[:, 2] = workers_labels.flatten()

    y_real = np.where(model_true, model_answer, 1 - model_answer)
    for i in range(n):
        y_workers[i*m:(i + 1)*m, 0] = i
        y_workers[i*m:(i + 1)*m, 1] = list(range(m))
        y_workers[i*m:(i + 1)*m, 3] = y_real[i]

    return x, y_real, y_workers


def save_points(x, y_real, y_workers, crowd_file, task_file,
                worker_column='worker_id', task_column='task_id', worker_ans_column='response', gold_ans_column='gold'):
    task_pd = pd.DataFrame(np.hstack(
        (np.array(list(range(x.shape[0]))).reshape((x.shape[0], 1)), y_real.reshape((x.shape[0], 1)), x)),
                             columns=[task_column] + [gold_ans_column] + ['{}_feature'.format(i) for i in range(x.shape[1])])
    crowds_pd = pd.DataFrame(y_workers, columns=[task_column, worker_column, worker_ans_column, gold_ans_column])

    task_pd.to_csv(task_file, sep='\t', index=False)
    crowds_pd.to_csv(crowd_file, sep='\t', index=False)

if __name__ == '__main__':
    # x, y_real, y_workers = generate_A_points(100, 10, 3, 0.3, 0.6, np.array([1, 2, 3]), 0.55)
    # save_points(x, y_real, y_workers, 'generated/crowd_A.tsv', 'generated/tasks_A.tsv')
    #
    # x, y_real, y_workers = generate_B_points(100, 10, 3, 0.3, 0.6, np.array([1, 2, 3]), 0.55)
    # save_points(x, y_real, y_workers, 'generated/crowd_B.tsv', 'generated/tasks_B.tsv')
    #
    # x, y_real, y_workers = generate_AB_points(100, 10, 3, 0.3, 0.6, np.array([1, 2, 3]), 0.55)
    # save_points(x, y_real, y_workers, 'generated/crowd_AB.tsv', 'generated/tasks_AB.tsv')
    #
    x, y_real, y_workers = generate_points(2, 4, 3, 0.7)
    save_points(x, y_real, y_workers, 'generated/crowd.tsv', 'generated/tasks.tsv')