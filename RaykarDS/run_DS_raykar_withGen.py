import argparse
import pandas as pd
import numpy as np

from em_DSraykar import EM_DS_Raykar
from generate_data import generate_points, create_dfs


def run_ds_reykar(x, y_real, y_workers, l, real_w):

    em_ds_raykar = EM_DS_Raykar(x, y_workers, y_real, l, verbose=True)
    # alpha, beta, w, mu = em_ds_raykar.em_algorithm()
    # print('!!!!!!!!!!!!l={}!!!!!!!!!!!!!'.format(l))
    # print('alpha={}'.format(alpha))
    # print('beta={}'.format(beta))
    # print('w={}'.format(w))
    # print('real_w{}'.format(real_w))
    # print('grad w={}'.format(em_ds_raykar.grad_w(w, mu)))
    # print('Elog={}'.format(np.exp(em_ds_raykar.e_loglikelihood(alpha, beta, w, mu)/x.shape[0])))
    # print("Overall crossentropy: {}".format((-y_real*np.log(mu)).sum()))
    # print("Real crossentropy: {}".format((-y_real*np.log(y_real + 1e-9)).sum()))

    real_mu = y_real

    alpha, beta, w = em_ds_raykar.update_vars(real_w, real_mu)
    print('Ereallog={}'.format(np.exp((em_ds_raykar.e_loglikelihood(alpha, beta, real_w, real_mu))/x.shape[0])))
    mu = em_ds_raykar.update_mu(alpha, beta, real_w)
    print("Error after update mu: {}".format((abs(mu - real_mu)).mean()))
    print("Elog after update mu: {}".format(em_ds_raykar.e_loglikelihood(alpha, beta, real_w, mu)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n',
                        type=int,
                        default=10000,
                        help='Number of points')

    parser.add_argument('-m',
                        type=int,
                        default=2,
                        help='Number of workers')

    parser.add_argument('-d',
                        type=int,
                        default=3,
                        help='Number of task features')

    parser.add_argument('-l',
                        type=int,
                        default=1,
                        help='The quality of model')

    parser.add_argument('--task-column',
                        type=str,
                        default='task_id',
                        help='Name of column with task id')

    parser.add_argument('--worker-column',
                        type=str,
                        default='worker_id',
                        help='Name of column with worker id')

    parser.add_argument('--worker-ans-column',
                        type=str,
                        default='response',
                        help='Name of column with answer of worker')

    parser.add_argument('--gold-ans-column',
                        type=str,
                        default='gold',
                        help='Name of column with golden answer')

    args = parser.parse_args()

    x, y_real, y_workers, y_all, w = generate_points(args.n, args.m, args.d, args.l)

    run_ds_reykar(x, y_real, y_workers, args.l, w)

    task_pd, crowds_pd = create_dfs(x, y_real, y_all,
                args.worker_column, args.task_column, args.worker_ans_column, args.gold_ans_column)

    task_pd.to_csv('generated/crowds.tsv', sep='\t', index=False)
    crowds_pd.to_csv('generated/crowds.tsv', sep='\t', index=False)
