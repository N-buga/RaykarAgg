import argparse
import pandas as pd
import numpy as np

from em import EM_DS_Raykar


def run_ds_reykar(crowd_data, tasks_data, worker_column, task_column, worker_ans_column, gold_ans_column):
    cnt_answers_per_task = crowd_data[[worker_column, task_column]].groupby(task_column).count()

    if not (cnt_answers_per_task.iloc[:, 0].eq(cnt_answers_per_task.iloc[0, 0])).all():
        raise AttributeError('Different count of answers for tasks')

    diff_answers = crowd_data[gold_ans_column].unique()
    if diff_answers.shape[0] != 2:
        raise AttributeError('No binary answers')

    x = tasks_data.iloc[:, 2:].values
    y = crowd_data[[task_column, worker_ans_column]].groupby(task_column).apply(
        lambda g: pd.Series(g.iloc[:, 1].tolist())
    ).values
    y_real = np.squeeze(crowd_data[[task_column, gold_ans_column]].groupby(task_column).first().values)

    # for l in np.arange(0.1, 0.9, 0.1):
    l = 0.3
    em_ds_raykar = EM_DS_Raykar(x, y, l, verbose=True)
    alpha, beta, w, mu = em_ds_raykar.em_algorithm()
    print('!!!!!!!!!!!!l={}!!!!!!!!!!!!!'.format(l))
    print('alpha={}'.format(alpha))
    print('beta={}'.format(beta))
    print('w={}'.format(w))
    print("P real = {}".format(np.where(y_real == 1, mu, 1 - mu)))
    print('loglik={}'.format(em_ds_raykar.e_loglikelihood(em_ds_raykar.a(alpha), em_ds_raykar.b(beta), w, mu)))

    # real_alpha = 0.3*np.ones((alpha.shape[0],))
    # real_beta = 0.6*np.ones((alpha.shape[0],))
    # a = em_ds_raykar.a(real_alpha)
    # b = em_ds_raykar.b(real_beta)
    # real_w = np.array([1, 2, 3])
    # real_mu = em_ds_raykar.update_mu(a, b, real_w)
    # print('Real logik={}'.format(em_ds_raykar.e_loglikelihood(a, b, real_w, real_mu)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--crowd-filepath',
                        type=str,
                        metavar='PATH',
                        default='generated/crowd.tsv',
                        help='Path to tsv file with crowdsourced labels')

    parser.add_argument('-tf', '--task-filepath',
                        type=str,
                        metavar='PATH',
                        default='generated/tasks.tsv',
                        help='Path to tsv file with tasks descriptions')

    parser.add_argument('--worker-column',
                        type=str,
                        default='worker_id',
                        help='Name of column with worker id')

    parser.add_argument('--task-column',
                        type=str,
                        default='task_id',
                        help='Name of column with task id')

    parser.add_argument('--worker-ans-column',
                        type=str,
                        default='response',
                        help='Name of column with answer of worker')

    parser.add_argument('--gold-ans-column',
                        type=str,
                        default='gold',
                        help='Name of column with golden answer')

    args = parser.parse_args()

    crowd_data = pd.read_csv(args.crowd_filepath, sep='\t')
    tasks_data = pd.read_csv(args.task_filepath, sep='\t')

    run_ds_reykar(crowd_data, tasks_data,
                  args.worker_column, args.task_column, args.worker_ans_column, args.gold_ans_column)
