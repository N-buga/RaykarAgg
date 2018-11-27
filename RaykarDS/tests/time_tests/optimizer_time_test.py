import argparse
import os

import pandas as pd
import numpy as np
import time
from em_DSraykar import EM_DS_Raykar
from generate_data import generate_model_points, generate_DS_points
from optimizers import GradientDescentOptimizer, NewtonOptimizer, ScipyOptimizer


class OptimizerTimeTest:
    def __init__(self, dir, all_out_file_suffix, result_out_file_suffix):
        self.all_out_file_suffix = all_out_file_suffix
        self.result_out_file_suffix = result_out_file_suffix
        self.dir = dir

    def check_generation(self, x, y, w, alpha, beta, y_real, l, optimizer, acc, mu_exact_const, mu_ans_const):
        em_ds_raykar = EM_DS_Raykar(x, y, y_real=None, l=l, optimizer=optimizer)

        beg = time.time()
        res_alpha, res_beta, res_w, res_mu = em_ds_raykar.em_algorithm()
        end = time.time()

        if w is not None:
            if (np.abs(w / w[0] - res_w / res_w[0]) < acc).all():
                print("\t\tPassed #0: w is all close")
            else:
                print("\t\tWrong #0: w is too far")
                return end - beg, False
        if alpha is not None:
            if (np.abs(alpha - res_alpha) < acc).all():
                print("\t\tPassed #1: alpha is all close")
            else:
                print("\t\tWrong #1: alpha is too far")
                return end - beg, False
        if beta is not None:
            if (np.abs(beta - res_beta) < acc).all():
                print("\t\tPassed #2: beta is all close")
            else:
                print("\t\tWrong #2: beta is too far")
                return end - beg, False
        if y_real is not None:
            if (np.abs(y_real - res_mu) < acc).sum() > mu_exact_const * x.shape[0]:
                print("\t\tPassed #3: mu is quite accurate")
            else:
                print("\t\tWrong #3: mu is not quite accurate")
                return end - beg, False
            if (y_real == (res_mu > 0.5).astype(int)).sum() > mu_ans_const * x.shape[0]:
                print("\t\tPassed #4: res y is quite accurate")
            else:
                print("\t\tWrong #4: res y is not quite accurate")
                return end - beg, False

        return end - beg, True

    def test_raykar(self):
        out_file_prefix = 'raykar'

        x, y, y_real, y_all, alpha, beta, w = generate_model_points(50000, 5, 6,
                                                                    alpha=np.array([0.6, 0.7, 0.7, 0.8, 0.9]),
                                                                    beta=np.array([0.6, 0.7, 0.7, 0.8, 0.9]))

        cnt_trials = 10

        optimizers = [
            GradientDescentOptimizer(step=0.01, steps_count=50),
            GradientDescentOptimizer(step=0.1, steps_count=50),
            GradientDescentOptimizer(step=0.5, steps_count=100),
            NewtonOptimizer(step=0.01, steps_count=15),
            NewtonOptimizer(step=0.01, steps_count=10),
            NewtonOptimizer(step=0.05, steps_count=15),
            # ScipyOptimizer(method='CG'),
            # ScipyOptimizer(method='Newton-CG'),
            # ScipyOptimizer(method='BFGS'),
            # ScipyOptimizer(method='trust-NCG')
        ]

        index = ["Trial #{}".format(trial) for trial in list(range(cnt_trials))]
        columns = [opt.description() for opt in optimizers]

        times = pd.DataFrame(np.zeros((cnt_trials, len(optimizers))), index=index, columns=columns)
        check_result = pd.DataFrame(np.full((cnt_trials, len(optimizers)), True, dtype=bool), index=index,
                                    columns=columns)

        for i in range(cnt_trials):
            print("Trial {}".format(i))
            for j, optimizer in enumerate(optimizers):
                print("\tOptimizer {}".format(optimizer.description()))
                times.iloc[i, j], check_result.iloc[i, j] = self.check_generation(x, y, w, alpha, beta, y_real, 1,
                                                                                  acc=1e-2, optimizer=optimizer,
                                                                                  mu_exact_const=0.95,
                                                                                  mu_ans_const=0.99)
                print("\tTime: {}".format(times.iloc[i, j]))
                print("\tCheck result: {}".format(check_result.iloc[i, j]))

        times.mean(axis=0)

        with open(os.path.join(self.dir, out_file_prefix + self.all_out_file_suffix), 'w') as file_to:
            print("Times:", file=file_to)
            times.to_csv(file_to, sep='\t')
            print("Check result:", file=file_to)
            check_result.to_csv(file_to, sep='\t')

        with open(os.path.join(self.dir, out_file_prefix + self.result_out_file_suffix), 'w') as file_to:
            print("Mean times:", file=file_to)
            times.mean(axis=0).to_csv(file_to, sep='\t')
            print("Check result:", file=file_to)
            times.any(axis=0).to_csv(file_to, sep='\t')

    def test_DS(self):
        out_file_prefix = 'DS'

        x, y, y_real, y_all, alpha, beta, w = generate_DS_points(100000, 6, 6,
                                                                     alpha=np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]),
                                                                     beta=np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]))

        cnt_trials = 10

        optimizers = [
            GradientDescentOptimizer(step=0.01, steps_count=50),
            GradientDescentOptimizer(step=0.1, steps_count=50),
            GradientDescentOptimizer(step=0.1, steps_count=100),
            NewtonOptimizer(step=0.01, steps_count=15),
        ]

        index = ["Trial #{}".format(trial) for trial in list(range(cnt_trials))]
        columns = [opt.description() for opt in optimizers]

        times = pd.DataFrame(np.zeros((cnt_trials, len(optimizers))), index=index, columns=columns)
        check_result = pd.DataFrame(np.full((cnt_trials, len(optimizers)), True, dtype=bool), index=index,
                                    columns=columns)

        for i in range(cnt_trials):
            print("Trial {}".format(i))
            for j, optimizer in enumerate(optimizers):
                print("\tOptimizer {}".format(optimizer.description()))
                times.iloc[i, j], check_result.iloc[i, j] = self.check_generation(x, y, None, alpha, beta, y_real, 0,
                                                                                  optimizer=optimizer, acc=2e-2,
                                                                                  mu_exact_const=0.75,
                                                                                  mu_ans_const=0.95)
                print("\tTime: {}".format(times.iloc[i, j]))
                print("\tCheck result: {}".format(check_result.iloc[i, j]))

        times.mean(axis=0)

        with open(os.path.join(self.dir, out_file_prefix + self.all_out_file_suffix), 'w') as file_to:
            print("Times:", file=file_to)
            times.to_csv(file_to, sep='\t')
            print("Check result:", file=file_to)
            check_result.to_csv(file_to, sep='\t')

        with open(os.path.join(self.dir, out_file_prefix + self.result_out_file_suffix), 'w') as file_to:
            print("Mean times:", file=file_to)
            times.mean(axis=0).to_csv(file_to, sep='\t')
            print("Check result:", file=file_to)
            times.any(axis=0).to_csv(file_to, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--all-out',
                        default='all-time-test.tsv',
                        help='Output file for all statistics.')

    parser.add_argument('--result-out',
                        default='result-time-test.tsv',
                        help='Output file for result statistics')

    parser.add_argument('--dir',
                        default='.',
                        help='Directory for output')

    args = parser.parse_args()

    time_test = OptimizerTimeTest(args.dir, args.all_out, args.result_out)
    time_test.test_raykar()
    time_test.test_DS()
