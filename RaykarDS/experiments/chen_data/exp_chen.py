import json

import numpy as np
import sys

import time

from em_DSraykar import EM_DS_Raykar
from func_optimizers import AdaGradOptimizer
from models import LogisticRegressionModel
from transform_chen_data import ChenData


class ExpGens():
    def test_gen(self, model, data, l, out, log_file, boot_size=None, marks_percentage=None, cnt_trials=5, eval_raykards=True):
        print("l = {}".format(l))
        print("l = {}".format(l), file=out)
        print("l = {}".format(l), file=log_file)
        # print("--------------")
        # print("--------------", file=out)
        # print("--------------", file=log_file)

        ls = []

        times = []
        times_l = []

        alpha_l = []
        alpha = []

        beta_l = []
        beta = []

        acc_l = []
        acc = []

        max_steps = 100

        out.flush()
        log_file.flush()
        sys.stdout.flush()
        sys.stderr.flush()

        for i in range(cnt_trials):
            print(f"Trial #{i}", end=' ')
            sys.stdout.flush()

            x, y, y_real = data.bootstrap(size=boot_size, marks_percentage=marks_percentage, seed=i)

            beg_l = time.time()
            em_ds_raykar_l = EM_DS_Raykar(x, y, y_real=y_real, model=model, l=l, max_steps=max_steps)
            res_alpha_l, res_beta_l, res_w_l, res_p1_l, _ = em_ds_raykar_l.em_algorithm()
            times_l.append(time.time() - beg_l)
            alpha_l.append(res_alpha_l)
            beta_l.append(res_beta_l)

            if eval_raykards:
                beg = time.time()
                em_ds_raykar = EM_DS_Raykar(x, y, y_real=y_real, model=model, l=None, max_steps=max_steps)
                res_alpha, res_beta, res_w, res_p1, res_l = em_ds_raykar.em_algorithm()
                ls.append(res_l)
                times.append(time.time() - beg)
                alpha.append(res_alpha)
                beta.append(res_beta)

            index_gold = ~np.isnan(y_real)
            acc_l.append(
                (y_real[index_gold] == (res_p1_l[index_gold] > 0.5).astype(int)).sum() / y_real[index_gold].shape[0])

            if eval_raykards:
                acc.append(
                    (y_real[index_gold] == (res_p1[index_gold] > 0.5).astype(int)).sum() / y_real[index_gold].shape[0])

        print()

        ans = {
            'acc':
                {
                    'RaykarDS': sum(acc) / cnt_trials,
                    'RaykarDS_l': sum(acc_l) / cnt_trials,
                },
            'time':
                {
                    'RaykarDS': sum(times) / cnt_trials,
                    'RaykarDS_l': sum(times_l) / cnt_trials,
                }
        }

        sys.stdout.flush()
        sys.stderr.flush()

        for file in [sys.stdout, out, log_file]:
            print("TIMES:", file=file)
            if eval_raykards:
                print("time: {}".format(ans['time']['RaykarDS']), file=file)
            print("times_l: {}".format(ans['time']['RaykarDS_l']), file=file)

        for file in [log_file]:
            if eval_raykards:
                print("l = {}".format(sum(ls) / cnt_trials), file=file)
                print("l_mean = {}".format((sum(ls) / cnt_trials).mean()), file=file)

        for file in [sys.stdout, out, log_file]:
            print("Acc:", file=file)
            if eval_raykards:
                print("RaykarDS: {}".format(ans['acc']['RaykarDS']), file=file)
            print("RaykarDS_l: {}".format(ans['acc']['RaykarDS_l']), file=file)

            file.flush()

        return ans


if __name__ == '__main__':
    file_name = 'r_adagrad2'
    data_path = '../../data/wsdm.csv'

    with open(f"{file_name}.txt", 'w') as file_to:
        with open(f"{file_name}.log", 'w') as log_file:
            lambdas = [1] #[0, 1]  # , 0.8, 0.5, 0.2, 0]
            ans = {}
            ans_mf = {'Rakar': {},
                      'DS': {},
                      'RaykarDS': {}
                      }

            try:
                coeffs = [0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 15]
                eps = 1e-5
                percentage_of_marks = [15, 20, 30, 40, 60, 80, 100]

                for reg_type in ['lasso', 'ridge']:
                    for reg_coeff in coeffs:
                        for percent in percentage_of_marks:
                            print("___________________")
                            print("reg_type={}".format(reg_type))
                            print("reg_type={}".format(reg_type), file=file_to)
                            print("reg_coeff={}".format(reg_coeff))
                            print("reg_coeff={}".format(reg_coeff), file=file_to)
                            print("percentage={}".format(percent))
                            print("percentage={}".format(percent), file=file_to)
                            print("eps={}".format(eps))
                            print("eps={}".format(eps), file=file_to)

                            data = ChenData(data_path, max_features=None)
                            data.transform_points()

                            ans['{}_{}_{}_{}'.format(reg_type, reg_coeff, percent, eps)] = {}

                            if reg_coeff == coeffs[0]:
                                cur_lambdas = [0, 1]
                            else:
                                cur_lambdas = [1]

                            for l in cur_lambdas:
                                if l != 1:
                                    eval_raykards = False
                                else:
                                    eval_raykards = True

                                res = ExpGens().test_gen(
                                    LogisticRegressionModel(reg_coeff=reg_coeff, reg_type=reg_type,
                                                            optimizer=AdaGradOptimizer(eps=eps)),
                                    data, l, out=file_to,
                                    boot_size=data.X.shape[0],
                                    marks_percentage=percent,
                                    eval_raykards=eval_raykards,
                                    log_file=log_file
                                )

                                ans['{}_{}_{}_{}'.format(reg_type, reg_coeff, percent, eps)][l] = res

            finally:
                with open(f"{file_name}.json", 'w') as file_to:
                    json.dump(ans, file_to)
