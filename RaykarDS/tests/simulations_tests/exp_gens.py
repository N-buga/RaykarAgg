import json

import numpy as np
import sys

import time

from em_DSraykar import EM_DS_Raykar
from models import LogisticRegressionModel
from generate_data import generate_AB_points, generate_points


class ExpGens():
    def test_gen(self, model, generate_method, l, alpha, beta, out, cnt_trials=1):
        print("l = {}".format(l))
        print("l = {}".format(l), file=out)
        print("------------------")

        ls = []

        times = []
        times_l = []
        times_raykar = []
        times_DS = []

        diff_w_l = []
        diff_w = []
        diff_w_raykar = []

        diff_alpha_l = []
        diff_alpha = []
        diff_alpha_raykar = []
        diff_alpha_DS = []

        diff_beta_l = []
        diff_beta = []
        diff_beta_raykar = []
        diff_beta_DS = []

        acc_l = []
        acc = []
        acc_raykar = []
        acc_DS = []

        max_steps = 100

        for i in range(cnt_trials):
            print("Trial #{}".format(i))
            x, y, y_real, y_all, alpha, beta, w, real_l = generate_method(100000, alpha.shape[0], 6, l,
                                                                          alpha=alpha,
                                                                          beta=beta,
                                                                          random_state=i)

            beg_l = time.time()
            em_ds_raykar_l = EM_DS_Raykar(x, y, y_real=y_real, model=model, l=real_l, max_steps=max_steps)
            res_alpha_l, res_beta_l, res_w_l, res_p1_l, _ = em_ds_raykar_l.em_algorithm()
            times_l.append(time.time() - beg_l)

            beg = time.time()
            em_ds_raykar = EM_DS_Raykar(x, y, y_real=y_real, model=model, l=None, max_steps=max_steps)
            res_alpha, res_beta, res_w, res_mu, res_p1 = em_ds_raykar.em_algorithm()
            ls.append(res_p1)
            times.append(time.time() - beg)

            beg_DS = time.time()
            em_ds_raykar_DS = EM_DS_Raykar(x, y, y_real=y_real, model=model, l=0, max_steps=max_steps)
            res_alpha_DS, res_beta_DS, res_w_DS, res_p1_DS, _ = em_ds_raykar_DS.em_algorithm()
            times_DS.append(time.time() - beg_DS)

            beg_raykar = time.time()
            em_ds_raykar_raykar = EM_DS_Raykar(x, y, y_real=y_real, model=model, l=1, max_steps=max_steps)
            res_alpha_raykar, res_beta_raykar, res_w_raykar, res_p1_raykar, _ = em_ds_raykar_raykar.em_algorithm()
            times_raykar.append(time.time() - beg_raykar)

            diff_w_l.append(((w / w[0] - res_w_l / res_w_l[0]) ** 2).sum() / w.shape[0])
            diff_w_raykar.append(((w / w[0] - res_w_raykar / res_w_raykar[0]) ** 2).sum() / w.shape[0])
            diff_w.append(((w / w[0] - res_w / res_w[0]) ** 2).sum() / w.shape[0])

            diff_alpha_l.append(((alpha - res_alpha_l) ** 2).sum() / alpha.shape[0])
            diff_alpha.append(((alpha - res_alpha) ** 2).sum() / alpha.shape[0])
            diff_alpha_DS.append(((alpha - res_alpha_DS) ** 2).sum() / alpha.shape[0])
            diff_alpha_raykar.append(((alpha - res_alpha_raykar) ** 2).sum() / alpha.shape[0])

            diff_beta_l.append(((beta - res_beta_l) ** 2).sum() / beta.shape[0])
            diff_beta.append(((beta - res_beta) ** 2).sum() / beta.shape[0])
            diff_beta_DS.append(((beta - res_beta_DS) ** 2).sum() / beta.shape[0])
            diff_beta_raykar.append(((beta - res_beta_raykar) ** 2).sum() / beta.shape[0])

            acc_l.append((y_real == (res_p1_l > 0.5).astype(int)).sum() / y_real.shape[0])
            acc.append((y_real == (res_mu > 0.5).astype(int)).sum() / y_real.shape[0])
            acc_DS.append((y_real == (res_p1_DS > 0.5).astype(int)).sum() / y_real.shape[0])
            acc_raykar.append((y_real == (res_p1_raykar > 0.5).astype(int)).sum() / y_real.shape[0])

        ans = {
            'acc':
                {
                    'RaykarDS': sum(acc) / cnt_trials,
                    'RaykarDS_l': sum(acc_l) / cnt_trials,
                    'DS': sum(acc_DS) / cnt_trials,
                    'Raykar': sum(acc_raykar) / cnt_trials
                },
            'alpha':
                {
                    'RaykarDS': sum(diff_alpha) / cnt_trials,
                    'RaykarDS_l': sum(diff_alpha_l) / cnt_trials,
                    'DS': sum(diff_alpha_DS) / cnt_trials,
                    'Raykar': sum(diff_alpha_raykar) / cnt_trials
                },
            'beta':
                {
                    'RaykarDS': sum(diff_beta) / cnt_trials,
                    'RaykarDS_l': sum(diff_beta_l) / cnt_trials,
                    'DS': sum(diff_beta_DS) / cnt_trials,
                    'Raykar': sum(diff_beta_raykar) / cnt_trials
                },
            'time':
                {
                    'RaykarDS': sum(times) / cnt_trials,
                    'RaykarDS_l': sum(times_l) / cnt_trials,
                    'DS': sum(times_DS) / cnt_trials,
                    'Raykar': sum(times_raykar) / cnt_trials
                }
        }

        for file in [sys.stdout, out]:
            print("real_l={}".format(real_l))
            print("real_l={}".format(real_l), file=file)

            print("TIMES:", file=file)
            print("time: {}".format(ans['time']['RaykarDS']), file=file)
            print("times_l: {}".format(ans['time']['RaykarDS_l']), file=file)
            print("times_DS: {}".format(ans['time']['DS']), file=file)
            print("times_raykar: {}".format(ans['time']['Raykar']), file=file)

            print("l = {}".format(sum(ls) / cnt_trials), file=file)
            print("l_mean = {}".format((sum(ls) / cnt_trials).mean()), file=file)

            print("W:", file=file)
            print("RaykarDS: {}".format(sum(diff_w) / cnt_trials), file=file)
            print("RaykarDS_l: {}".format(sum(diff_w_l) / cnt_trials), file=file)
            print("Raykar:   {}".format(sum(diff_w_raykar) / cnt_trials), file=file)

            print("alpha:", file=file)
            print("RaykarDS: {}".format(ans['alpha']['RaykarDS']), file=file)
            print("RaykarDS_l: {}".format(ans['alpha']['RaykarDS_l']), file=file)
            print("DS:       {}".format(ans['alpha']['DS']), file=file)
            print("Raykar:   {}".format(ans['beta']['Raykar']), file=file)

            print("beta:", file=file)
            print("RaykarDS: {}".format(ans['beta']['RaykarDS']), file=file)
            print("RaykarDS_l: {}".format(ans['beta']['RaykarDS_l']), file=file)
            print("DS:       {}".format(ans['beta']['DS']), file=file)
            print("Raykar:   {}".format(ans['beta']['Raykar']), file=file)

            print("Acc:", file=file)
            print("RaykarDS: {}".format(ans['acc']['RaykarDS']), file=file)
            print("RaykarDS_l: {}".format(ans['acc']['RaykarDS_l']), file=file)
            print("DS:       {}".format(ans['acc']['DS']), file=file)
            print("Raykar:   {}".format(ans['acc']['Raykar']), file=file)

        return ans, real_l


if __name__ == '__main__':
    file_name = 'res_test_01'

    with open(f"{file_name}.txt", 'w') as file_to:
        alphas = [np.array([0.6, 0.6, 0.6, 0.7, 0.7]),
                  np.array([0.6, 0.7, 0.7, 0.7, 0.8]),
                  np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]),
                  np.array([0.6, 0.7, 0.7, 0.8, 0.9])]
        betas = [np.array([0.6, 0.6, 0.6, 0.7, 0.7]),
                 np.array([0.6, 0.7, 0.7, 0.7, 0.8]),
                 np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]),
                 np.array([0.6, 0.7, 0.7, 0.8, 0.9])]

        lambdas = [1, 0.8, 0.5, 0.2, 0]
        ans = {}

        try:
            for method in [generate_AB_points, generate_points]:
                method_name = "METHOD {}".format(method.__name__)
                print("       " + method_name)
                print("       " + method_name, file=file_to)
                ans[method_name] = {}
                for i in range(0, len(alphas)):
                    ans[method_name][str((alphas[i].tolist(), betas[i].tolist()))] = {}
                    print("        alpha={}; beta={}".format(alphas[i], betas[i]))
                    print("        alpha={}; beta={}".format(alphas[i], betas[i]), file=file_to)
                    for l in lambdas:
                        res, real_l = ExpGens().test_gen(LogisticRegressionModel(), method, l, alpha=alphas[i], beta=betas[i], out=file_to)
                        ans[method_name][str((alphas[i].tolist(), betas[i].tolist()))][real_l] = res
        finally:
            with open(f"{file_name}.json", 'w') as file_to:
                json.dump(ans, file_to)
