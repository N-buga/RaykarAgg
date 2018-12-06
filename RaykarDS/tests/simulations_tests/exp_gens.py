import numpy as np
import sys

from em_DSraykar import EM_DS_Raykar
from generate_data import generate_AB_points, generate_points


class ExpGens():
    def test_gen(self, generate_method, l, alpha, beta, out, cnt_trials=1):
        print("l = {}".format(l))
        print("l = {}".format(l), file=out)
        print("------------------")

        ls = []

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
            x, y, y_real, y_all, alpha, beta, w = generate_method(100000, alpha.shape[0], 6, l,
                                                                  alpha=alpha,
                                                                  beta=beta,
                                                                  random_state=i)

            em_ds_raykar = EM_DS_Raykar(x, y, y_real=None, l=l, max_steps=max_steps)
            res_alpha_l, res_beta_l, res_w_l, res_mu_l, _ = em_ds_raykar.em_algorithm()
            em_ds_raykar = EM_DS_Raykar(x, y, y_real=None, l=None, max_steps=max_steps, verbose=True)
            res_alpha, res_beta, res_w, res_mu, res_l = em_ds_raykar.em_algorithm()
            ls.append(res_l)

            em_ds_raykar = EM_DS_Raykar(x, y, y_real=None, l=0, max_steps=max_steps)
            res_alpha_DS, res_beta_DS, res_w_DS, res_mu_DS, _ = em_ds_raykar.em_algorithm()
            em_ds_raykar = EM_DS_Raykar(x, y, y_real=None, l=1, max_steps=max_steps)
            res_alpha_raykar, res_beta_raykar, res_w_raykar, res_mu_raykar, _ = em_ds_raykar.em_algorithm()

            diff_w_l.append(((w / w[0] - res_w_l / res_w_l[0]) ** 2).sum())
            diff_w_raykar.append(((w / w[0] - res_w_raykar / res_w_raykar[0]) ** 2).sum())
            diff_w.append(((w / w[0] - res_w / res_w[0]) ** 2).sum())

            diff_alpha_l.append(((alpha - res_alpha_l) ** 2).sum())
            diff_alpha.append(((alpha - res_alpha) ** 2).sum())
            diff_alpha_DS.append(((alpha - res_alpha_DS) ** 2).sum())
            diff_alpha_raykar.append(((alpha - res_alpha_raykar) ** 2).sum())

            diff_beta_l.append(((beta - res_beta_l) ** 2).sum())
            diff_beta.append(((beta - res_beta) ** 2).sum())
            diff_beta_DS.append(((beta - res_beta_DS) ** 2).sum())
            diff_beta_raykar.append(((beta - res_beta_raykar) ** 2).sum())

            acc_l.append((y_real == (res_mu_l > 0.5).astype(int)).sum())
            acc.append((y_real == (res_mu > 0.5).astype(int)).sum())
            acc_DS.append((y_real == (res_mu_DS > 0.5).astype(int)).sum())
            acc_raykar.append((y_real == (res_mu_raykar > 0.5).astype(int)).sum())

        for file in [sys.stdout, out]:
            print("l = {}".format(sum(ls) / cnt_trials), file=file)
            print("l_mean = {}".format((sum(ls) / cnt_trials).mean()), file=file)

            print("W:", file=file)
            print("RaykarDS: {}".format(sum(diff_w) / cnt_trials), file=file)
            print("RaykarDS_l: {}".format(sum(diff_w_l) / cnt_trials), file=file)
            print("Raykar:   {}".format(sum(diff_w_raykar) / cnt_trials), file=file)

            print("alpha:", file=file)
            print("RaykarDS: {}".format(sum(diff_alpha) / cnt_trials), file=file)
            print("RaykarDS_l: {}".format(sum(diff_alpha_l) / cnt_trials), file=file)
            print("DS:       {}".format(sum(diff_alpha_DS) / cnt_trials), file=file)
            print("Raykar:   {}".format(sum(diff_alpha_raykar) / cnt_trials), file=file)

            print("beta:", file=file)
            print("RaykarDS: {}".format(sum(diff_beta) / cnt_trials), file=file)
            print("RaykarDS_l: {}".format(sum(diff_beta_l) / cnt_trials), file=file)
            print("DS:       {}".format(sum(diff_beta_DS) / cnt_trials), file=file)
            print("Raykar:   {}".format(sum(diff_beta_raykar) / cnt_trials), file=file)

            print("Acc:", file=file)
            print("RaykarDS: {}".format(sum(acc) / cnt_trials), file=file)
            print("RaykarDS_l: {}".format(sum(acc_l) / cnt_trials), file=file)
            print("DS:       {}".format(sum(acc_DS) / cnt_trials), file=file)
            print("Raykar:   {}".format(sum(acc_raykar) / cnt_trials), file=file)


if __name__ == '__main__':
    with open("res_exp.txt", "w") as file_to:
        alphas = [np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]), np.array([0.6, 0.7, 0.7, 0.8, 0.9])]
        betas = [np.array([0.6, 0.7, 0.7, 0.8, 0.9, 0.9]), np.array([0.6, 0.7, 0.7, 0.8, 0.9])]

        for method in [generate_points]: #[generate_AB_points, generate_points]:
            print("       METHOD {}".format(method.__name__))
            print("       METHOD {}".format(method.__name__), file=file_to)
            for i in range(len(alphas)):
                print("        alpha={}; beta={}".format(alphas[i], betas[i]))
                print("        alpha={}; beta={}".format(alphas[i], betas[i]), file=file_to)
                ExpGens().test_gen(method, 0, alpha=alphas[i], beta=betas[i], out=file_to)
                ExpGens().test_gen(method, 0.2, alpha=alphas[i], beta=betas[i], out=file_to)
                ExpGens().test_gen(method, 0.5, alpha=alphas[i], beta=betas[i], out=file_to)
                ExpGens().test_gen(method, 0.8, alpha=alphas[i], beta=betas[i], out=file_to)
                ExpGens().test_gen(method, 1, alpha=alphas[i], beta=betas[i], out=file_to)
