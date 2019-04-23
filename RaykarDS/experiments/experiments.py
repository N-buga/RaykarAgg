import json
import time
import numpy as np
import scipy.special

from data_storage.data_storage import DataStorage
from em_DSraykar import EM_DS_Raykar
from models import Model

VERY_BIG_NUMBER = 1.0e9


class Experiments:
    def __init__(self, data: DataStorage):
        self.data = data

    @classmethod
    def run_RaykarDS(cls, list_features: [np.array], list_workers_answers: [np.array], model: Model,
                     l: float = None, max_steps=200):
        """
        Run DSRaykar algorithm for each trial, calculate mean and std.

        :param list_features: list of NxM features of N tasks for each trial.
        :param list_workers_answers: list of NxW answers of workers for each trial.
        :param model: Model to use as classification model in RaykarDS.
        :param l: l in [0, 1]
        :param max_steps: Maximum steps to make in EM algorithm when optimize.
        :return: list of results.
        """

        alphas = []
        betas = []
        ws = []
        p1s = []
        lambdas = []
        likelihoods = []
        times = []

        for i, features in enumerate(list_features):
            workers_answers = list_workers_answers[i]
            beg_l = time.time()

            em_ds_raykar = EM_DS_Raykar(features, workers_answers, model=model, l=l, max_steps=max_steps)
            res_alpha, res_beta, res_w, res_p1, res_l, res_likelihood = em_ds_raykar.em_algorithm()

            times.append(time.time() - beg_l)
            likelihoods.append(res_likelihood)
            lambdas.append(scipy.special.expit(res_l))
            p1s.append(res_p1)
            ws.append(res_w)
            betas.append(res_beta)
            alphas.append(res_alpha)

        return {'alphas': np.array(alphas),
                'betas': np.array(betas),
                'ws': np.array(ws),
                'p1s': np.array(p1s),
                'lambdas': np.array(lambdas),
                'likelihoods': np.array(likelihoods),
                'times': np.array(times) }

    @classmethod
    def run_Raykar(cls, list_features: [np.array], list_workers_answers: [np.array], model: Model,
                   max_steps: int = 200):
        """
        Run Raykar algorithm for each trial, calculate mean and std.

        :param list_features: list of NxM features of N tasks for each trial.
        :param list_workers_answers: list of NxW answers of workers for each trial.
        :param model: Model to use as classification model in RaykarDS.
        :param max_steps: Maximum steps to make in EM algorithm when optimize.
        :return: list of results.
        """

        return cls.run_RaykarDS(list_features, list_workers_answers, model, VERY_BIG_NUMBER, max_steps)

    @classmethod
    def run_DS(cls, list_features: [np.array], list_workers_answers: [np.array], model: Model, max_steps: int = 200):
        """
        Run DS algorithm for each trial, calculate mean and std.

        :param list_features: list of NxM features of N tasks for each trial.
        :param list_workers_answers: list of NxW answers of workers for each trial.
        :param model: Model to use as classification model in RaykarDS.
        :param max_steps: Maximum steps to make in EM algorithm when optimize.
        :return: list of results.
        """

        return cls.run_RaykarDS(list_features, list_workers_answers, model, -VERY_BIG_NUMBER, max_steps)

    def bootstrap_data(self, cnt_trials, size: int = None, marks_percentage: int = None, at_least: int = 1, seeds: list = None):
        """
        Bootstrap tasks for each seed.
        :param cnt_trials: Number of trials.
        :param size: Number of task to choose.
        :param marks_percentage: Percentage of marks to leave.
        :param at_least: Minimum number of marks to leave for each task.
        :param seeds: Seeds for bootstrap. <seeds> == None or len(<seeds>) >= <cnt_trials>
        :return: list[featueres], list[workers_answers], list[true_answers]
        """

        if seeds is None:
            seeds = list(range(cnt_trials))

        list_features, list_worker_answers, list_real_answers = [], [], []

        for seed in seeds:
            features, worker_answers, real_answers = self.data.bootstrap(size, marks_percentage, at_least, seed)
            list_features.append(features)
            list_worker_answers.append(worker_answers)
            list_real_answers.append(real_answers)

        return list_features, list_worker_answers, list_real_answers

    @staticmethod
    def get_statistics(list_real_answers: [np.array], result: dict):
        acc = []
        for i, cur_real_answer in enumerate(list_real_answers):
            cur_p1 = result['p1s'][i]
            index_gold = ~np.isnan(cur_real_answer)
            acc.append((cur_real_answer[index_gold] == (cur_p1[index_gold] > 0.5).astype(int)).sum() / index_gold.sum())

        result['acc'] = np.array(acc)
        result['mean_acc'] = np.mean(result['acc'])
        result['std_acc'] = np.std(result['acc'])
        result['mean_time'] = np.mean(result['times'])
        result['std_time'] = np.std(result['times'])

        def arrays_tolist(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            else:
                return value

        return {k: arrays_tolist(v) for k, v in result.items()}

    def run_experiments(self, boot_params=None, boot_param_descriptions=None,
                        RaykarDS_params=None, RaykarDS_params_descriptions=None,
                        Raykar_params=None, Raykar_params_descriptions=None,
                        DS_params=None, DS_params_descriptions=None,
                        verbose=True, file_to_save='saving.json'):
        """
        Run set od experiments and calculate result.
        :param boot_params: list[dict] -- Dict of params to pass to bootstrap_data
        :param boot_param_descriprions: list[str] -- Output strings if verbose==True and key in result.
        :param RaykarDS_params: list[dict] -- Dict of params to pass to pass to RaykarDS.
        :param RaykarDS_params_descriptions: list[str] -- Output strings if verbose==True and key in result.
        :param Raykar_params: list[dict] -- Dict of params to pass to pass to Raykar.
        :param Raykar_params_descriptions: list[str] -- Output strings if verbose==True and key in result.
        :param DS_params: list[dict] -- Dict of params to pass to pass to DS.
        :param DS_params_descriptions: list[str] -- Output strings if verbose==True and key in result.
        :return:
        """

        result = {}

        if boot_params is None:
            boot_params = []

        if RaykarDS_params is None:
            RaykarDS_params = []

        if Raykar_params is None:
            Raykar_params = []

        if DS_params is None:
            DS_params = []

        for i, boot_param in enumerate(boot_params):
            cur_boot_descr = boot_param_descriptions[i]
            if verbose:
                print(cur_boot_descr)

            result[cur_boot_descr] = {}
            list_features, list_worker_answers, list_real_answers = self.bootstrap_data(**boot_param)
            for j, RaykarDS_param in enumerate(RaykarDS_params):
                cur_descr = RaykarDS_params_descriptions[j]
                if verbose:
                    print(cur_descr)
                RaykarDS_result = self.run_RaykarDS(list_features, list_worker_answers, **RaykarDS_param)
                result[cur_boot_descr][cur_descr] = self.get_statistics(list_real_answers, RaykarDS_result)

                with open(file_to_save, 'w') as file_to:
                    json.dump(result, file_to)

                with open('stat' + file_to_save, 'w') as file_to:
                    json.dump(result, file_to)

            for j, Raykar_param in enumerate(Raykar_params):
                cur_descr = Raykar_params_descriptions[j]
                if verbose:
                    print(cur_descr)
                Raykar_result = self.run_Raykar(list_features, list_worker_answers, **Raykar_param)
                result[cur_boot_descr][cur_descr] = self.get_statistics(list_real_answers, Raykar_result)

                with open(file_to_save, 'w') as file_to:
                    json.dump(result, file_to)

            for j, DS_param in enumerate(DS_params):
                cur_descr = DS_params_descriptions[j]
                if verbose:
                    print(cur_descr)
                DS_result = self.run_DS(list_features, list_worker_answers, **DS_param)
                result[cur_boot_descr][cur_descr] = self.get_statistics(list_real_answers, DS_result)

                with open(file_to_save, 'w') as file_to:
                    json.dump(result, file_to)

        return result
