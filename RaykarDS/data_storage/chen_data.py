"""
Описание формата: http://www-personal.umich.edu/~kevynct/datasets/wsdm_crowdflower_2013_columns.txt
Данные: http://www-personal.umich.edu/~kevynct/datasets/wsdm_rankagg_2013_readability_crowdflower_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from data_storage.bm25_transformer import Bm25Transformer
from data_storage.data_storage import DataStorage


class ChenData(DataStorage):
    def __init__(self, file: str, max_features: int = None, workers_per_task: int = None, golden_boarder: int = None):
        """

        :param file: The name of file with data.
        :param max_features: Maximum number of most popular features(words) to choose from all.
        :param workers_per_task: Number of workers with answer to leave for each task.
        :param golden_boarder: If task have more marks than this number it interprets as golden task with golden mark
        equal to MV aggregation.
        """
        self.workers_per_task = workers_per_task
        self.golden_boarder = golden_boarder
        self.max_features = max_features

        self.row_data = pd.read_csv(file)
        self.row_data = self.row_data[self.row_data['please_decide_which_passage_is_more_difficult_gold'] != "Passage B is more difficult.\nI don't know or can't " \
                                                                                              "decide."]

        index_workers_0 = self.row_data['please_decide_which_passage_is_more_difficult'] == 'Passage A is more difficult.'
        index_workers_1 = self.row_data['please_decide_which_passage_is_more_difficult'] == 'Passage B is more difficult.'
        index_workers_nan = self.row_data['please_decide_which_passage_is_more_difficult'] == "I don't know or can't " \
                                                                                              "decide."

        index_real_0 = self.row_data['please_decide_which_passage_is_more_difficult_gold'] == 'Passage A is more difficult.'
        index_real_1 = self.row_data['please_decide_which_passage_is_more_difficult_gold'] == 'Passage B is more difficult.'

        self.row_data.loc[index_workers_0, 'please_decide_which_passage_is_more_difficult'] = 0
        self.row_data.loc[index_workers_1, 'please_decide_which_passage_is_more_difficult'] = 1
        self.row_data.loc[index_workers_nan, 'please_decide_which_passage_is_more_difficult'] = np.nan

        self.row_data.loc[index_real_0, 'please_decide_which_passage_is_more_difficult_gold'] = 0
        self.row_data.loc[index_real_1, 'please_decide_which_passage_is_more_difficult_gold'] = 1

        self.workers_id = self.row_data[['_worker_id']]\
            .drop_duplicates().reset_index(drop=True)\
            .reset_index()\
            .set_index('_worker_id')

        self.tasks_id = self.row_data[['_unit_id']]\
            .drop_duplicates()\
            .reset_index()\
            .set_index(['_unit_id'])

        if self.golden_boarder is None:
            self.golden_boarder = self.workers_id.shape[0] + 10

        if self.workers_per_task is None:
            self.workers_per_task = self.workers_id.shape[0] + 10


    @staticmethod
    def create_features(texts, tasks_a, tasks_b, max_features: int = None):
        """
        Create features from texts.
        :param texts: Array of texts to create set of words.
        :param tasks_a: Array of texts of task A.
        :param tasks_b: Array of texts of task B.
        :param max_features: Maximum number of most popular features(words) to choose from all.
        :return: BM25 embeddings of texts.
        """
        bm25_transformer = Pipeline([
            ('vect', CountVectorizer(max_features=max_features, stop_words='english')),
            ('bm25', Bm25Transformer())
        ]).fit(texts)

        bm25_A = bm25_transformer.transform(tasks_a).toarray()
        bm25_B = bm25_transformer.transform(tasks_b).toarray()

        return np.concatenate([bm25_A, bm25_B, np.ones((tasks_a.shape[0], 1))], axis=1)

    def transform_points(self, seed: int = 0):
        """
        Transform text A and text B to features.

        :param seed: Seed to use when choose workers for tasks (to leave no more then self.workers_per_task)
        :return:
        """
        tasks_groupby = self.row_data.groupby(['_unit_id'])
        tasks = tasks_groupby.first()

        vocab = np.concatenate([np.unique(tasks['passage_a'].values), np.unique(tasks['passage_b'].values)])
        self.X = \
            self.create_features(vocab, tasks['passage_a'].values, tasks['passage_b'].values, max_features=self.max_features)

        self.y = np.empty((self.X.shape[0], self.workers_id.shape[0]))
        self.y[:] = np.nan

        self.y_real = tasks['please_decide_which_passage_is_more_difficult_gold'].values


        # TODO: optimize!!
        np.random.seed(seed)

        cnt_new_gold = 0
        i = 0
        for ind, df in tasks_groupby:
            new_df = df[~pd.isnull(df['please_decide_which_passage_is_more_difficult'])]

            if new_df.shape[0] == 0:
                continue

            workers_votes = new_df['please_decide_which_passage_is_more_difficult'].values

            if workers_votes.shape[0] >= self.golden_boarder and np.isnan(self.y_real[i]):
                self.y_real[i] = max(set(workers_votes.tolist()), key=workers_votes.tolist().count)
                cnt_new_gold += 1

            cnt_choose = min(self.workers_per_task, len(workers_votes))
            choose_inds = np.random.choice(list(range(workers_votes.shape[0])), cnt_choose, replace=False)

            self.y[i, self.workers_id.loc[new_df._worker_id.values[choose_inds]].values] = \
                workers_votes[choose_inds].reshape((self.workers_id.loc[new_df._worker_id.values[choose_inds]].values.shape[0], 1))
            i += 1

        workers_with_answers = np.argwhere(~np.isnan(self.y).all(axis=0)).squeeze()
        tasks_with_answers = np.argwhere(~np.isnan(self.y).all(axis=1)).squeeze()

        self.y = self.y[tasks_with_answers, :][:, workers_with_answers]
        self.X = self.X[tasks_with_answers]

        self.y_real = tasks.iloc[tasks_with_answers]['please_decide_which_passage_is_more_difficult_gold'].values
        self.gold_tasks_index = np.argwhere(~np.isnan(self.y_real)).squeeze()

        pass

    def bootstrap(self, size: int = None, marks_percentage: int = None, at_least:int = 1, seed:int = 0):
        """
        Choose uniformly <size> tasks and <cnt_marks> so that each task would have at least <at_least> marks. If for some
        task there are less marks then <at_least> all marks will be taken.

        :param size:
        :param cnt_marks:
        :param at_least:
        :param seed: Seed of random.
        :return: Numpy array of tasks' features, workers marks and real answer.
        """
        if size is None:
            size = self.X.shape[0]

        index_to_take = self.gold_tasks_index
        cnt_more_take = size - index_to_take.shape[0]

        np.random.seed(seed)
        index_to_take = np.concatenate((index_to_take, np.random.choice(list(range(self.X.shape[0])), cnt_more_take)))

        boot_y = self.y[index_to_take]

        marks_to_take = []
        for i in range(boot_y.shape[0]):
            cnt_to_take = min((~np.isnan(boot_y[i, :])).sum(), at_least)
            cur_marks_to_take = \
                np.random.choice(np.argwhere(~np.isnan(boot_y[i, :])).squeeze(), cnt_to_take, replace=False).tolist()
            marks_to_take += [i*boot_y.shape[1] + cur_mark_to_take for cur_mark_to_take in cur_marks_to_take]

        if marks_percentage is None:
            cnt_marks = (~np.isnan(boot_y)).sum()
        else:
            cnt_marks = int(marks_percentage*(~np.isnan(self.y)).sum()//100)

        assert (size*at_least <= cnt_marks)
        cnt_marks = min(cnt_marks, (~np.isnan(boot_y)).sum())

        flatten_boot_y = boot_y.flatten()
        flatten_boot_y[marks_to_take] = None

        more_marks_to_take = cnt_marks - len(marks_to_take)

        marks_to_take += np.random.choice(np.argwhere(~np.isnan(flatten_boot_y)).squeeze(), more_marks_to_take, replace=False).tolist()

        worse_boot_y_flatten = np.empty((flatten_boot_y.shape[0],))
        worse_boot_y_flatten[:] = None
        worse_boot_y_flatten[marks_to_take] = boot_y.flatten()[marks_to_take]

        worse_boot_y = worse_boot_y_flatten.reshape(boot_y.shape)

        assert ((~np.isnan(worse_boot_y)).sum() == cnt_marks)

        workers_with_answers = np.argwhere(~np.isnan(worse_boot_y).all(axis=0)).squeeze()

        return self.X[index_to_take], worse_boot_y[:, workers_with_answers], self.y_real[index_to_take]


if __name__ == '__main__':
    """
    Check if transformation works, count the consistency of labels. 
    """

    filepath = 'datasets/wsdm.csv'
    chendata = ChenData(filepath)
    chendata.transform_points()

    _, boot_y, _ = chendata.bootstrap(400*chendata.X.shape[0])
    cnt_matches = 0
    for i in range(boot_y.shape[0]):
        inds = np.random.choice(np.argwhere(~np.isnan(boot_y[i, :])).squeeze(), 2, replace=False).tolist()
        cnt_matches += (boot_y[i, inds[0]] == boot_y[i, inds[1]])

    print("Agreement: {}".format(cnt_matches/boot_y.shape[0]))



    print("Random: {}")
