import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from bm25 import Bm25Transformer


class ChenData:
    def __init__(self, file, max_features=100, workers_per_task=100, golden_boarder=100):
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

    def create_features(self, texts, tasks_a, tasks_b, max_features=None):
        bm25_transformer = Pipeline([
            ('vect', CountVectorizer(max_features=max_features, stop_words='english')),
            ('tfidf', Bm25Transformer())
        ]).fit(texts)

        bm25_A = bm25_transformer.transform(tasks_a).toarray()
        bm25_B = bm25_transformer.transform(tasks_b).toarray()

        return np.concatenate([bm25_A, bm25_B], axis=1)

    def transform_points(self, seed=0):
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
                self.y_real[i] = max(set(workers_votes.tolist()), key = workers_votes.tolist().count)
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

    def bootstrap(self, size=None, seed=0):
        if size is None:
            size = self.X.shape[0]

        index_to_take = np.squeeze(self.gold_tasks_index)
        cnt_more_take = size - index_to_take.shape[0]

        np.random.seed(seed)
        index_to_take = np.concatenate((index_to_take, np.random.choice(list(range(self.X.shape[0])), cnt_more_take)))

        boot_y = self.y[index_to_take]
        workers_with_answers = np.argwhere(~np.isnan(boot_y).all(axis=0)).squeeze()

        return self.X[index_to_take], boot_y[:, workers_with_answers], self.y_real[index_to_take]


if __name__ == '__main__':
    filepath = 'data/wsdm.csv'
    chendata = ChenData(filepath)
    chendata.transform_points()

    lr = LinearRegression()
    lr.fit(chendata.X[chendata.gold_tasks_index], chendata.y_real[chendata.gold_tasks_index])
    print(
        ((lr.predict(chendata.X[chendata.gold_tasks_index]) > 0.5).astype(int) == chendata.y_real[chendata.gold_tasks_index]).mean())
