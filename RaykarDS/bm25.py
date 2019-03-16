import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES


class Bm25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=2.0, b=0.75):
        self.k = k
        self.b = b

    # noinspection PyUnusedLocal,PyIncorrectDocstring
    def fit(self, X, y=None):
        """Learn the idf vector (global term weights) and normalized document lengths

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        lengths = X.sum(axis=1)
        # noinspection PyAttributeOutsideInit
        self._mean_length = np.mean(lengths)

        transformer = TfidfTransformer()
        transformer.fit(X)
        # noinspection PyAttributeOutsideInit
        self._idf = transformer.idf_

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to an Okapi BM25 representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        lengths = X.sum(axis=1)
        # noinspection PyAttributeOutsideInit
        norm_lengths = lengths / self._mean_length

        data = np.zeros(X.nnz)
        start = 0
        for i in range(X.shape[0]):
            row_len = len(X.indices[X.indptr[i]:X.indptr[i + 1]])
            data[start:start + row_len] = X.data[start:start + row_len] + self.k * (
                    1 - self.b + self.b * norm_lengths[i])
            start = start + row_len
        X.data = (self.k + 1) * X.data / data

        X = X.multiply(sp.csr_matrix(self._idf))
        return X
