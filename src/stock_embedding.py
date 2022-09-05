from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from matplotlib.pyplot import axis
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

from data.fetch_daily_return import (daily_return_calc, fetch_daily_price,
                                     fetch_sp500_list)


class StockEmbedding:
    def __init__(self,
                 return_df: Union[None, pd.DataFrame]=None,
                 start_date: str=None,
                 end_date: str=None) -> None:
        """return_df has columns of ['date', 'ticker', 'daily_return']
        """
        if return_df is None:
            sp_500 = fetch_sp500_list()
            sp_500_tickers = list(sp_500.keys())
            out = fetch_daily_price(sp_500_tickers, start=start_date, end=end_date, batch_size=20)
            return_df = daily_return_calc(out)
        if not sum([e in ['date', 'ticker', 'daily_return'] for e in return_df.columns]) == 3:
            raise ValueError("return df must contains columns of ['date', 'ticker', 'daily_return']")

        self.gics_info = fetch_sp500_list()
        self.tickers = None
        self.weighted_walks = None
        self.embedding = None
        self.embedding_df = None
        self.return_df = return_df
        self.corr_mat = self.return_corr_mat_calc()
        self.dist_mat = self._corr_to_distance()

    def save_return_df(self, fname='stock_return.csv') -> None:
        """save stock return data to csv to save time

        Args:
            fname (str, optional): file name. Defaults to 'stock_return.csv'.
        """
        self.return_df.to_csv(fname, index=False)


    def _return_df_transform(self):
        """each row is a series of return, order by date ascendingly
        """
        out = self.return_df.pivot(index='ticker', columns='date', values='daily_return').dropna(how='any')
        self.tickers = out.index.to_numpy()

        return out


    def return_corr_mat_calc(self) -> pd.DataFrame:
        # TODO: add weighted option
        return_reshape = self._return_df_transform().to_numpy()
        ndays = return_reshape.shape[1]
        term1 = return_reshape @ return_reshape.T

        row_sum = return_reshape.sum(axis=1).reshape([-1, 1])
        term2 = row_sum @ row_sum.T / ndays
        numerator = term1 - term2

        # denominator
        row_sq_sum = (return_reshape * return_reshape).sum(axis=1).reshape([-1, 1])

        var = row_sq_sum - np.square(row_sum) / ndays

        denominator = np.sqrt(var @ var.T)

        return numerator / denominator

    def _corr_to_distance(self) -> csr_matrix:
        return csr_matrix(np.triu(np.sqrt(2 * (1 - self.corr_mat))))

    def _mst_corr(self) -> np.array:
        mst = minimum_spanning_tree(self.dist_mat).toarray()
        sparsified_corr_mat = self.corr_mat.copy()
        sparsified_corr_mat[mst == 0] = 0

        return sparsified_corr_mat


    def _to_sg_graph(self) -> StellarGraph:
        """convert sparsified corr to stellargraph obj
        # https://stellargraph.readthedocs.io/en/stable/demos/basics/loading-pandas.html
        e.x.:
            a -- b
            | \  |
            |  \ |
            d -- c
            edges = pd.DataFrame(
                {"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "a", "c"]}
                )
        """
        # sparse_corr has len(self.tickers) - 1 non-zero element
        sparse_corr = self._mst_corr()
        source_idx, target_idx = np.nonzero(sparse_corr)
        source_ticker = self.tickers[source_idx]
        target_ticker = self.tickers[target_idx]
        weight = sparse_corr[np.nonzero(sparse_corr)]
        edges = pd.DataFrame({"source": source_ticker, "target": target_ticker, 'weight': weight})

        return StellarGraph(edges=edges)


    def _sim_random_walk(self, r: int, l: int, p: float, q: float, seed: int=42) -> None:
        """simulate random walk based on node2vec algorithm

        Args:
            r (int): the number of random walks per root node
            l (int): the length for each random walk
            p (float): prob. of a random walk will return to the node it visited previously
            q (float): prob. of a random walk will explore the unexplored part of the graph
            seed (int): random seed
        """
        g = self._to_sg_graph()
        rw = BiasedRandomWalk(g)
        self.weighted_walks = rw.run(
            nodes=g.nodes(),  # root nodes
            length=l,  # maximum length of a random walk
            n=r,  # number of random walks per root node
            p=p,  # Defines (unormalised) probability, 1/p, of returning to source node
            q=q,  # Defines (unormalised) probability, 1/q, for moving away from source node
            weighted=True,  # for weighted random walks
            seed=seed  # random seed fixed for reproducibility
        )


    def learn_embedding(self,
                        r: int,
                        l: int,
                        p: float,
                        q: float,
                        vector_size: int,
                        window: int,
                        sg: int=1,
                        epochs: int=5,
                        workers=4,
                        seed: int=42) -> Dict[str, np.array]:
        """fit a word2vec model and learn the stock embeddings

        Args:
            r (int): number of random walk from each node in the network
            l (int): length for each random walk from each node in the network
            p (float): probability of a random walk will return to the node it visited previously
            q (float): probability of a random walk will explore the unexplored part of the graph
            vector_size (int): embedding size, dim in the paper
            window (int): Maximum distance between the current and predicted word within sentence. w in the paper
            sg (int, optional): 1 skip-gram, CBOW othrewise
            epochs (int, optional): Number of iterations (epochs) over the corpus. Defaults to 5.
            workers (int, optional): number of works in multiprocessing. Defaults to 4.
            seed (int): random seed

        Returns:
            Dict[str, np.array]: the stock ticker as key and the embedding as value
        """
        self._sim_random_walk(r, l, p, q, seed)
        weighted_model = Word2Vec(self.weighted_walks,
                                  vector_size=vector_size,
                                  window=window,
                                  min_count=0,
                                  sg=sg,
                                  workers=workers,
                                  epochs=epochs)

        embedding = {k: weighted_model.wv[k] for k in self.tickers}
        self.embedding = embedding
        self.embedding_df = pd.DataFrame(embedding)

        return embedding


    @staticmethod
    def _euclidean_distance(vec1: np.array, vec2: np.array):
        return np.sqrt(np.square(vec1 - vec2).sum())


    @staticmethod
    def _cosine_similarity(vec1: np.array, vec2: np.array):
        return vec1 @ vec2 / (np.sqrt(np.square(vec1).sum()) * np.sqrt(np.square(vec2).sum()))


    def calc_dist(self, ticker1: str, ticker2: str, method: str='cosine') -> float:
        """calculate similarity  bwtween two stocks

        Args:
            ticker1 (str): ticker
            ticker2 (str): ticker

        Raises:
            ValueError: if self.embedding is not computed

        Returns:
            float: Euclidean distance
        """
        if self.embedding is None:
            raise ValueError('No embedding dict, run learn_embedding method first')

        if method == 'euclidean':
            return self._euclidean_distance(self.embedding[ticker1], self.embedding[ticker2])

        if method == 'cosine':
            return self._cosine_similarity(self.embedding[ticker1], self.embedding[ticker2])


    def _calc_all_euclidean(self, ticker: str) -> pd.Series:
        target_vec = self.embedding_df[ticker]
        dists = np.sqrt(np.square((self.embedding_df.transform(lambda x: x - target_vec)).to_numpy()).sum(axis=0))
        out = pd.Series(dists, index=self.embedding_df.columns)

        return out


    def _calc_all_cosine(self, ticker):
        target_vec = self.embedding_df[ticker]
        out = self.embedding_df.apply(lambda x: self._cosine_similarity(target_vec, x), axis=0)

        return out


    def calc_all_distance(self, ticker: str, method='euclidean'):
        if method == 'euclidean':
            return self._calc_all_euclidean(ticker)

        if method == 'cosine':
            # cosine 'distance' between 0 and 1
            return (1 - self._calc_all_cosine(ticker)) / 2

        raise ValueError(f'Input method {method} is not supported')


    def calc_all_similarity(self, ticker: str) -> pd.Series:
        """calculate cosine similarity between the target stock and the rest

        Args:
            ticker (str): ticker of the target stock

        Returns:
            pd.Series: distance with ticker as index
        """
        return self._calc_all_cosine(ticker)


    def find_similar_stocks(self, ticker: str, n: int=10, method: str='cosine') -> pd.DataFrame:
        """find top n most similar stocks for the given ticker

        Args:
            ticker (str): ticker of the target stock
            n (int, optional): number of similar companies returned. Defaults to 10.
            method (str, optional): cosine distance or euclidean distance. Defaults to 'cossine'.

        Returns:
            pd.DataFrame: with columns of ['ticker', 'similarity']
        """
        out = self.calc_all_distance(ticker, method).sort_values().iloc[1:].head(n).reset_index(name='distance')
        out.columns = ['ticker', 'distance']
        out['sector'] = [self.gics_info[t]['GICS_sector'] for t in out['ticker']]
        out['group'] = [self.gics_info[t]['GICS_group'] for t in out['ticker']]
        out['industry'] = [self.gics_info[t]['GICS_industry'] for t in out['ticker']]

        return out


    def infer_industry(self, ticker: str, n: int=20, method: str='cosine') -> pd.DataFrame:
        """find GICS sector and industry of top similar stocks

        Args:
            ticker (str): ticker of the target company
            n (int, optional): number of tickers returned. Defaults to 20.
            method (str, optional): distance measure. Defaults to 'cosine'.

        Returns:
            pd.DataFrame: similar tickers as a DataFrame
        """
        similar_stock = self.find_similar_stocks(ticker, n, method)
        # map gics based on ticker
        similar_stock['industry'] = [self.gics_info[t]['GICS_industry'] for t in similar_stock['ticker']]

        return similar_stock['industry'].value_counts().index[0]


    def infer_multiple_industry(self, ticker: str, n: int=30, method: str='cosine'):
        # TODO: does it make sense?
        all_distance = self.calc_all_distance(ticker, method=method).sort_values()[1: n]
        dist_df = all_distance.reset_index()
        dist_df.columns = ['ticker', 'distance']
        dist_df['industry'] = [self.gics_info[t]['GICS_industry'] for t in dist_df ['ticker']]

        industry_summary = dist_df.groupby('industry').agg({'distance': np.mean}).sort_values('distance')


    def identify_not_match_stock(self, tickers: List[str]) -> str:
        """Answer questions like: Does not match from JPM, MS, GS, GOOGL -> GOOGL

        Args:
            tickers (List[str]): input of tickers

        Returns:
            str: ticker doesn't match the rest
        """
        similarity = []
        for ticker in tickers:
            embedding = self.embedding_df[ticker]
            other_tickers = [e for e in tickers if e != ticker]
            avg_embedding = self.embedding_df[other_tickers].apply(np.mean, axis=1)
            similarity.append(self._cosine_similarity(embedding, avg_embedding))

        not_match = tickers[np.argmin(similarity)]

        print(f"Does not match from {', '.join(tickers)}: {not_match}")

        return not_match


    def identify_similar_stock(self, ticker: str, tickers: List[str]) -> str:
        """answer questions like Most similar to GOOGL given JNJ, MS, MOS, META: META

        Args:
            ticker (str): target stock
            tickers (List[str]): candidate stocks

        Returns:
            str: the most simiilar stock
        """
        embedding = self.embedding_df[ticker]
        similarity = []

        for t in tickers:
            similarity.append(self._cosine_similarity(embedding, self.embedding_df[t]))

        match = tickers[np.argmax(similarity)]

        print(f"Most similar to {ticker} given {', '.join(tickers)}: {match}")

        return match


    def calc_vmeasure(self, gics_type='industry') -> float:
        """Calculate vmeasure of the true industry labels and predicted labels based on k-means

        Args:
            gics_type (str, optional): type of GICS as true label. Defaults to 'industry'.

        Returns:
            float: v-measure
        """
        embedding = self.embedding_df.T
        tickers = embedding.index

        if gics_type == 'industry':
            true_label = [self.gics_info[t]['GICS_industry'] for t in tickers]
        elif gics_type == 'sector':
            true_label = [self.gics_info[t]['GICS_sector'] for t in tickers]
        elif gics_type == 'subindustry':
            true_label = [self.gics_info[t]['GICS_subindustry'] for t in tickers]
        elif gics_type == 'group':
            true_label = [self.gics_info[t]['GICS_group'] for t in tickers]
        else:
            raise ValueError(f'wrong gics type, {gics_type} is not supported')

        X = embedding.to_numpy()
        kmeans = KMeans(n_clusters=len(set(true_label)), random_state=0).fit(X)
        label = kmeans.labels_

        return v_measure_score(true_label, label)


    def analogical_inference(self, ticker: str, ticker_1: str, ticker_2: str) -> Tuple[str, float]:
        """answer question like "JPM is to GS as JNJ is to ?"

        Args:
            ticker (str): target ticker to infer the counterpart
            ticker_1 (str): ticker for inference
            ticker_2 (str): target ticker of target1

        Returns:
            Tuple[str, float]: return counterpart ticker and its similarity
        """
        vec = self.embedding_df[ticker_2] - self.embedding_df[ticker_1] + self.embedding_df[ticker]
        out = self.embedding_df \
            .apply(lambda x: self._cosine_similarity(vec, x), axis=0) \
            .sort_values(ascending=False)

        if out.index[0] == ticker:
            analog_ticker = out.index[1]
            similarity = out[1]
        else:
            analog_ticker = out.index[0]
            similarity = out[0]

        print(f'{ticker_1} is to {ticker_2} as {ticker} is to {analog_ticker}')

        return analog_ticker, similarity


    def hyperparam_tuning(self,
                          param_lst: List[Dict[str, float]]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """tune hyperparameter r, l, p, q, w, and dim

        Args:
            param_lst (List[Dict[str, float]]): candidate hyperparameter with dictionary of keys r, l, p, q, w and dim

        Returns:
            Tuple(pd.DataFrame, Dict[str, float]): summary DataFrame and the optimal parameter
        """
        r_lst = []
        l_lst = []
        p_lst = []
        q_lst = []
        dim_lst = []
        w_lst = []
        sector_vm = []
        group_vm = []
        industry_vm = []
        subindustry_vm = []

        for param in param_lst:
            r_lst.append(param['r'])
            l_lst.append(param['l'])
            p_lst.append(param['p'])
            q_lst.append(param['q'])
            dim_lst.append(param['dim'])
            w_lst.append(param['w'])

            self.learn_embedding(
                r=param['r'],
                l=param['l'],
                p=param['p'],
                q=param['q'],
                vector_size=param['dim'],
                window=param['w']
                )
            sector_vm.append(self.calc_vmeasure('sector'))
            group_vm.append(self.calc_vmeasure('group'))
            industry_vm.append(self.calc_vmeasure('industry'))
            subindustry_vm.append(self.calc_vmeasure('subindustry'))

        sum_df = pd.DataFrame({
            'r': r_lst,
            'l': l_lst,
            'p': p_lst,
            'q': q_lst,
            'dim': dim_lst,
            'w': w_lst,
            'sector': sector_vm,
            'group': group_vm,
            'industry': industry_vm,
            'subindustry': subindustry_vm
        })

        sum_df['average'] = (sum_df['sector'] + sum_df['group'] + sum_df['industry'] + sum_df['subindustry']) / 4
        opt_param = param_lst[np.argmax(sum_df['average'])]

        return sum_df, opt_param


if __name__ == '__main__':
    rdf = pd.read_csv('src/data/daily_return.csv')
    se_obj = StockEmbedding(rdf)
    # corr = se_obj._sim_random_walk(r=50, l=100, p=2, q=0.5)
    # se_obj.learn_embedding(r=50, l=100, p=2, q=0.5, vector_size=16, window=5)
    # test = se_obj.find_similar_stock_gics('JPM', n=10)
    # test = se_obj.find_similar_stocks('JPM', n=10)
    # test = se_obj.find_similar_stock('JPM', n=10)

    param_lst = [{'l': 50, 'r': 10, 'p': 0.5, 'q': 2, 'w': 5, 'dim': 16},
                 {'l': 100, 'r': 50, 'p': 2, 'q': 0.5, 'w': 5, 'dim': 16},
                 {'l': 200, 'r': 10, 'p': 2, 'q': 0.5, 'w': 5, 'dim': 32}]

    se_obj.hyperparam_tuning(param_lst)