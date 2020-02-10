import pandas as pd
from scipy.spatial import distance
import numpy as np
from sklearn import decomposition, preprocessing


class SimilarityMeasure:
    def __init__(self, covariance_matrix):
        self.iv = np.linalg.inv(covariance_matrix)
        diagonal = np.diagonal(covariance_matrix)

        self.gamma = 1. / (2 * np.average(diagonal))

    def __call__(self, a, b):
        d = distance.mahalanobis(a, b, self.iv)
        s = np.exp((-d ** 2) * self.gamma)
        return s


class NeighbourhoodSearcher:
    def __init__(self, universe, sim_measure):
        self.U = universe
        self.sim = sim_measure

    def __call__(self, x, theta) -> set:
        nh = set()
        for name, *y in self.U.itertuples():
            d = self.sim(x, y)
            if d >= theta:
                nh.add(name)
        return nh


def rough_membership(left_set, right_set, beta=0):
    if left_set:  # if the left set is not null
        m = len(left_set & right_set) / len(left_set)
        if m <= beta:
            m = 0
        elif m >= (1. - beta):
            m = 1
        else:
            m = (m-beta)/(1-2*beta)
    else:  # avoid division by 0, better safe than sorry
        m = 1
    return m


class ToleranceApproximator:
    def __init__(self):
        self.U = None
        self.neighbourhood = None

    def fit(self, projection_table, variance=0.95):
        # scale the data
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(projection_table)

        # apply PCA
        PCA = decomposition.PCA(variance)
        pca_data = PCA.fit_transform(scaled_data)

        # get the covariance matrix from the PCA mapped space
        covariance = np.cov(pca_data, rowvar=False)
        similarity = SimilarityMeasure(covariance_matrix=covariance)
        # U is the whole set of individuals in the ontology
        self.U = pd.DataFrame(pca_data, index=projection_table.index)
        self.neighbourhood = NeighbourhoodSearcher(self.U, similarity)

    def approximate(self, examples, theta, beta=0.):
        upper = set()
        lower = set()
        pairs = dict()
        for ind_name, *ind_data in self.U.itertuples():
            neigh_set = self.neighbourhood(ind_data, theta)
            membership = rough_membership(neigh_set, examples, beta)
            if membership > 0:
                upper.add(ind_name)
                pairs[ind_name] = neigh_set
            if membership == 1:
                lower.add(ind_name)

        print(pairs)
        return upper, lower, pairs
