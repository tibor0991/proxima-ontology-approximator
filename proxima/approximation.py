import pandas as pd
from scipy.spatial import distance
import numpy as np
from sklearn import decomposition, preprocessing


class SimilarityMeasure:
    def __init__(self, covariance_matrix):
        self.iv = np.linalg.inv(covariance_matrix)
        self.gamma = 1. / (2 * covariance_matrix[0, 0])
        """
        with np.printoptions(precision=3, suppress=True):
            print("Covariance matrix:\n", covariance_matrix)
            print("Precision matrix:\n", self.iv)
            print("Gamma:", self.gamma)
        """

    def __call__(self, a, b):
        d = distance.mahalanobis(a, b, self.iv)
        s = np.exp((-d ** 2) * self.gamma)
        return s


class ContextualSimilarityMeasure:
    def __init__(self, p=2):
        self.p = p
        pass

    def __call__(self, a, b):
        # assuming a and b are of the same length
        d = np.power(np.sum(np.power(np.abs(a - b) / len(a), self.p)), 1. / self.p)
        return 1. - d


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


def rough_membership(a_set, b_set):
    # Szymkiewicz-Simpson coefficient, modified to check only for "A ⊆ B"
    if a_set:
        return len(a_set & b_set) / len(a_set)
    else:
        return 1


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

    def approximate(self, positive_examples, negative_examples, n_samples=None, theta=None):
        """

        :param positive_examples:   list of positive examples (provided as strings)
        :param negative_examples:   list of negative examples (provided as strings)
        :param n_samples:           if set, attempts to learn a better theta
        :param theta:               range of neighbourhood function
        :return:
        """
        upper = set()
        lower = set()
        if theta is not None:
            for ind_name, *ind_data in self.U.itertuples():
                neigh_set = self.neighbourhood(ind_data, theta)
                membership = rough_membership(neigh_set, positive_examples)
                if membership > 0:
                    upper.add(ind_name)
                if membership == 1:
                    lower.add(ind_name)

        return upper, lower

    def __call__(self, positive_examples, negative_examples, n_samples=None):
        return self.approximate(positive_examples, negative_examples, n_samples)
