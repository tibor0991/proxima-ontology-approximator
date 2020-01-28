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
            if self.sim(x, y) >= theta:
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
        self.similarity = None
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
        self.similarity = SimilarityMeasure(covariance_matrix=covariance)

        # U is the whole set of individuals in the ontology
        self.U = pd.DataFrame(pca_data, index=projection_table.index)
        self.neighbourhood = NeighbourhoodSearcher(self.U, self.similarity)

    def approximate(self, positive_examples, negative_examples, n_samples=10):
        upper = set()
        lower = set()
        # upper theta: the minimum theta at which the consistency error on the upper approximation is at its lowest
        # lower theta: the minimum theta at which the lower approximation covers the most of the positive examples set
        theta_u = 0.
        last_index = 0
        sample_points = np.linspace(0., 1., n_samples + 1)
        for sample_index, sample in enumerate(sample_points):
            theta_u = sample
            upper, error_u = self._get_upper(positive_examples, negative_examples, sample)
            last_index = sample_index
            if error_u == 0:
                break
        theta_l = 0.
        sample_points = np.linspace(theta_u, 1., n_samples + 1 - last_index)
        for sample in sample_points:
            theta_l = sample
            lower, error_l = self._get_lower(positive_examples, sample)
            if error_l == 0:
                break

        print("Lower theta:", theta_l, lower)
        print("Upper theta:", theta_u, upper)

        return upper, lower

    def __call__(self, positive_examples, negative_examples, n_samples=10):
        return self.approximate(positive_examples, negative_examples, n_samples)

    def _get_upper(self, positive, negative, theta_u):
        upper = set()
        for ind_name, *ind_data in self.U.itertuples():
            neigh_set = self.neighbourhood(ind_data, theta_u)
            membership = rough_membership(neigh_set, positive)
            if membership > 0:
                upper.add(ind_name)

        consistency_error = rough_membership(upper, negative)
        return upper, consistency_error

    def _get_lower(self, positive, theta_l):  # code smell here: can I avoid replicating code?
        lower = set()
        for ind_name, *ind_data in self.U.itertuples():
            neigh_set = self.neighbourhood(ind_data, theta_l)
            membership = rough_membership(neigh_set, positive)
            if membership == 1:
                lower.add(ind_name)
        coverage_index = 1. - rough_membership(positive, lower)
        return lower, coverage_index


"""
from sklearn import cluster


class RbfDistance:
    def __init__(self, cov):
        self.iv = np.linalg.inv(cov)
        self.gamma = 1. / (2 * cov[0, 0])

    def __call__(self, a, b):
        x = distance.mahalanobis(a, b, IV=self.iv)
        d = np.exp(-(x ** 2) * self.gamma)
        return 1. - d

class ClusterRoughApproximator:
    def __init__(self, projection_table, variance=0.95):
        # scale the data
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(projection_table)
        # print("Shape of the scaled dataset:", scaled_data.shape)

        # apply PCA
        PCA = decomposition.PCA(variance)
        pca_data = PCA.fit_transform(scaled_data)
        # print("Shape of the PCA-transformed dataset:", pca_data.shape)

        # U is the whole set of individuals in the ontology
        U = pd.DataFrame(pca_data, index=projection_table.index)

        distance_measure = RbfDistance(cov=np.cov(pca_data, rowvar=False))

        clusterer = cluster.OPTICS(min_samples=2, eps=0.05, metric=distance_measure)

        clusterer.fit_predict(U)

        print("Clusters:", set(clusterer.labels_))

        self.clusters = dict()
        labels = set(clusterer.labels_)
        labels.remove(-1)
        for label in labels:
            self.clusters[label] = set([name for name, label in zip(U.index, clusterer.labels_)])

        for k in self.clusters.keys():
            print(k, ":", self.clusters[k])

    def __call__(self, positive_set, negative_set):
        upper = set()
        lower = set()
        for label, cluster_set in self.clusters.items():
            if inclusion_index(cluster_set, positive_set) > 0:
                upper = upper.union(cluster_set)
            if inclusion_index(cluster_set, positive_set) == 1:
                lower = lower.union(cluster_set)

        print("Upper:", upper)
        print("Lower:", lower)
        return upper, lower
"""
