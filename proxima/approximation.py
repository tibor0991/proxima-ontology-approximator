import pandas as pd
import scipy
import owlready2 as or2
import numpy as np
from sklearn import decomposition, preprocessing


def render_colon(e: or2.entity) -> str:
    return "%s:%s" % (e.namespace.name, e.name)


def project_ontology(onto: or2.owl_ontology, remapper=None) -> pd.DataFrame:
    or2.set_render_func(render_colon)  # WARNING: unset me
    classes = list(onto.classes())
    individuals = list(onto.individuals())
    # print("Classes in ontology (%d):" % len(classes), classes)
    # print("Individuals in ontology (%d):" % len(individuals), individuals)
    disjunction_dict = dict()
    for c in classes:
        disjunction_dict[c] = set()
    disjunction_pairs = onto.disjoint_classes()
    for pair in disjunction_pairs:
        first, second = pair.entities
        disjunction_dict[first].add(second)
        disjunction_dict[second].add(first)
    projection_table = pd.DataFrame(data='UNCERTAIN', index=list(map(str, individuals)),
                                    columns=list(map(str, classes)))
    for c in classes:
        true_set = set(onto.search(type=c))
        false_set = set()
        for not_c in disjunction_dict[c]:
            false_set = false_set.union(set(onto.search(type=not_c)))
        false_set = set(map(str, false_set))
        for t in true_set:
            projection_table.at[str(t), str(c)] = 'TRUE'
        for f in false_set:
            projection_table.at[str(f), str(c)] = 'FALSE'
    # print(projection_table)
    or2.set_render_func(or2.default_render_func)  # unsets the render function
    if remapper:
        return projection_table.apply(np.vectorize(remapper))
    else:
        return projection_table


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
        d = scipy.spatial.distance.mahalanobis(a, b, self.iv)
        s = np.exp((-d ** 2) * self.gamma)
        return s


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


def inclusion_index(a_set, b_set):
    # Szymkiewicz-Simpson coefficient, modified to check only for "A ⊆ B"
    if a_set:
        return len(a_set & b_set) / len(a_set)
    else:
        return 1


def get_context_sets(data, context):
    print(data[context])
    for name, *row in data[context].itertuples():
        print(name, ":", row, "=", sum(row) / len(context))

    pseudo_class = [(name, sum(row) / len(context)) for name, *row in data[context].itertuples()]

    definitive = [name for name, value in pseudo_class if value == 1]
    outside = [name for name, value in pseudo_class if value == 0]

    return set(definitive), set(outside)


class ToleranceRoughApproximator:
    """
    -Inclusion index
    -Neighbourhood function
    -Similarity Relation
    -Positive examples (and negative examples)

    """

    def __init__(self, projection_table, variance=0.95, metric='mahalanobis'):
        # builds the U dataset, components x samples
        # print("Shape of the original dataset:", projection_table.shape)

        # scale the data
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(projection_table)
        # print("Shape of the scaled dataset:", scaled_data.shape)

        # apply PCA
        PCA = decomposition.PCA(variance)
        pca_data = PCA.fit_transform(scaled_data)
        # print("Shape of the PCA-transformed dataset:", pca_data.shape)

        # get the covariance matrix from the PCA mapped space
        covariance = np.cov(pca_data, rowvar=False)
        self.similarity = SimilarityMeasure(covariance_matrix=covariance)

        # U is the whole set of individuals in the ontology
        self.U = pd.DataFrame(pca_data, index=projection_table.index)
        self.neighbourhood = NeighbourhoodSearcher(self.U, self.similarity)
        pass

    def __call__(self, positive_examples, negative_examples, n_samples=10):
        upper = set()
        lower = set()
        # heuristic rule #1: the lower theta will be always higher than the upper theta
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

    def _get_upper(self, positive, negative, theta_u):
        upper = set()
        for ind_name, *ind_data in self.U.itertuples():
            neigh_set = self.neighbourhood(ind_data, theta_u)
            membership = inclusion_index(neigh_set, positive)
            if membership > 0:
                upper.add(ind_name)

        consistency_error = inclusion_index(upper, negative)
        return upper, consistency_error

    def _get_lower(self, positive, theta_l):  # code smell here: can I avoid replicating code?
        lower = set()
        for ind_name, *ind_data in self.U.itertuples():
            neigh_set = self.neighbourhood(ind_data, theta_l)
            membership = inclusion_index(neigh_set, positive)
            if membership == 1:
                lower.add(ind_name)
        coverage_index = 1. - inclusion_index(positive, lower)
        return lower, coverage_index


class DataRemapper:
    def __init__(self, values_dict):
        self.values_dict = values_dict

    def __call__(self, value):
        return self.values_dict[value]
