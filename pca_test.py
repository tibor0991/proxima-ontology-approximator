import scipy
from numpy.lib import recfunctions
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition, preprocessing, neighbors


class SimilarityMeasure:
    def __init__(self, covariance_matrix):
        self.iv = np.linalg.inv(covariance_matrix)
        self.gamma = 1. / (2 * covariance_matrix[0, 0])
        with np.printoptions(precision=3, suppress=True):
            print("Covariance matrix:\n", covariance_matrix)
            print("Precision matrix:\n", self.iv)
            print("Gamma:", self.gamma)

    def __call__(self, a, b):
        d = scipy.spatial.distance.mahalanobis(a, b, self.iv)
        s = np.exp((-d ** 2) * self.gamma)
        return s


class NeighbourhoodSearcher:
    def __init__(self, universe, sim_measure, theta=1.0):
        self.theta = theta
        self.U = universe
        self.sim = sim_measure

    def __call__(self, individual) -> set:
        nh = set()
        x_name, x = individual
        for name, y in self.U:
            if self.sim(x, y) >= self.theta:
                nh.add(name)
        return nh


def inclusion_index(a_set, b_set):
    # Szymkiewicz-Simpson coefficient, modified to check only for "A included in B"
    if a_set:
        return len(a_set & b_set) / len(a_set)
    else:
        return 1


def get_context_sets(data, context):
    for name, t in zip(list_IndividualNames, data[context]):
        print(name, ":", t, "=", sum(t)/len(context))

    pseudoclass = [sum(t)/len(context) for t in data[context]]

    definitive = [name for name, value in zip(list_IndividualNames, pseudoclass) if value == 1]
    possible = [name for name, value in zip(list_IndividualNames, pseudoclass) if value >= 0]
    outside = [name for name, value in zip(list_IndividualNames, pseudoclass) if value <= 0]

    return set(definitive), set(possible), set(outside)


if __name__ == '__main__':
    # Load the stored data from the
    input_file_path = r"C:\Users\Gianf\Dropbox\Tesi\table.csv"

    with open(input_file_path) as f:
        input_data = np.recfromcsv(f, converters={'Name': lambda s: str(s)}, names=True, encoding='utf-8',
                                   case_sensitive=True)

    # Create the list of individuals
    list_IndividualNames = list(input_data['Name'])
    print("Individuals: ", list_IndividualNames)

    # create the dict of classes
    list_ClassesNames = list(input_data.dtype.names)
    print("Classes: ", list_ClassesNames)

    # dropping the name field
    nameless_data = recfunctions.drop_fields(input_data, 'name')
    raw_X = nameless_data.view(np.int).reshape(len(nameless_data), -1)

    print("Shape of the original dataset:", raw_X.shape)

    # scale the data
    scaler = preprocessing.StandardScaler()
    scaled_X = scaler.fit_transform(raw_X)
    print("Shape of the scaled dataset:", scaled_X.shape)

    # apply PCA
    PCA = decomposition.PCA(0.95)
    red_X = PCA.fit_transform(scaled_X)
    print("Shape of the PCA-transformed dataset:", red_X.shape)

    # get the covariance matrix from the PCA mapped space
    covariance = np.cov(red_X, rowvar=False)

    # U is the whole set of individuals in the ontology
    U = list(zip(list_IndividualNames, list(red_X)))

    sim = SimilarityMeasure(covariance)
    neighbourhood = NeighbourhoodSearcher(U, sim, 0.99)

    # gets the positive set relative to a context
    context = ['wineWine', 'wineRedWine']
    context_set, possible_set, outside_set = get_context_sets(input_data, context)
    print(context_set)

    # gets the set of all individuals belonging to a concept

    lower_approx = set()
    upper_approx = set()
    excluded = set()

    for ind in U:
        ind_name, _ = ind
        neigh_set = neighbourhood(ind)
        print("Neighbourhood of", ind_name, ":", neigh_set)
        membership = inclusion_index(neigh_set, context_set)

        if membership == 1.:
            lower_approx.add(ind_name)
        if membership > 0:
            upper_approx.add(ind_name)
        if membership == 0:
            excluded.add(ind_name)

    print("======================================= Approximations results for", context, ": ======================================= ")
    print("Upper approx (Error : %3f): " % inclusion_index(upper_approx, outside_set), upper_approx)
    print("Lower approx (Error : %3f): " % inclusion_index(lower_approx, outside_set), lower_approx)

    # exit the program
    exit()