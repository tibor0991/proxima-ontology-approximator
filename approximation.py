import csv
import tkinter
from tkinter import filedialog
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import numpy as np
from sklearn import decomposition, preprocessing

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

    def __call__(self, x) -> set:
        nh = set()
        for name, *y in self.U.itertuples():
            if self.sim(x, y) >= self.theta:
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
        print(name, ":", row, "=", sum(row)/len(context))

    pseudo_class = [(name, sum(row)/len(context)) for name, *row in data[context].itertuples()]

    definitive = [name for name, value in pseudo_class if value == 1]
    # possible = [name for name, value in zip(list_IndividualNames, pseudo_class) if value >= 0]
    outside = [name for name, value in pseudo_class if value == 0]

    return set(definitive), set(outside)


class RoughApproximator:
    def __init__(self, projection_table, variance=0.95):
        pass

    def __call__(self, context):
        pass


class DataRemapper:
    def __init__(self, values_dict):
        self.values_dict = values_dict

    def __call__(self, value):
        return self.values_dict[value]


if __name__ == '__main__':
    tkinter.Tk().withdraw()
    input_file_path = tkinter.filedialog.askopenfilename()

    remapper = DataRemapper({'TRUE': 1, 'FALSE': 0, 'UNCERTAIN': 0.5})

    projection_table = pd.read_csv(input_file_path, delimiter=";", index_col=0).apply(np.vectorize(remapper))

    print(projection_table)

    # Create the list of individuals
    list_IndividualNames = list(projection_table.index);
    print("Individuals: ", list_IndividualNames)

    # create the dict of classes
    list_ClassesNames = list(projection_table.columns)
    print("Classes: ", list_ClassesNames)

    print("Shape of the original dataset:", projection_table.shape)

    # scale the data
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(projection_table)
    print("Shape of the scaled dataset:", scaled_data.shape)

    # apply PCA
    PCA = decomposition.PCA(0.95)
    pca_data = PCA.fit_transform(scaled_data)
    print("Shape of the PCA-transformed dataset:", pca_data.shape)

    # get the covariance matrix from the PCA mapped space
    covariance = np.cov(pca_data, rowvar=False)

    # U is the whole set of individuals in the ontology
    U = pd.DataFrame(pca_data, index=list_IndividualNames)

    sim = SimilarityMeasure(covariance)
    neighbourhood = NeighbourhoodSearcher(U, sim)
    errors = []
    # gets the positive set relative to a context
    context = ['wine:Wine', 'wine:Zinfandel']
    context_set, outside_set = get_context_sets(projection_table, context)
    print(context_set)
    samples = np.linspace(0, 1., num=11, endpoint=True)
    for sample in samples:
        neighbourhood.theta = sample

        # gets the set of all individuals belonging to a concept
        lower_approx = set()
        upper_approx = set()

        for ind_name, *ind_data in U.itertuples():

            neigh_set = neighbourhood(ind_data)
            membership = inclusion_index(neigh_set, context_set)

            if membership == 1.:
                lower_approx.add(ind_name)
            if membership > 0:
                upper_approx.add(ind_name)

        print("== Approximations results for", context, " with theta = %f:" % neighbourhood.theta)
        print("-Upper approx (Error : %3f): " % inclusion_index(upper_approx, outside_set), upper_approx)
        print("-Lower approx (Error : %3f): " % inclusion_index(lower_approx, outside_set), lower_approx)

        errors.append(inclusion_index(upper_approx, outside_set))

    plt.plot(samples, errors)
    plt.show()
    # exit the program
    exit()