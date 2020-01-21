import csv
import tkinter
from tkinter import filedialog
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import numpy as np
from sklearn import decomposition, preprocessing
import seaborn as sns


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
        print(name, ":", row, "=", sum(row) / len(context))

    pseudo_class = [(name, sum(row) / len(context)) for name, *row in data[context].itertuples()]

    definitive = [name for name, value in pseudo_class if value == 1]
    outside = [name for name, value in pseudo_class if value == 0]

    return set(definitive), set(outside)


class RoughApproximator:
    """
    -Inclusion index
    -Neighbourhood function
    -Similarity Relation
    -Positive examples (and negative examples)

    """

    def __init__(self, projection_table, variance=0.95, metric='mahalanobis'):
        # builds the U dataset, components x samples
        print("Shape of the original dataset:", projection_table.shape)

        # scale the data
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(projection_table)
        print("Shape of the scaled dataset:", scaled_data.shape)

        # apply PCA
        PCA = decomposition.PCA(variance)
        pca_data = PCA.fit_transform(scaled_data)
        print("Shape of the PCA-transformed dataset:", pca_data.shape)

        # get the covariance matrix from the PCA mapped space
        covariance = np.cov(pca_data, rowvar=False)
        self.similarity = SimilarityMeasure(covariance_matrix=covariance)

        # U is the whole set of individuals in the ontology
        self.U = pd.DataFrame(pca_data, index=projection_table.index)
        self.neighbourhood = NeighbourhoodSearcher(self.U, self.similarity)
        pass

    def __call__(self, positive_examples, negative_examples, theta='auto'):
        upper = set()
        lower = set()

        if theta != 'auto':
            self.neighbourhood.theta = theta

        for ind_name, *ind_data in self.U.itertuples():
            neigh_set = self.neighbourhood(ind_data)
            membership = inclusion_index(neigh_set, positive_examples)

            if membership == 1.:
                lower.add(ind_name)
            if membership > 0:
                upper.add(ind_name)

            error_index = inclusion_index(upper, negative_examples)

        return upper, lower, error_index

    def plot_approximation(self):
        sns.set(style="ticks", color_codes=True)
        U_tagged = self.U.loc[:, 0:4].copy()
        classification = []
        for ind in list(self.U.index):
            if ind in lower_approx:
                classification.append('LOWER')
            elif ind in upper_approx:
                classification.append('UPPER')
            else:
                classification.append('OUTSIDE')
        U_tagged['class'] = classification

        g = sns.pairplot(U_tagged, hue='class')

        plt.show()


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
    list_IndividualNames = list(projection_table.index)
    print("Individuals: ", list_IndividualNames)

    # create the dict of classes
    list_ClassesNames = list(projection_table.columns)
    print("Classes: ", list_ClassesNames)

    print("Shape of the original dataset:", projection_table.shape)

    approximator = RoughApproximator(projection_table)

    # gets the positive (and negative) set relative to a context
    positive_set = set([name for name, value in projection_table['wine:Zinfandel'].iteritems() if value == 1])
    positive_set.add('wine:BancroftChardonnay')
    negative_set = set([name for name, value in projection_table['wine:Wine'].iteritems() if value == 0])

    lower_approx, upper_approx, error = approximator(positive_set, negative_set, theta=0.5)

    print("=== Approximations results ===")
    print("Definitively:", lower_approx)
    print("Probably:", upper_approx)
    print("Error:", error)

    approximator.plot_approximation()

    # exit the program
    exit()
