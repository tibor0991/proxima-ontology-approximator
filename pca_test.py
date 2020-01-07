import csv
import json
from decimal import Decimal

from numpy.lib import recfunctions
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition, neighbors, preprocessing


def get_neighbourhoods(names, sim_matrix, theta) -> dict:
    """
    For each individual and for a given theta, return the neighbors list
        names: the names of the individuals
        sim_matrix: a similarity/distance matrix, that is a NxN square matrix where N = number of individuals
        theta: a certain theta to generate the neighbourhoods
    """
    dict_neighbourhood = dict()
    used_names = []
    for row_index, ind_name in enumerate(names):
        if ind_name not in used_names:
            distance_row = sim_matrix[row_index, :]
            truth_array = [d >= theta for d in distance_row]
            neigh = [column_name for t, column_name in zip(truth_array, names) if t]
            used_names += neigh
            #print("Neighbourhood of ", ind_name, ": ", neigh)
            dict_neighbourhood[ind_name] = set(neigh)
        else:
            #print("%s has already been picked, discarding..." % ind_name)
            pass
    return dict_neighbourhood


def naive_similarity(a, b, p=2) -> float:
    basic_sims = (np.ones(a.shape) - abs(a - b)) / len(a)
    power_sims = np.power(basic_sims, p)
    root_sum = np.power(np.sum(power_sims), 1./p)
    return root_sum


def get_similarities(dataset, method='naive'):
    sim_matrix = np.ndarray(shape=(len(dataset), len(dataset)))
    for first in range(0, len(dataset)):
        for second in range(first, len(dataset)):
            sim_matrix[first, second] = naive_similarity(dataset[first], dataset[second])
            sim_matrix[second, first] = sim_matrix[first, second]
    return sim_matrix


# %% Load the stored data from the .json
input_file_path = r"C:\Users\Gianf\Dropbox\Tesi\table.csv"

with open(input_file_path) as f:
    input_data = np.recfromcsv(f, converters={'Name': lambda s: str(s)}, names=True, encoding='utf-8')

# create the list of individuals
list_IndividualNames = input_data['name']

#create the dict of classes
list_ClassesNames = input_data.dtype.names
print(list_ClassesNames)
# dropping the name field
nameless_data = recfunctions.drop_fields(input_data, 'name')

raw_X = nameless_data.view(np.int).reshape(len(nameless_data), -1)

#print(raw_X)
print("Shape of the original dataset:", raw_X.shape)

# %% PCA reduction

# scale the data
scaler = preprocessing.StandardScaler()
scaled_X = scaler.fit_transform(raw_X)
print("Shape of the scaled dataset:", scaled_X.shape)

# apply PCA
PCA = decomposition.PCA(0.95)
red_X = PCA.fit_transform(scaled_X)
print("Shape of the PCA-transformed dataset:", red_X.shape)

# build a covariance matrix from red_X
cov_mat = np.cov(red_X, rowvar=False)

print("Shape of the covariance matrix: ", cov_mat.shape)

# Compute Mahalanobis' distance pairwise
dist = neighbors.DistanceMetric.get_metric('mahalanobis', V=cov_mat)
pair_distances = dist.pairwise(red_X)

# fake normalization (just remaps the values in a [0...1] range)
max_dist = np.max(pair_distances)
pair_distances /= max_dist
pair_distances = np.subtract(1., pair_distances)

print("Shape of pair sets:", pair_distances.shape)
with np.printoptions(precision=3, suppress=True):
    print(pair_distances)

neigh_dict = get_neighbourhoods(list_IndividualNames, pair_distances, 0.95)
print(len(list_IndividualNames))
# %% Given a set of positive individuals, put in the lower approximation
# subsets for which the intersection is equal to the subset itself

lower_approx = []
upper_approx = []
outside_approx = []
query = 'winezinfandel'
class_set = set([name for name, value in zip(list_IndividualNames, input_data[query]) if value == 1])
print("Check which subset intersects with ", class_set)

for k in neigh_dict.keys():
    if neigh_dict[k].issubset(class_set): #checks for lower approx
        print("The set %s is a lower approximation for %s" % (k, query))
        lower_approx.append(neigh_dict[k])
        pass
    elif len(neigh_dict[k] & class_set) > 0:
        print("The set %s is an upper approximation for %s" % (k, query))
        upper_approx.append(neigh_dict[k])
        pass
    else:
        print("The set %s is not an approximation for %s" % (k, query))
        outside_approx.append(neigh_dict[k])
        pass
print("=== For the naive rough sets approach: ===")
print("The lower approximation for %s is:" % query, lower_approx)
print("The upper approximation for %s is:" % query, upper_approx)
print("The items outside any approximations for %s are: " % query, outside_approx)

# exit the program
exit()
