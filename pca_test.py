import csv
import json
from decimal import Decimal

from numpy.lib import recfunctions
from scipy.io import arff
import numpy as np

from sklearn import decomposition, neighbors, preprocessing

"""
0) (optional) compute the concept value for each individual and each concept by querying a reasoner
1) Load the Concepts X Individuals table from an arff file
2) Apply a PCA reduction on the dataset
3) Compute covariance and use it to obtain a Mahalanobis distance matrix
4) with nh(x, theta) = neighbourhood of individual x within theta
5) C_upper = all those x for which its neighbourhood intersects with set C
6) C_lower = all those x for which its neighbourhood is a subset of C
"""

# Load the stored data from the
input_file_path = r"C:\Users\Gianf\Dropbox\Tesi\table.csv"

with open(input_file_path) as f:
    input_data = np.recfromcsv(f, converters={'Name': lambda s: str(s)}, names=True, encoding='utf-8',
                               case_sensitive=True)

# Create the list of individuals
list_IndividualNames = list(input_data['Name'])
print(list_IndividualNames)

# create the dict of classes
list_ClassesNames = list(input_data.dtype.names)
print(list_ClassesNames)
# dropping the name field
nameless_data = recfunctions.drop_fields(input_data, 'name')

raw_X = nameless_data.view(np.int).reshape(len(nameless_data), -1)

# print(raw_X)
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

# remap values (just remaps the values in a [0...1] range)
max_dist = np.max(pair_distances)
pair_distances /= max_dist
pair_distances = np.subtract(1., pair_distances)

print("Shape of pair sets:", pair_distances.shape)
with np.printoptions(precision=3, suppress=True):
    print(pair_distances)

theta = 0.8


def neighbourhood(x) -> set:
    """
    Returns the neighbourhood of x within a certain eta
    :param x: name of the individual
    :return: the neighbourhood of x, the set of all individuals with a similarity of at least theta
    """
    row_index = list_IndividualNames.index(x)
    distance_row = pair_distances[row_index]
    nh = set([name for name, d in zip(list_IndividualNames, distance_row) if d >= theta])
    return nh


# gets the set of all individuals belonging to a concept
query = 'wineZinfandel'
class_set = set([name for name, value in zip(list_IndividualNames, input_data[query]) if value == 1])

lower_approx = []
upper_approx = []
excluded = []

for ind in list_IndividualNames:
    neigh_set = neighbourhood(ind)
    if (neigh_set & class_set) == neigh_set:
        print(neigh_set, class_set)
        lower_approx.append(ind)
    elif len(neigh_set & class_set) > 0:
        upper_approx.append(ind)
    else:
        excluded.append(ind)

print("Approximation for concept \"%s\" with theta=%f" % (query, theta))
print("Upper approx: ", upper_approx)
print("Lower approx: ", lower_approx)
print("Excluded individuals: ", excluded)

# print("Final check:", True if (len(upper_approx) + len(lower_approx) + len(excluded)) == len(list_IndividualNames) else False)

# exit the program
exit()
