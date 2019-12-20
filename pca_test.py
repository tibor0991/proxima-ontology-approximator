import csv
import json

from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition, neighbors, preprocessing

# %% Load the stored data from the .json
input_file_path = r"C:\Users\Gianf\Desktop\table.csv"

with open(input_file_path) as f:
    raw_X = np.genfromtxt(f, delimiter=',', converters={0: lambda s: str(s, 'utf-8'), } )
print (raw_X)

print("Shape of the original dataset:", raw_X.shape)
print("individual names: ", raw_X[:, 0])
# scale the data
scaler = preprocessing.StandardScaler()
scaler.fit(raw_X)
scaled_X = scaler.transform(raw_X)
print("Shape of the scaled dataset:", scaled_X.shape)

# apply PCA
PCA = decomposition.PCA(0.95)
PCA.fit(scaled_X)
red_X = PCA.transform(scaled_X)
print("Shape of the PCA-transformed dataset:", red_X.shape)

# build a covariance matrix from red_X
cov_mat = np.cov(red_X, rowvar=False)

print("Shape of the covariance matrix: ", cov_mat.shape)

# %% Compute Mahalanobis' distance pairwise
dist = neighbors.DistanceMetric.get_metric('mahalanobis', V=cov_mat)
pair_distances = dist.pairwise(red_X)

# fake normalization (just remaps the values in a [0...1] range)
max_dist = np.max(pair_distances)
pair_distances /= max_dist
pair_distances = np.subtract(1, pair_distances)

print("Shape of pair sets:", pair_distances.shape)
with np.printoptions(precision=3, suppress=True):
    print(pair_distances)

"""
#%% Plotting the dataset along two components obtained from another PCA filter

plotPCA = decomposition.PCA(n_components=2)
plot_X = plotPCA.fit_transform(raw_X)

plt.plot(plot_X, 'ro')
plt.show()
"""
# %% exit the program
exit()
