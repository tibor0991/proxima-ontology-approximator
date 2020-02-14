import numpy as np
import owlready2 as owl
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

from proxima import utils, ontology
import pandas as pd
from sklearn import decomposition, preprocessing, model_selection, svm

input_ontology = utils.open_file("Open OWL Ontology", 'owl')
onto = owl.get_ontology(input_ontology).load()

with onto:
    owl.sync_reasoner_pellet()

classes = {str(c):c for c in onto.classes()}
individuals = {str(i):i for i in onto.individuals()}

table_index = individuals.keys()
table_columns = classes.keys()

table_path = utils.open_file("Open projection table", 'csv')
projection_table = pd.read_csv(table_path, sep=';', index_col=0).apply(np.vectorize(ontology.default_remapper))


process_steps = [('scaling', preprocessing.StandardScaler()),
                   ('reduction', decomposition.KernelPCA(kernel='rbf')),
                   ('estimator', svm.SVC(kernel='linear'))]

data_pipe = Pipeline(process_steps)

# search values for gamma
gamma_range = np.logspace(-9, 3, 13)

# search values for n_components
max_components_exp = np.floor(np.log2(len(individuals)))
components_range = np.logspace(1, max_components_exp, max_components_exp-1, base=2)

#search values for C
C_range = np.logspace(-2, 10, 13)

# list of class tags (for the SVM)
examples = set(onto.search(type=classes['wine:Zinfandel']))
examples.add(individuals['wine:LongridgeMerlot'])

# get LCS of examples
coverage = set()
for e in examples:
    for c in e.is_a:
        coverage.add(c)
LCS_construct = owl.Or(coverage)
LCS_set = set(onto.search(type=LCS_construct))

svm_labels = pd.Series(data={name:('POSITIVE' if i in LCS_set else 'NEGATIVE') for name, i in individuals.items()})


# store ranges in the params dict
param_grid = dict(reduction__gamma=gamma_range,
                  reduction__n_components=components_range,
                  estimator__C=C_range)

cv = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = model_selection.GridSearchCV(data_pipe, param_grid=param_grid, cv=cv, verbose=10)
grid.fit(projection_table, svm_labels)

print("Best parameters:", grid.best_params_)

decomposition.KernelPCA