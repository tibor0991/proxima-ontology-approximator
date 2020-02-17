import types

import numpy as np
import owlready2 as owl
from sklearn.pipeline import Pipeline

from proxima import utils, ontology, approximation
import pandas as pd
from sklearn import decomposition, preprocessing, model_selection, svm
import pickle

# __________________________________________________ONTOLOGY____________________________________________________________
input_ontology = utils.open_file("Open OWL Ontology", 'owl')
onto = owl.get_ontology(input_ontology).load()

with onto:
    owl.sync_reasoner_pellet()

classes = {str(c): c for c in onto.classes()}
individuals = {str(i): i for i in onto.individuals()}

table_index = individuals.keys()
table_columns = classes.keys()

table_path = utils.open_file("Open projection table", 'csv')
projection_table = pd.read_csv(table_path, sep=';', index_col=0).apply(np.vectorize(ontology.default_remapper))

# get list of examples
# search_list = ['wine:RoseDAnjou', 'wine:SevreEtMaineMuscadet', 'wine:ChateauMargaux', 'wine:ChateauLafiteRothschildPauillac', 'wine:ElyseZinfandel', 'wine:ChiantiClassico']
# concept_examples = [individuals[s] for s in search_list]
concept_examples = set(onto.search(type=classes['wine:Zinfandel']))
concept_examples.add(individuals['wine:LongridgeMerlot'])

# get LCS of examples
coverage = set()
for e in concept_examples:
    for c in e.is_a:
        coverage.add(c)
LCS = owl.Or(coverage)
LCS_set = set(onto.search(type=LCS))

# get subset of dataset samples over which the model should be built
training_samples = [name for name, entity in individuals.items() if entity in concept_examples or entity not in LCS_set]
training_table = projection_table.loc[training_samples]
target_class = pd.Series({name: 'POSITIVE' if individuals[name] in concept_examples else 'NEGATIVE'
                          for name in training_samples})

# _____________________________________________TRAINING_________________________________________________________________
process_steps = [('scaling', preprocessing.RobustScaler()),
                 ('reduction', decomposition.PCA(whiten=True)),
                 ('estimator', svm.SVC(kernel='linear'))]

data_pipe = Pipeline(process_steps)

train = utils.ask_boolean("Training", "Train over data?")
if train:
    # search values for n_components (PCA)
    lPCA_n_components_range = list(np.logspace(np.log(0.20), np.log2(0.95), 10, base=2)) + \
                              list(np.floor(np.logspace(1, np.log2(len(classes)), 8, base=2)).astype(int))

    # search values for C
    C_range = list(np.logspace(-2, 10, 13))

    # store ranges in the params dict
    param_grid = [
        {'reduction__n_components': lPCA_n_components_range,
         'estimator__C': C_range}
    ]
    print(param_grid)

    cv = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = model_selection.GridSearchCV(estimator=data_pipe, param_grid=param_grid, cv=cv, verbose=100, n_jobs=-1)
    grid.fit(training_table, target_class)

    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    best_params = grid.best_params_

    # exports the best params
    pipe_params_export = utils.save_file("Export parameters:", 'pkl')
    with open(pipe_params_export, mode='wb') as out:
        pickle.dump(best_params, out, pickle.HIGHEST_PROTOCOL)

else:
    # load pre-computed parameters
    pipe_params_import = utils.open_file("Open a previously exported parameter pack", 'pkl')
    if pipe_params_import:
        with open(pipe_params_import, mode='rb') as f:
            best_params = pickle.load(f)
    else:
        best_params = {'estimator__C': 1000000.0,
                       'reduction': decomposition.PCA(copy=True, iterated_power='auto',
                                                      n_components=0.20269956628651734,
                                                      random_state=None, svd_solver='auto', tol=0.0, whiten=True),
                       'reduction__n_components': 0.20269956628651734,
                       'scaling': preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)}

        pipe_params_export = utils.save_file("Export parameters:", 'pkl')
        with open(pipe_params_export, mode='wb') as out:
            pickle.dump(best_params, out, pickle.HIGHEST_PROTOCOL)

print("Using parameters:", best_params)

data_pipe.set_params(**best_params)
data_pipe.fit(training_table, target_class)
clf = data_pipe['estimator']

w_norm = np.linalg.norm(clf.coef_)

half_dist = 1. / w_norm  # distance between supports and hard margin in the linear svm

print("The distance between the support and the margin is:", half_dist)

# search the neighbourhood of each example item
U_data = data_pipe[:2].fit_transform(projection_table)
U = pd.DataFrame(data=U_data, index=projection_table.index)

examples_labels = [str(i) for i in concept_examples]
LCS_labels = [str(i) for i in LCS_set]

LCS_U = U.loc[LCS_labels]

# LCS Distance Matrix (DM)
from sklearn import metrics

LCS_dm_sq = metrics.pairwise_distances(LCS_U, metric='euclidean')
LCS_DM = pd.DataFrame(data=LCS_dm_sq, index=LCS_labels, columns=LCS_labels)

examples_set = set(examples_labels)
examples_neighbourhoods = {label: set([other_label
                                       for other_label in LCS_labels
                                       if LCS_DM.at[label, other_label] < half_dist])
                           for label in examples_set}

boundary_set = set.union(*examples_neighbourhoods.values()) - examples_set
boundary_neighbourhoods = {label: set([other_label
                                       for other_label in LCS_labels
                                       if LCS_DM.at[label, other_label] < half_dist])
                           for label in boundary_set}

all_neighbourhoods = {**boundary_neighbourhoods, **examples_neighbourhoods}
print("Example individuals:", examples_set)
print("Boundary individuals:", boundary_set)

concept_name = 'MyPersonalSelection'
upper, lower = [], []
with onto:
    # builds the similarity relation
    relation_name = "isSimilar_wrt_" + concept_name
    sim_relation = types.new_class(relation_name, (owl.SymmetricProperty, owl.ReflexiveProperty,))

    # builds the upper approximation class
    _upperClass = types.new_class("Possibly_" + concept_name, (owl.Thing,))
    _upperClass.is_a = [LCS]
    _upperClass.equivalent_to.extend([sim_relation.some(owl.OneOf(concept_examples))])  # existential restriction
    # builds the lower approximation class
    _lowerClass = types.new_class("Definitively_" + concept_name, (owl.Thing,))
    _lowerClass.equivalent_to.extend([sim_relation.only(owl.OneOf(concept_examples))])  # universal restriction

    # for each key in the pairs dictionary, add a similarity relation
    for label, neighbourhood in all_neighbourhoods.items():
        center = individuals[label]
        neighbours = [individuals[neighbour] for neighbour in neighbourhood]

        # should result in: center.isSimilar_wrt_{concept_name}.extend(neighbours)
        append_to_property = 'center.' + relation_name + '.extend(neighbours)'
        eval(append_to_property)

        print("Neighbourhood of %s:" % label, neighbourhood)
        m_index = approximation.rough_membership(neighbourhood, examples_set)
        print("\tInclusion index:", m_index)
        if m_index == 1:
            lower.append(label)
            center.is_a.append(_lowerClass)
        if m_index > 0:
            upper.append(label)
            center.is_a.append(_upperClass)

print("Upper approximation:", upper)
print("Lower approximation:", lower)

# finally, export the ontology
ontology_export_path = utils.save_file("Export ontology to:", 'owl')
onto.save(ontology_export_path)