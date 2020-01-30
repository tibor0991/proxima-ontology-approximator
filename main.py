import tkinter
from tkinter import filedialog

import pandas as pd

from proxima import approximation, ontology

tkinter.Tk().withdraw()
input_file_path = tkinter.filedialog.askopenfilename(title="Select an OWL ontology file:")

# terminate if there's no file path provided
if not input_file_path:
    exit()

# loads the ontology into the manager
onto_mgr = ontology.OntologyManager(input_file_path)

load_from_file = False

if load_from_file:
    table_path = tkinter.filedialog.askopenfilename(title="Select a prebuilt projection table:")
    onto_mgr.load_table(table_path)
else:
    onto_mgr.build_table(use_reasoner=False)

# Build an approximator
approximator = approximation.ToleranceApproximator()
approximator.fit(onto_mgr.get_mapped_table())

# provide a set of positive and negative elements
positive = onto_mgr.search_individuals(class_name='wine:Zinfandel', requested_value='TRUE', as_strings=True)
negative = onto_mgr.search_individuals(class_name='wine:Wine', requested_value='FALSE', as_strings=True)

# modify the set of positive examples in order to fake a less crisp set
positive.add('wine:LongridgeMerlot')

# ______________________________________________________________________________________________________________________
# get the coverage of the positive set
coverage = onto_mgr.get_coverage(positive)
print("Coverage", coverage)

# pop one (random?) positive from the set of positives before proceeding
first_positive_index = positive.pop()

candidates_indexes = positive.union(negative)
candidate_table = onto_mgr.projection_table.loc[candidates_indexes, coverage]

# candidate elimination:
# S - most specific hypothesis
# G - least general hypothesis

for p in positive:
    candidate_table.loc[p, 'CLASS'] = '+++'
for n in negative:
    candidate_table.loc[n, 'CLASS'] = '---'

print(candidate_table)


def generalize(cx, s, index):
    for i in index:
        if cx[i] != s[i]:
            s[i] = '?'


def specialize(cx, s, g, index):
    g_specs = []
    for i in index:
        if g[i] == '?' and cx[i] != s[i]:
            new_spec = g.copy()
            new_spec[i] = cx[i]
            g_specs.append(new_spec)
    return g_specs


def consistent(s, g, index):
    is_consistent = True
    for i in index:
        if g[i] != '?':
            is_consistent = (g[i] == s[i] or s[i] == '?')
            if not is_consistent:
                break
    return is_consistent


S = onto_mgr.projection_table.loc[first_positive_index, coverage]
G = list()
G.append(pd.Series(data=['?' for c in coverage], index=coverage))
print("S:", S.tolist())
print("G:", [g.tolist() for g in G])

for c, cx in candidate_table.iterrows():
    print("cx:", cx.tolist())
    if cx['CLASS'] == '+++':
        # generalize the most specific hypothesis
        generalize(cx, S, coverage)
        # remove the general hypothesises that do not satisfy the most specific one
        consistent_G = []
        for g in G:
            if consistent(S, g, coverage):
                consistent_G.extend(g)
        G.clear()
        G = consistent_G
    else:
        new_G = []
        for g in G:
            specializations = specialize(cx, S, g, coverage)
            if specializations:
                new_G.extend(specializations)
            else:
                new_G.append(g)
        G.clear()
        G = new_G

    print("S:", S.tolist())
    print("G:", [g.tolist() for g in G])

# print the positive and negative sets
print("Positive examples:", positive)
print("Negative examples:", negative)
# ______________________________________________________________________________________________________________________
# Run the approximator with the given example sets
upper_names, lower_names = approximator.approximate(positive, negative, theta=0.9)

upper = onto_mgr.search_individuals(names=upper_names)
lower = onto_mgr.search_individuals(names=lower_names)

print("Upper approx:", upper_names, "\n", "Lower approx:", lower_names)
onto_mgr.insert_approximated_concept('SelectWine', upper, lower)

output_file = tkinter.filedialog.asksaveasfilename(defaultextension=".owl")
onto_mgr.export_ontology(output_file)

print("Exported modified ontology as ", output_file)
