import tkinter
from tkinter import filedialog

from proxima import approximation, ontology

tkinter.Tk().withdraw()
input_file_path = tkinter.filedialog.askopenfilename()

# terminate if there's no file path provided
if not input_file_path:
    exit()

onto_mgr = ontology.OntologyManager(input_file_path)

# Build an approximator
approximator = approximation.ToleranceApproximator()
approximator.fit(onto_mgr.projection_table)

# provide a set of positive and negative elements
positive = onto_mgr.search_individuals('wine:Zinfandel', 'TRUE')
negative = onto_mgr.search_individuals('wine:Wine', 'FALSE')

# modify the set of positive examples in order to fake a less crisp set
# TODO: this, but with individuals instead of names

print("Positive examples:", positive)
print("Negative examples:", negative)

# Run the approximator with the given example sets
upper, lower = approximator(positive, negative)

onto_mgr.insert_approximated_concept('SelectWine', upper, lower) # TODO: this does not work yet!!!

