import tkinter
from tkinter import filedialog
from proxima import approximation, ontology

tkinter.Tk().withdraw()
input_file_path = tkinter.filedialog.askopenfilename(title="Select an OWL ontology file:")

# terminate if there's no file path provided
if not input_file_path:
    exit()

# loads the ontology into the manager
onto_mgr = ontology.OntologyManager(input_file_path)

load_from_file = True

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

# print the positive and negative sets
print("Positive examples:", positive)
print("Negative examples:", negative)
# ______________________________________________________________________________________________________________________
# Run the approximator with the given example sets
upper_names, lower_names = approximator.approximate(positive, negative, theta=0.8)

upper = onto_mgr.search_individuals(names=upper_names)
lower = onto_mgr.search_individuals(names=lower_names)

print("Upper approx:", upper_names, "\n", "Lower approx:", lower_names)
onto_mgr.insert_approximated_concept('SelectWine', upper, lower)

output_file = tkinter.filedialog.asksaveasfilename(defaultextension=".owl")
onto_mgr.export_ontology(output_file)

print("Exported modified ontology as ", output_file)
