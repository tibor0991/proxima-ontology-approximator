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
    onto_mgr.build_table(mode='from_file', table_path=table_path)
else:
    onto_mgr.build_table(use_reasoner=True)

# Build an approximator
approximator = approximation.ToleranceApproximator()

# provide a set of positive and negative elements
positive_names = set(onto_mgr.get_individuals(mode='by_class', class_name='wine:Zinfandel', value='TRUE', as_strings=True))

# modify the set of positive examples in order to fake a less crisp set
positive_names.add('wine:LongridgeMerlot')

positive = list(onto_mgr.get_individuals(mode='by_names', names=positive_names))

# print the positive and negative sets
print("Positive examples:", positive)

# Run the approximator with the given example sets
upper, lower = onto_mgr.approximate_concept('MyPersonalSelection', positive, approximator, theta=0.6)

print("Upper approx:", upper, "\n\r", "Lower approx:", lower)

output_file = tkinter.filedialog.asksaveasfilename(defaultextension=".owl")
onto_mgr.export_ontology(output_file)

print("Exported modified ontology as ", output_file)
