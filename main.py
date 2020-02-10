from proxima import approximation, ontology, utils

input_file_path = utils.open_file("Select an OWL ontology to open:", 'owl')

# terminate if there's no file path provided
if not input_file_path:
    exit()

# loads the ontology into the manager
onto_mgr = ontology.OntologyManager(input_file_path)

load_from_file = utils.ask_boolean("Projection table", "Load projection table from file?")

if load_from_file:
    table_path = utils.open_file("Select a prebuilt projection table:", 'csv')
    onto_mgr.build_table(mode='from_file', table_path=table_path, check_consistency=True)
else:
    onto_mgr.build_table(mode='reasoner', check_consistency=True)
    table_export = utils.save_file("Save projection table to:", 'csv')
    onto_mgr.export_table(table_export)

# Build an approximator
approximator = approximation.ToleranceApproximator()

# let's build a set of positive and negative elements
positive_names = set(onto_mgr.get_individuals(mode='by_class', class_name='wine:Zinfandel', value='TRUE', as_strings=True))

# modify the set of positive examples in order to fake a less crisp set
positive_names.add('wine:LongridgeMerlot')

positive = list(onto_mgr.get_individuals(mode='by_names', names=positive_names))

# print the positive and negative sets
print("Positive examples:", positive)

# Run the approximator with the given example sets
upper, lower = onto_mgr.approximate_concept('MyPersonalSelection', positive, approximator, theta=0.1)

print("Upper approx:", upper, "\n\r", "Lower approx:", lower)

output_file = utils.save_file("Save modified ontology to:", 'owl')
onto_mgr.export_ontology(output_file)

print("Exported modified ontology as ", output_file)
