from proxima import utils, ontology, approximation

input_file_path = utils.open_file("Select an OWL ontology to open:", 'owl')

# terminate if there's no file path provided
if not input_file_path:
    exit()

# loads the ontology into the manager
onto_mgr = ontology.OntologyManager(input_file_path)

# loads the table from a file
table_path = utils.open_file("Select a prebuilt projection table:", 'csv')
onto_mgr.build_table(mode='from_file', table_path=table_path, check_consistency=True)

# Build an approximator
approximator = approximation.ToleranceApproximator()

# let's build a set of positive and negative elements
positive_names = set(onto_mgr.get_individuals(mode='by_class', class_name='wine:Zinfandel', value='TRUE', as_strings=True))

# modify the set of positive examples in order to fake a less crisp set
positive_names.add('wine:LongridgeMerlot')

# print the positive and negative sets
print("Positive examples:", positive_names)

# manual mode, doesn't insert the approximations back into the ontology
approximator.fit(onto_mgr.get_mapped_table())
upper, lower = approximator.approximate(examples=positive_names, theta=0.4, beta=0.)

print("Upper approximation:", upper)
print("Lower Approximation:", lower)

# good approximation rule #1: the example set must ALWAYS be a subset of the upper approximation
rule_1 = positive_names.issubset(upper)

# good approximation rule #2: the example set must ALWAYS be a superset of the lower approximation
rule_2 = positive_names.issuperset(lower)

# good approximation rule #3: the lower approximation must not be empty
rule_3 = len(lower) > 0

# good approximation rule #4: the upper approximation must not contain elements outside the LCS
positive_inds = set(onto_mgr.get_individuals(mode='by_names', names=positive_names))
LCS = onto_mgr.get_LCS_construct(positive_inds)
LCS_set = set(map(str,onto_mgr.onto.search(type=LCS)))
print("LCS set:", LCS_set)
rule_4 = upper.issubset(LCS_set)
if rule_1 & rule_2 & rule_3 & rule_4:
    print("This is a good approximation!")
else:
    print("This approximation is not correct...")