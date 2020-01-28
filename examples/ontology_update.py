import tkinter
import random
from tkinter import filedialog
import owlready2 as or2
import types


tkinter.Tk().withdraw()
input_file_path = tkinter.filedialog.askopenfilename()
onto = or2.get_ontology(input_file_path).load()

sample_class = list(onto.classes())[0]
sample_individuals = list(onto.individuals())[:10]
random.shuffle(sample_individuals)


with onto:
    test_class_name = "TestConcept"
    test_class = types.new_class(test_class_name, (or2.Thing,))

    for i in sample_individuals:
        i.is_a.append(test_class)



for cl in onto.classes():
    if cl == test_class:
        print("%s<====" % cl)
    else:
        print(cl)

Z = onto['CotturiZinfandel']
for s, p, o in onto.get_triples():
    print(s, p, o)
exit()
