import numpy as np

import proxima
import owlready2 as or2
import tkinter
from tkinter import filedialog
import pandas as pd

from proxima import approximation as prx

tkinter.Tk().withdraw()
input_file_path = tkinter.filedialog.askopenfilename()

# terminate if there's no file path provided
if not input_file_path:
    exit()

remapper = prx.DataRemapper({'TRUE': 1, 'FALSE': 0, 'UNCERTAIN': 0.5})
# open and load the ontology at the defined filepath
#onto = or2.get_ontology(input_file_path).load()

# get the dataframe with the projection
#proj_table = prx.project_ontology(onto, remapper)

proj_table = pd.read_csv(input_file_path, delimiter=";", index_col=0).apply(np.vectorize(remapper))

# Build an approximator
approximator = prx.ToleranceRoughApproximator(projection_table=proj_table, variance=0.95)

# provide a set of positive and negative elements
positive = set([name for name, value in proj_table['wine:Zinfandel'].iteritems() if value == remapper('TRUE')])
negative = set([name for name, value in proj_table['wine:Wine'].iteritems() if value == remapper('FALSE')])

# modify the set of positive examples in order to fake a less crisp set
positive.add('wine:MariettaPetiteSyrah')
positive.add('wine:FormanCabernetSauvignon')

print("Positive examples:", positive)
print("Negative examples:", negative)

# Run the approximator with the given example sets
upper, lower = approximator(positive, negative)



