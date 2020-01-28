import owlready2 as or2
import pandas as pd
import numpy as np
import types

import time


def render_colon(e: or2.entity) -> str:
    return "%s:%s" % (e.namespace.name, e.name)

class DataRemapper:
    def __init__(self, values_dict):
        self.values_dict = values_dict

    def __call__(self, value):
        return self.values_dict[value]

_PT_UNCERTAIN = 'UNCERTAIN'
_PT_TRUE = 'TRUE'
_PT_FALSE = 'FALSE'

default_remapper = DataRemapper({_PT_TRUE: 1, _PT_FALSE: 0, _PT_UNCERTAIN: 0.5})


class OntologyManager:
    def __init__(self, path, remapper=default_remapper):
        # loads an ontology and stores classes and individuals in two dictionaries
        or2.set_render_func(render_colon)  # WARNING: unset me
        onto = or2.get_ontology(path).load()
        classes = {str(c): c for c in onto.classes()}
        individuals = {str(i): i for i in onto.individuals()}

        # build the projection table
        projection_table = pd.DataFrame(data=_PT_UNCERTAIN, index=individuals.keys(),
                                        columns=classes.keys())
        # run the reasoner
        with onto:
            or2.sync_reasoner_pellet()

        # build the disjunction lookup dict
        disjunctions = {str(c): set() for c in onto.classes()}
        for pair in onto.disjoint_classes():
            first, second = pair.entities
            disjunctions[str(first)].add(second)
            disjunctions[str(second)].add(first)

        # faster method
        for c_name, c in classes.items():
            true_set = set(c.instances())
            false_set = set()
            for d in disjunctions[c_name]:
                false_set = false_set.union(d.instances())

            for t in true_set:
                projection_table.at[str(t), c_name] = _PT_TRUE
            for f in false_set:
                projection_table.at[str(f), c_name] = _PT_FALSE
        # 6.9 seconds for .instances() method
        projection_table = projection_table.apply(np.vectorize(remapper))

        or2.set_render_func(or2.default_render_func)  # unsets the render function

        # saves the variables in the object
        self.onto = onto
        self.projection_table = projection_table
        self.classes = classes
        self.individuals = individuals
        self.remapper = remapper

    def insert_approximated_concept(self, concept_name, upper_elements, lower_elements):
        """
        Inserts a pair of classes which represent an approximate concept in the form of C = < U, L >, where:
        - C is the approximate concept
        - U is the Upper approximation of C (named as "Possibly"+[base concept name] )
        - L is the Uower approximation of C (named as "Definitively"+[base_concept_name] )
        :param concept_name: Base name of the approximate concept
        :param upper_elements: sets of individuals that belong to the upper approximation
        :param lower_elements: sets of individuals that belong to the lower approximation
        :return: The modified ontology O'
        """
        # build the two approximation classes
        upper_name = "Possibly" + concept_name
        lower_name = "Definitively" + concept_name
        # add the classes to the ontology
        UpperClass = types.new_class(upper_name, (or2.Thing,))
        LowerClass = types.new_class(lower_name, (UpperClass,))

        # for each individual, add a relation to either or both classes
        for u_element in upper_elements:
            u_element.is_a.append(UpperClass)

        for l_element in lower_elements:
            u_element.is_a.append(LowerClass)

    def search_individuals(self, class_name, requested_value):
        series = self.projection_table[class_name]
        remap_value = self.remapper(requested_value)
        return [self.individuals[name] for name, value in series.iteritems() if value == remap_value]

if __name__ == '__main__':

    ont_mgr = OntologyManager(r"C:\Users\Gianf\Dropbox\Tesi\Ontologie\wine_disjoints.owl")
    csv_table = pd.read_csv(r"C:\Users\Gianf\Dropbox\Tesi\newTable.csv", delimiter=";", index_col=0).apply(np.vectorize(default_remapper))
    df1 = csv_table.sort_index()
    df2 = ont_mgr.projection_table[csv_table.columns].sort_index()

    print(df1.equals(df2))
