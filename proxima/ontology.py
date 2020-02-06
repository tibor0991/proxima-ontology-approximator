import owlready2 as or2
import pandas as pd
import numpy as np
import types


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
    def __init__(self, path, remapper=default_remapper, **kwargs):
        # loads an ontology and stores classes and individuals in two dictionaries
        or2.set_render_func(render_colon)  # WARNING: unset me
        _onto = or2.get_ontology(path).load()

        # consistency check
        with _onto:
            or2.sync_reasoner_pellet()

        # saves a pair of name/value for each class and individual
        _classes = {str(c): c for c in _onto.classes()}
        _individuals = {str(i): i for i in _onto.individuals()}

        # builds the projection table
        _projection_table = pd.DataFrame(data=_PT_UNCERTAIN, index=_individuals.keys(),
                                        columns=_classes.keys())

        # here goes the projection method


        _projection_table = _projection_table.apply(np.vectorize(remapper))

        """ # scheduled for deletion
        
        projection_table = projection_table.apply(np.vectorize(remapper))
        """
        print(list(onto.classes()))
        or2.set_render_func(or2.default_render_func)  # unsets the render function

        # saves the variables in the object
        self.onto = onto
        self.projection_table = projection_table
        self.classes = classes
        self.individuals = individuals
        self.remapper = remapper

    def _projection_by_outsourced_table(self, path, table):
        table =
        return

    def _projection_by_disjunction(self, onto, classes, table):
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
                table.at[str(t), c_name] = _PT_TRUE
            for f in false_set:
                table.at[str(f), c_name] = _PT_FALSE
        # 6.9 seconds for .instances() method
        return

    def _projection_by_reasoning(self, onto, classes, table):
        # list that holds all the references to the not_ classes
        classes_with_complements = []
        with onto:
            for c_name, c_item in classes.items():
                neg_class = types.new_class('NOT_' + c_item.name, (or2.Thing,))
                neg_class.equivalent_to = [or2.Not(c_item)]
                or2.AllDisjoint([c_item, neg_class])
                classes_with_complements.append((c_name, c_item, neg_class))

        # run the reasoner
        with onto:
            or2.sync_reasoner_pellet()

        for c_name, c, n in classes_with_complements:
            true_set = set(c.instances())
            false_set = set(n.instances())

        for t in true_set:
            table.at[str(t), c_name] = _PT_TRUE
        for f in false_set:
            table.at[str(f), c_name] = _PT_FALSE


    def insert_approximated_concept(self, concept_name, upper_elements, lower_elements):

        # build the two approximation classes
        upper_name = "Possibly_" + concept_name
        lower_name = "Definitively_" + concept_name
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
