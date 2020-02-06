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
# sets the global render func
or2.set_render_func(render_colon)

class OntologyManager:
    def __init__(self, path):
        # loads an ontology and stores classes and individuals in two dictionaries
        _onto = or2.get_ontology(path).load()
        _classes = {str(c): c for c in _onto.classes()}
        _individuals = {str(i): i for i in _onto.individuals()}

        # saves the variables in the object
        self.onto = _onto
        self.projection_table = None
        self.classes = _classes
        self.individuals = _individuals

    def build_table(self, **kwargs):
        # build the projection table
        _table = pd.DataFrame(data=_PT_UNCERTAIN, index=self.individuals.keys(), columns=self.classes.keys())
        # run the reasoner
        use_reasoner = kwargs['use_reasoner'] or False
        if use_reasoner:
            with self.onto:
                # runs a consistency check
                or2.sync_reasoner_pellet()
                pass

        projection_mode = kwargs['mode'] or 'disjunction'

        if projection_mode == 'disjunction':
            # builds the disjunction lookup dict
            disjunctions = {c: set() for c in self.classes.keys()}
            for pair in self.onto.disjoint_classes():
                first, second = pair.entities
                disjunctions[str(first)].add(second)
                disjunctions[str(second)].add(first)

            # faster method
            for c_name, c in self.classes.items():
                true_set = set(c.instances())
                false_set = set()
                for d in disjunctions[c_name]:
                    false_set = false_set.union(d.instances())
                for t in true_set:
                    _table.at[str(t), c_name] = _PT_TRUE
                for f in false_set:
                    _table.at[str(f), c_name] = _PT_FALSE

        elif projection_mode == 'from_file':
            table_path = kwargs['table_path']
            if table_path:
                _table = pd.read_csv(table_path, delimiter=";", index_col=0)
            else:
                raise

        elif projection_mode == 'reasoner':
            #check if __fast__ reasoning is enabled
            fast_build = kwargs['use_fast_reasoning'] or False
            # list that holds all the references to the not_ classes
            classes_with_complements = []
            with self.onto:
                for c_name, c_item in self.classes.items():
                    neg_class = types.new_class('NOT_' + c_item.name, (or2.Thing,))
                    neg_class.equivalent_to = [or2.Not(c_item)]
                    if fast_build:
                        or2.AllDisjoint([c_item, neg_class])
                    classes_with_complements.append((c_name, c_item, neg_class))

            # run the reasoner
            with self.onto:
                or2.sync_reasoner_pellet()

            # sets the cells with the respective value
            for c_name, c, not_c in classes_with_complements:
                true_set = set(c.instances())
                false_set = set(not_c.instances())

            for t in true_set:
                _table.at[str(t), c_name] = _PT_TRUE
            for f in false_set:
                _table.at[str(f), c_name] = _PT_FALSE

        # assigns the newly-built table to the inner parameter
        self.projection_table = _table
        pass

    def export_table(self, path_to_csv):
        self.projection_table.to_csv(path_to_csv, header=True)
        return

    def get_mapped_table(self, remapper=default_remapper):
        return self.projection_table.apply(np.vectorize(remapper))

    def insert_approximated_concept(self, concept_name, upper_elements, lower_elements):
        raise NotImplementedError()

    def search_individuals(self, class_name=None, requested_value=None, names=None, as_strings=False):
        raise NotImplementedError()

    def export_ontology(self, path):
        self.onto.save(path)

    def get_coverage(self, examples):
        coverage = set()
        for e in examples:
            for c in e.is_a:
                coverage.add(str(c))
        return [v for k, v in self.classes.items() if k in coverage]


if __name__ == '__main__':
    import tkinter
    from tkinter import filedialog

    tkinter.Tk().withdraw()
    onto_load_path = tkinter.filedialog.askopenfilename(title="Select an OWL ontology file:")
    ont_mgr = OntologyManager(onto_load_path)

    ont_mgr.build_table(use_reasoner=True)
    export_disjoint = tkinter.filedialog.asksaveasfilename(defaultextension=".csv")
    ont_mgr.export_table(export_disjoint)
