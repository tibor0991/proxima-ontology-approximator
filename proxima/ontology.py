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
        use_reasoner = kwargs.get('use_reasoner') or False
        if use_reasoner:
            with self.onto:
                # runs a consistency check
                or2.sync_reasoner_pellet()
                pass

        projection_mode = kwargs.get('mode') or 'disjunction'

        if projection_mode == 'disjunction':
            # builds the disjunction lookup dict
            disjunctions = {c: set() for c in self.classes.keys()}
            for pair in self.onto.disjoint_classes():
                first, second = pair.entities
                disjunctions[str(first)].add(second)
                disjunctions[str(second)].add(first)

            # faster method, but less accurate
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
            # check if __fast__ reasoning is enabled
            fast_build = kwargs.get('use_fast_reasoning') or False
            # list that holds all the references to the not_ classes
            classes_with_complements = []
            with self.onto:
                for c_name, c_item in self.classes.items():
                    neg_class = types.new_class('NOT_' + c_item.name, (or2.Thing,))
                    neg_class.equivalent_to = [or2.Not(c_item)]
                    if not fast_build:
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
        _table = self.projection_table
        # sort alphabetically before exporting
        header = _table.columns
        _table = _table.sort_index(axis=1)
        _table = _table.sort_index()
        # export
        _table.to_csv(path_to_csv, sep=';', index_label='Name')
        return

    def get_mapped_table(self, remapper=default_remapper):
        return self.projection_table.apply(np.vectorize(remapper))

    def insert_approximated_concept(self, concept_name, upper_set, lower_set, LCS_construct):
        with self.onto:
            UpperClass = types.new_class("Possibly_"+concept_name, (LCS_construct,))
            LowerClass = types.new_class("Definitively_"+concept_name, (UpperClass,))

            for u in upper_set:
                u.is_a.append(UpperClass)
            for l in lower_set:
                l.is_a.append(LowerClass)

            or2.sync_reasoner_pellet()

    def get_individuals(self, mode, **kwargs):
        if mode == 'by_class':
            class_name = kwargs['class_name']
            search_value = kwargs['value']
            as_strings = kwargs.get('as_strings') or False
            ind_series = self.projection_table[class_name]
            for name, projection in ind_series.iteritems():
                if projection == search_value:
                    if as_strings:
                        yield name  # return the name as string
                    else:
                        yield self.individuals[name]    # return the name with that name

        elif mode == 'by_names':
            requested_names = kwargs['names']
            for name, ind in self.individuals.items():
                if name in requested_names:
                    yield ind

        else:
            raise Exception(r"Unrecognized mode: it should be either 'by_class' or 'by_name'.")

    def export_ontology(self, path):
        self.onto.save(path)

    def get_LCS_construct(self, examples):
        coverage = set()
        for e in examples:
            for c in e.is_a:
                coverage.add(c)

        equivalence_construct = or2.Nothing
        for c in coverage:
            equivalence_construct = equivalence_construct | c
        return [equivalence_construct]



if __name__ == '__main__':
    import sys
    import tkinter
    from tkinter import filedialog

    tkinter.Tk().withdraw()
    onto_load_path = tkinter.filedialog.askopenfilename(title="Select an OWL ontology file:")
    ont_mgr = OntologyManager(onto_load_path)

    if 'LCS_test' in sys.argv:
        csv_path = tkinter.filedialog.askopenfilename(title="Select a CSV table file:")
        ont_mgr.build_table(mode='from_file', table_path=csv_path)
        ex = list(ont_mgr.get_individuals(mode='by_class', class_name='wine:Zinfandel', value='TRUE'))
        print("Does it retrieve the examples?", ex)
        LCS = ont_mgr.get_LCS_construct(ex)
        print(LCS)
        LCS_set = ont_mgr.onto.search(type=LCS)
        ont_mgr.insert_approximated_concept('Zinfandel', ex, ex, LCS)
        output_file = tkinter.filedialog.asksaveasfilename(defaultextension=".owl")
        ont_mgr.export_ontology(output_file)
        exit()





    # csv_path = tkinter.filedialog.askopenfilename(title="Select a CSV table file:")
    export_disjoint = tkinter.filedialog.asksaveasfilename(defaultextension=".csv", filetypes=(('CSV record', '.csv'),))
    ont_mgr.build_table(mode='reasoner', use_fast_reasoning=True)
    ont_mgr.export_table(export_disjoint)
