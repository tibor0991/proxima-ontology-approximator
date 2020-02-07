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
        """
        Builds (or loads) the projection table with the given parameters.

        :key use_reasoner: shall a consistency check be run before building the projection table? (default is False)
        :key mode: specifies how the projection table must be built:
            - 'disjunction': The projection value is computed by looking up the disjunction pairs (faster, but prone to errors);
            - 'from_file': The projection table is loaded from the path provided in the 'table_path' key;
            - 'reasoner': The table is built using the results obtained from the reasoner (slowest but correct).
        :key table_path: if mode = 'from_file', specifies the file path from which the table must be loaded
        :return:
        """
        # build the projection table
        _table = pd.DataFrame(data=_PT_UNCERTAIN, index=self.individuals.keys(), columns=self.classes.keys())
        # run the reasoner
        use_reasoner = kwargs.get('use_reasoner') or False
        if use_reasoner:
            with self.onto:
                # runs a consistency check
                or2.sync_reasoner_pellet()
                pass

        projection_mode = kwargs.get('mode') or 'reasoner'

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
        # Build using the reasoner
        elif projection_mode == 'reasoner':
            # list that holds all the references to the not_ classes
            classes_with_complements = []
            with self.onto:
                for c_name, c_item in self.classes.items():
                    neg_class = types.new_class('NOT_' + c_item.name, (or2.Thing,))
                    neg_class.equivalent_to = [or2.Not(c_item)]
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
        else:
            raise Exception("ERROR: Unrecognized mode '%s'." % projection_mode)

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
            _upperClass = types.new_class("Possibly_"+concept_name, (or2.Thing,))
            _upperClass.is_a = [LCS_construct]  # LCS subsumes upper approximation
            _lowerClass = types.new_class("Definitively_"+concept_name, (_upperClass,))

            for u in upper_set:
                u.is_a.append(_upperClass)
            for l in lower_set:
                l.is_a.append(_lowerClass)

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

        equivalence_construct = or2.Or(coverage)
        return equivalence_construct

    def approximate_concept(self, name, examples, approximator, theta):
        concept_set = set([str(ex) for ex in examples])
        approximator.fit(self.get_mapped_table())
        upper_names, lower_names = approximator.approximate(concept_set, theta)
        upper = [ind for name, ind in self.individuals.items() if name in upper_names]
        lower = [ind for name, ind in self.individuals.items() if name in lower_names]
        eq_construct = self.get_LCS_construct(examples)
        self.insert_approximated_concept(name, upper, lower, eq_construct)
        return upper, lower