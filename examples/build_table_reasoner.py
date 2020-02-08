from proxima import utils, ontology

# opens an OWL ontology
onto_path = utils.open_file("Open an OWL Ontology to project:", 'owl')

# sets up a manager object
ont_mgr = ontology.OntologyManager(onto_path)

# builds a table using Pellet
ont_mgr.build_table(mode='reasoner', check_consistency=True)

# exports the built table
export_path = utils.save_file("Save the projection table to:", 'csv')
ont_mgr.export_table(export_path)