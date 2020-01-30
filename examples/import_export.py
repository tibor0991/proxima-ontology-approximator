import owlready2 as or2

onto = or2.get_ontology(r"C:\Users\Gianf\Dropbox\Tesi\Ontologie\wine.owl").load()
onto.save(r"C:\Users\Gianf\Dropbox\Tesi\Ontologie\wine_exported.owl")