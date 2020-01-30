import pandas as pd



"""
Version space learning through candidate elimination needs:
- A table representing all features, plus a class feature (enjoySports in the toy example by Mitchell)
- (in this case) a list of possible values an hypothesis can assume (true, false or uncertain)

Returns the most specific hypothesis S and the least generic hypothesis G
- while S is a single tuple, G is a list of tuples linked together through an OR
"""

#build positive examples:
class VersionSpaceLearner:
    def __init__(self):
        pass

    def generalize(self, cx, s, index):
        for i in index:
            if cx[i] != s[i]:
                s[i] = '?'


    def specialize(self, cx, s, g, index):
        g_specs = []
        for i in index:
            if g[i] == '?' and cx[i] != s[i]:
                new_spec = g.copy()
                new_spec[i] = cx[i]
                g_specs.append(new_spec)
        return g_specs


    def consistent(self, s, g, index):
        is_consistent = True
        for i in index:
            if g[i] != '?':
                is_consistent = (g[i] == s[i] or s[i] == '?')
                if not is_consistent:
                    break
        return is_consistent

if __name__ == "__main__":
    projection_table = pd.read_csv(r"C:\Users\Gianf\Dropbox\Tesi\newTable.csv", delimiter=';', index_col=0)

    positive_examples = [name for name, value in projection_table['wine:Zinfandel'].iteritems() if value == 'TRUE']
    negative_examples = [name for name, value in projection_table['wine:Wine'].iteritems() if value == 'FALSE']
