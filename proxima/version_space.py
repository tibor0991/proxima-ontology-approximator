import pandas as pd



"""
Version space learning through candidate elimination needs:
- A table representing all features, plus a class feature (enjoySports in the toy example by Mitchell)
- (in this case) a list of possible values an hypothesis can assume (true, false or uncertain)

Returns the most specific hypothesis S and the least generic hypothesis G
"""
"""
#build positive examples:
class VersionSpaceLearner:
    def __init__(self, data):
        self.data = data
        pass

    def generalize(self, cx, s):
        for i in range(0, len(s)):
            if cx[i] != s[i]:
                s[i] = '?'


    def specialize(self, cx, s, g):
        for i in range(0, len(s)):
            if g[i] == '?' and cx[i] != s[i]:
                new_spec = g.copy()
                new_spec[i] = s[i]
                yield new_spec


    def consistent(self, s, g):
        is_consistent = True
        for i in range(0, len(s)):
            if s[i] == '?':
                is_consistent = (g[i] == '?')
                if not is_consistent:
                    break
        if is_consistent:
            print(g, "is consistent with", s)
        else:
            print(g, "is not consistent, must be pruned.")

        return is_consistent

    def __call__(self):
        S = list(self.data.iloc[0])[0:5]
        self.data.drop(0)
        G = [['?' for i in range(0, 5)]]

        for _, cx in self.data.iterrows():
            if cx['CLASS'] == '+':
                self.generalize(cx, S)
            else:
                expanded_G = []
                for g in G:
                    expanded_G.extend(list(self.specialize(cx, S, g)))
                G.clear()
                G = expanded_G

            consistent_G = []
            for g in G:
                if self.consistent(S, g):
                    consistent_G.append(g)
            G.clear()
            G = consistent_G

            print("S:",S)
            print("G:", [g for g in G])
            # stop conditions
            # 1) S = G
            if S in G:
                print("The search is over")
                break
            else:
                print("The search continues...")
        return S, G
"""


def test_hypothesis(ex, h):
    """
    Returns true if the example ex is consistent with the hypothesis h
    """
    single_check = lambda ex_i, h_i: (ex_i == h_i) or (h_i == '?')
    result = all([single_check(e, h_i) for e, h_i in zip(ex, h)])
    print(ex, '==' if result else '!=', h)
    return result

def learn_version_space(examples: pd.DataFrame, values=set(['TRUE', 'UNCERTAIN', 'FALSE'])):
    first_positive = examples.iloc[]
    for e, ex in examples.iterrows():
        if ex['CLASS'] == '+':
            # for each g in G, if test(ex, g) is negative remove g
            for g in G:
                if not test(ex, g):
                    # remove g from G
            for s in S:
                if not test(ex, s):
                    # remove s from S
                    # insert s'=generalize(s, ex) into S
            # remove non-maximal hypothesis (???) from S
        elif ex['CLASS'] == '-':
            if test(ex, s):
                # remove s from S
            if test(ex, g):
                # remove g from G
                # insert g' = specialize(g) in G




if __name__ == "__main__":
    test_data = [{'origin': 'Japan', 'manufacturer':'Honda', 'color':'blue', 'decade':'1980', 'type': 'economy', 'CLASS':'+'},
                 {'origin': 'Japan', 'manufacturer':'Toyota', 'color':'green', 'decade':'1970', 'type': 'sports', 'CLASS':'-'},
                 {'origin': 'Japan', 'manufacturer':'Toyota', 'color':'blue', 'decade':'1990', 'type': 'economy', 'CLASS':'+'},
                 {'origin': 'USA', 'manufacturer':'Chrysler', 'color':'red', 'decade':'1980', 'type': 'economy', 'CLASS':'.'},
                 {'origin': 'Japan', 'manufacturer':'Honda', 'color':'white', 'decade':'1980', 'type': 'economy', 'CLASS':'+'}]
    candidate_table = pd.DataFrame(test_data)
    learn_version_space(candidate_table)

