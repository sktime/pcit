from scipy import stats
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from support import descendants
from compare import pred_indep

class find_dag():
    def __init__(self, X, confidence=0.05, whichseed=1):
        self.number_tests = 1
        self.confidence = confidence
        self.cond_sets = dict()
        self.X = X
        self.skeleton = None
        self.n = self.X.shape[1]
        np.random.seed(whichseed)

    def powerset(self, n, p, q, i):
        xs = list(range(n))
        combinations = itertools.chain.from_iterable(itertools.combinations(xs, n) for n in range(len(xs) + 1))
        combinations = [x for x in combinations if len(x) == i and p not in x and q not in x]
        return combinations

    def find_forks(self, n):
        combinations = self.powerset(n, [], [], 3)
        combinations = [x for x in combinations if (self.skeleton[x[0], x[1]] + self.skeleton[x[0], x[2]] +
                                                    self.skeleton[x[1], x[2]] == 2) and (
                        2 not in (self.skeleton[x[0], x[1]], self.skeleton[x[0], x[2]],
                                  self.skeleton[x[1], x[2]]))]
        middle_node = [[i for i in x if np.sum(self.skeleton[i, x]) == 2] for x in combinations]
        edge_nodes = [[i for i in x if not np.sum(self.skeleton[i, x]) == 2] for x in combinations]

        return middle_node, edge_nodes

    def cond_indep_test(self, X, Y, Z='empty'):
        p_values_adj, = pred_indep(Y, X, z = Z)
        return p_values_adj

    def test_indep(self, p, q, i):
        n = self.X.shape[1]
        combinations = self.powerset(n, p, q, i)

        depend = 1
        for idx in combinations:
            p_val = self.cond_indep_test(self.X[:, p], self.X[:, q], self.X[:, idx])
            if p_val < self.confidence / self.number_tests:
                depend = 0
                self.cond_sets[p, q] = idx
                break
            self.number_tests += 1
        return depend

    def step1(self):
        old_skel = 0
        while old_skel < np.sum(self.skeleton == 2):
            old_skel = np.sum(self.skeleton == 2)
            for i in range(self.n):
                z = descendants(self.skeleton).dir_desc(i)
                if len(z) == 0:
                    continue
                for j in z:
                    y = descendants(self.skeleton).undir_neighb(j)
                    if len(y) == 0:
                        continue
                    for k in y:
                        self.skeleton[j, k] = 2
                        self.skeleton[k, j] = 0
                        break
                    break
                break

    def step2(self):
        old_skel = 0
        while old_skel < np.sum(self.skeleton == 2):
            old_skel = np.sum(self.skeleton == 2)
            for i in range(self.n):
                z = descendants(self.skeleton).all_desc(i)
                y = descendants(self.skeleton).undir_neighb(i)
                y = [x for x in y if x in z]
                if len(y) == 0:
                    continue
                self.skeleton[i, y] = 2
                self.skeleton[y, i] = 0
                break

    def step3(self):
        old_skel = 0
        while old_skel < np.sum(self.skeleton == 2):
            old_skel = np.sum(self.skeleton == 2)
            middle_node, edge_nodes = self.find_forks(self.n)
            for i in range(len(middle_node)):
                x_desc = descendants(self.skeleton).dir_desc(edge_nodes[i][0])
                y_desc = descendants(self.skeleton).dir_desc(edge_nodes[i][1])
                z_neighb = descendants(self.skeleton).undir_neighb(middle_node[i])
                w = list(set(x_desc) & set(y_desc) & set(z_neighb))
                if len(w) == 0:
                    continue
                self.skeleton[w, middle_node[i]] = 2
                self.skeleton[middle_node[i], w] = 0
                break

    def pc_skeleton(self):
        n = self.n
        self.skeleton = np.array([[int(x > y) for x in range(n)] for y in range(n)])
        i = 0
        while i < n:
            for q in range(n):
                for p in range(n):
                    link = self.skeleton[p, q]
                    if link == 0:
                        pass
                    else:
                        self.skeleton[p, q] = self.test_indep(p, q, i)
            i += 1
        self.skeleton = np.maximum(self.skeleton, self.skeleton.transpose())
        return self.skeleton

    def find_v_struct(self):
        middle_node, edge_nodes = self.find_forks(self.n)
        for i in range(len(middle_node)):
            if middle_node[i][0] in self.cond_sets[tuple(edge_nodes[i])]:
                self.skeleton[middle_node[i][0], edge_nodes[i][0]] = 0
                self.skeleton[middle_node[i][0], edge_nodes[i][1]] = 0
                self.skeleton[edge_nodes[i][0], middle_node[i][0]] = 2
                self.skeleton[edge_nodes[i][1], middle_node[i][0]] = 2

        return self.skeleton

    def pc_dag(self):
        self.pc_skeleton()
        self.find_v_struct()
        old_skel = None
        while not np.array_equal(old_skel, self.skeleton):
            old_skel = self.skeleton.copy()
            self.step1()
            self.step2()
            self.step3()

        for i in range(self.n):
            for j in range(i):
                if self.skeleton[i, j] == 1 and any(self.skeleton[:, i] == 2):
                    self.skeleton[i, j] = 2
                    self.skeleton[j, i] = 0

        desc_dict = dict()
        for i in range(self.n):
            desc_dict[i] = descendants(self.skeleton).all_desc(i)

        i = 0
        ancestral_order = list()
        while len(desc_dict) > 0:
            desc_round = sum([desc_dict[i] for i in desc_dict], [])
            ancestral_order += [x for x in range(self.n) if x not in desc_round + ancestral_order]
            [desc_dict.pop(i, None) for i in ancestral_order]
            i += 1

        for i in range(self.n):
            for j in range(i):
                if self.skeleton[i, j] == 1 and ancestral_order[i] > ancestral_order[j]:
                    self.skeleton[i, j] = 2
                    self.skeleton[j, i] = 0

        self.skeleton = self.skeleton / 2

        # G = nx.from_numpy_matrix(self.skeleton, create_using = nx.DiGraph())
        # nx.draw_networkx(G)
        # plt.show()

        return self.skeleton