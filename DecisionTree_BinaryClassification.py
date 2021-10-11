import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class My_Binary_Decision_Tree_Classifier:

  def __init__(self, df, positive_label_name):

    self.df = df
    self.positive_label_name = positive_label_name

  def _cross_entropy_(self, probs):

    if 0 in probs: return 0

    result = np.sum(probs * np.log2(probs))

    return -result

  def _totall_entropy_(self, data):

    plus = len(data[data[data.columns[-1]] == self.positive_label_name]) / len(data)
    probs = np.array([plus, 1 - plus])

    return self._cross_entropy_(probs)

  def _calculate_gained_info_(self, total_entropy, data, attr):

    unique_values = np.unique(data[attr])
    weights = []
    entropies = []

    for val in unique_values:

      weight = len(data[data[attr] == val]) / len(data)

      plus_por = len(data[(data[attr] == val) & (data[data.columns[-1]] == self.positive_label_name)]) / len(data[data[attr] == val])
      minus_por = 1 - plus_por

      weights.append(weight)
      entropies.append(self._cross_entropy_(np.array([plus_por, minus_por])))

    weights, entropies = np.array(weights), np.array(entropies)

    E = np.sum(weights * entropies)

    return total_entropy - E

  def _find_the_best_attribute_(self, data):

    gained_info = {}

    TE = self._totall_entropy_(data)

    for att in data.columns[:-1]:

      info = self._calculate_gained_info_(TE, data, att)

      gained_info[att] = float("%.5f"%info)

    aux = {v:k for (k, v) in gained_info.items()}

    selected_att = aux[max(aux.keys())]

    print(f"Total Entropy: {TE}")
    print(f"Gained info -> {gained_info}")
    print(f"{selected_att} has been selected for this expansion")
    print('___________________________________________________')

    return selected_att

  def _stop_expanding_(self, data, att):

    unique_values = np.unique(data[att])

    stop = []

    for val in unique_values:

      M = len(data[(data[att] == val) & (data[data.columns[-1]] == self.positive_label_name)])

      if not M or M == len(data[data[att] == val]):

        stop.append(True)

      else:

        stop.append(False)

    return stop

  def _grow_branch_(self, previous_best_att, expanding):

    un = np.unique(self.df[previous_best_att])
    nodes_after_branches = {}

    for br in range(len(expanding)):

      if not expanding[br]:

        print(f"{un[br]}\n")

        df_ = self.df[self.df[previous_best_att] == un[br]]

        best_att = self._find_the_best_attribute_(df_)

        nodes_after_branches[un[br]] = best_att

      else:

        print(f"{un[br]}\n\nIt does not need to be expanded")
        print('___________________________________________________')
        nodes_after_branches[un[br]] = 'leaf node'

    return nodes_after_branches

  def fit(self):

    print('Start\n')

    Root = self._find_the_best_attribute_(self.df)

    EXP = self._stop_expanding_(self.df, Root)

    BRNCHS = self._grow_branch_(Root, EXP)

    print(f"Trained Decision Tree\n\n{(Root, BRNCHS)}\n")


df = pd.read_csv('Tennis.csv')
df.index = df['Day']
df.drop(['Day'], axis=1, inplace=True)

DTBC = My_Binary_Decision_Tree_Classifier(df, 'Yes')
DTBC.fit()

plt.imshow(mpimg.imread('dt.png'))
plt.show()
