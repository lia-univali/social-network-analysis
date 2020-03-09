import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modularitydensity import metrics
from modularitydensity.fine_tuned_modularity import fine_tuned_clustering_q
from modularitydensity.fine_tuned_modularity_density import fine_tuned_clustering_qds

def mapname(name):
  print(name)
  return name.lower()

df = pd.read_csv('cc9_rel_undirected_nozeroes.csv')
df = df.rename(mapper=mapname, axis='columns')
print(df)
G = nx.from_pandas_edgelist(df, edge_attr=['weight', 'change'])
# G = nx.les_miserables_graph()
G = nx.relabel.convert_node_labels_to_integers(G)

print(G)

adj = nx.to_scipy_sparse_matrix(G)

for gr in nx.connected_component_subgraphs(G):
  # Nodes of the subgraph 'gr'
  nodes_gr = list(gr)
  print(nodes_gr)


c = fine_tuned_clustering_q(G)
print(c)
Q = metrics.modularity_r(adj, c, np.unique(c), r=0)
D = metrics.modularity_density(adj, c, np.unique(c))

print("Modularity Q:", Q)
print("Modularity D:", D)
