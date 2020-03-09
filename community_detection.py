import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modularitydensity import metrics
from modularitydensity.fine_tuned_modularity import fine_tuned_clustering_q
from modularitydensity.fine_tuned_modularity_density import fine_tuned_clustering_qds

def detect(rel_df, resolution=0.0):
  df = rel_df.copy(deep=True)
  df = df.rename(mapper=lambda name: name.lower(), axis='columns')
  
  G_original = nx.from_pandas_edgelist(df, edge_attr=['weight', 'change'])
  G = nx.relabel.convert_node_labels_to_integers(G_original)

  # Ensure everyone is present, even if no edge points towards them
  # if nodes is not None:
  #   for node in nodes:
  #     if G.has_node(node) == False:
  #       G.add_node(node)

  adj = nx.to_scipy_sparse_matrix(G)

  density = False
  c = None

  if density:
    c = fine_tuned_clustering_qds(G)
  else :
    c = fine_tuned_clustering_q(G, r=resolution)

  Q = metrics.modularity_r(adj, c, np.unique(c), r=resolution)
  D = metrics.modularity_density(adj, c, np.unique(c))

  print("Communities:", c)
  print("#comms: ", len(np.unique(c)), 'given by', np.unique(c))
  print("Modularity Q:", Q)
  print("Modularity D:", D)
  print("Mode: ", 'D' if density else 'Q')

  
  for i, node in enumerate(G_original.nodes()):
    G_original.add_node(node, modularity_class=str(c[i]))

  # print(G_original.nodes())

  return {
    'graph': G_original,
    'communities': c,
    'Q': Q,
    'D': D,
  }

  