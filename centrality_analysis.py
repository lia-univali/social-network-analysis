import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import centrality

def weighted_degree_centrality(G):
  centrality_dict = dict()

  for node, neighbors in G.adjacency():
    print("Neighbors of", node)
    
    value = 0
    for neighbor, attributes in neighbors.items():
      value += attributes['weight']
    
    centrality_dict[node] = value
    print(value)

  # for k, v in centrality_dict.items():
  #   print(k, v)

  return centrality_dict


# def weighted_indegree_centrality(G):
  # centrality_dict = dict()

  # for node in G.nodes():
  #   print("Node: ", node)

  #   value = 0
  #   for source, target, data in G.in_edges(node, data=True):
  #     value += data['weight']
  #     # print(data)

  #   print(value)
  #   centrality_dict[node] = value

  # return weighted_degree_centrality(G.reverse())


def analyze(directed_df, undirected_df, auxiliary_df):
  directed_df = directed_df.copy(deep=True)
  undirected_df = undirected_df.copy(deep=True)

  directed_df = directed_df.rename(mapper=lambda name: name.lower(), axis='columns')
  undirected_df = undirected_df.rename(mapper=lambda name: name.lower(), axis='columns')

  G = nx.from_pandas_edgelist(directed_df, edge_attr=['weight', 'change'], create_using=nx.DiGraph)
  G_undirected = nx.from_pandas_edgelist(undirected_df, edge_attr=['weight', 'change'])

  alpha_coef = 0.9

  alpha = alpha_coef / max(nx.adjacency_spectrum(G).real)
  alpha_undirected = alpha_coef / max(nx.adjacency_spectrum(G_undirected).real)

  centralities = {
    'out_degree': weighted_degree_centrality(G),
    'in_degree': weighted_degree_centrality(G.reverse()),
    'undirected_degree': weighted_degree_centrality(G_undirected),
    
    'out_eigenvector': centrality.eigenvector_centrality(G, weight='weight'),
    'in_eigenvector': centrality.eigenvector_centrality(G.reverse(), weight='weight'),
    'undirected_eigenvector': centrality.eigenvector_centrality(G_undirected, weight='weight'),
    
    'out_closeness': centrality.closeness_centrality(G, distance='weight'),
    'in_closeness': centrality.closeness_centrality(G.reverse(), distance='weight'),
    'undirected_closeness': centrality.closeness_centrality(G_undirected, distance='weight'),

    'out_betweenness': centrality.betweenness_centrality(G, weight='weight'),
    'in_betweenness': centrality.betweenness_centrality(G.reverse(), weight='weight'),
    'undirected_betweenness': centrality.betweenness_centrality(G_undirected, weight='weight'),

    'out_katz': centrality.katz_centrality(G, alpha=alpha, weight='weight'),
    'in_katz': centrality.katz_centrality(G.reverse(), alpha=alpha, weight='weight'),
    'undirected_katz': centrality.katz_centrality(G_undirected, alpha=alpha, weight='weight')
  }

  for centrality_type in centralities.keys():
    directed_df[centrality_type] = np.NaN

  augmented_auxiliary_df = auxiliary_df.copy(deep=True)

  for key, row in augmented_auxiliary_df.iterrows():
    node = row['docid']
    for centrality_type, values in centralities.items():
      if node in values:
        augmented_auxiliary_df.at[key, centrality_type] = values[node]
  
  print(augmented_auxiliary_df)
  return augmented_auxiliary_df