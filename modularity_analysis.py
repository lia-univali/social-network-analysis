import pandas as pd
import numpy as np
import preprocessing
import community_detection
import network_utils
import modularity_analysis
import networkx as nx

def export_result(G, communities, file_name=None):
  nodes = {
    'Id': list(G.nodes()),
    'Label': list(G.nodes()),
    'ModularityClass': communities
  }

  nodes_df = pd.DataFrame.from_dict(nodes)
  
  if file_name is not None:
    nodes_df.to_csv(file_name)

  return nodes_df


def analyze_threshold(rel_df, threshold):
  print(threshold)

  nodes = rel_df['Source'].unique()
  print(len(nodes))

  filtered_df = rel_df.copy(deep=True)
  filtered_df = rel_df[rel_df['Level'] >= threshold]
  result = community_detection.detect(filtered_df, resolution=0)

  final_graph = result['graph']
  final_communities = list(result['communities'])

  highest = np.max(final_communities)

  for node in nodes:
    if final_graph.has_node(node) == False:
      final_graph.add_node(node)

      highest += 1
      final_communities.append(highest)
      # print("Node", node, "added with community", highest)


  return export_result(final_graph, final_communities)


def analyze_scale(rel_df, exponent):
  scaled_df = rel_df.copy(deep=True)
  scaled_levels = scaled_df['Level'].map(lambda level: level ** exponent)

  scaled_df['Level'] = scaled_levels
  scaled_df['Weight'] = scaled_levels

  # result = community_detection.detect(scaled_df, resolution=0)
  result = community_detection.detect(scaled_df)
    
  final_graph = result['graph']
  final_communities = result['communities']

  return export_result(final_graph, final_communities)

  
