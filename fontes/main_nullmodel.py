import pandas as pd
import numpy as np
import preprocessing
import community_detection
import network_utils
import modularity_analysis
import variance_analysis
import centrality_analysis
import blansky_analysis
import chart_generator
import time

output_file = 'cc9_rel_deduced_directed.csv'
output_file_undirected = 'cc9_rel_deduced_undirected.csv'

def convert():
  rel_df = pd.read_csv("cc9_rel.csv")
  aux_df = pd.read_csv("cc9_auxiliary.csv")

  rel_df = preprocessing.preprocess(rel_df, aux_df)
  # rel_df = network_utils.to_undirected(rel_df)
  # rel_df = preprocessing.remove_null_links(rel_df)

  rel_df = rel_df.sort_values(['Source', 'Target'])

  rel_df.to_csv(output_file, index=None)


def analyze():
  original_rel_df = pd.read_csv(output_file)
  aux_df = pd.read_csv("cc9_auxiliary.csv")

  aux_df = preprocessing.fill_missing_scores(original_rel_df, aux_df)
  
  null_results = []

  for z in range(500):
    rel_df = original_rel_df.copy(deep=True)

    if z > 0:
      # apply the null model transformation
      rel_df = network_utils.apply_null_transformation(rel_df)

    undirected_df = network_utils.to_undirected(rel_df, save_to=None)

    mode = 'scale'
    write = False

    # for i in range(10 + 1):
    for i in range(7, 8):
      threshold = i / 2.0
      exponent = i

      modularity_result = None
      if mode == 'threshold':
        modularity_result = modularity_analysis.analyze_threshold(undirected_df, threshold)
      else:
        modularity_result = modularity_analysis.analyze_scale(undirected_df, exponent)

      # Prepare things for the merge
      modularity_result = modularity_result.rename(columns={'Label': 'docid'})

      # Merge the auxiliary DataFrame with the detected communities
      merged = pd.merge(aux_df, modularity_result, on='docid')
      print(merged)

      if write == True:
        if mode == 'threshold':
          merged.to_csv("cc9_auxiliary_merged_threshold_%.2f.csv" % threshold, index=None)
        else:
          merged.to_csv("cc9_auxiliary_merged_w^%d.csv" % exponent, index=None)

      # time.sleep(10)
      # variance_analysis.analyze(merged)
      # centrality_df = centrality_analysis.analyze(rel_df, undirected_df, merged)

      result = blansky_analysis.analyze(rel_df, merged)
      # print(dir(result))

      null_results.append({
        'f_pvalue': result.f_pvalue, 
        'rsquared': result.rsquared,
        'rsquared_adj': result.rsquared_adj,
        'transformed': 'yes' if z > 0 else 'no',
      })
      
    print(z)
    time.sleep(0.2)

  null_results_df = pd.DataFrame.from_records(null_results)
  print(null_results_df)
  null_results_df.to_csv('null_model_results_comm.csv', index=None)
  print("Exponent: ", i)


# convert()
analyze()