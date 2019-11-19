import pandas as pd
import numpy as np
from scipy import stats
import time

def describe_variance_change(before, after, pvalue, critical_p = 0.05):
  increased = after > before

  if pvalue > critical_p:
    return 'equal'
  else:
    if increased:
      return 'increased'
    else:
      return 'decreased'

def analyze_community(members, others, identifier):
  scores_18_1 = members['2018/1']
  scores_18_2 = members['2018/2']
  scores_19_1 = members['2019/1']
  scores_avg = members['average']


  others_f = others[['2018/1', 'ModularityClass']].sort_values(by='ModularityClass')
  others_variances = others_f.groupby('ModularityClass').var()
  variances = []

  for community, variance in others_variances.iterrows():
    variances.append(float(variance))

  others_mean_var = others['2018/1'].var()

  critical_p = 0.05
  levene_mean = stats.levene(scores_18_1, scores_19_1, center='mean')
  levene_median = stats.levene(scores_18_1, scores_19_1, center='median')
  bartlett = stats.bartlett(scores_18_1, scores_19_1)
  
  to_change = lambda pvalue: describe_variance_change(scores_18_1.var(), scores_19_1.var(), pvalue, critical_p) 

  return {
    'ModularityClass': identifier,
    'Members': len(members),
    # 'Mean18/1': scores_18_1.mean(),
    # 'Mean18/2': scores_18_2.mean(),
    # 'Mean19/1': scores_19_1.mean(),
    # 'MeanAvg': scores_avg.mean(),
    'Variance18/1': scores_18_1.var(),
    # 'Variance18/2': scores_18_2.var(),
    'Variance19/1': scores_19_1.var(),
    # 'VarianceAvg': scores_avg.var(),
    'LeveneMeanP': levene_mean.pvalue,
    'LeveneMean': to_change(levene_mean.pvalue),
    # 'LeveneMedianP': levene_median.pvalue,
    # 'LeveneMedian': to_change(levene_median.pvalue),
    # 'BartlettP': bartlett.pvalue,
    # 'Bartlett': to_change(bartlett.pvalue),
    'DeltaMeanVariance18/1': scores_18_1.var() - others_mean_var
  }

def analyze(aux_df):
  print("\n\n==== VARIANCE ANALYSIS ====\n")

  # everyone = analyze_community(aux_df, 'everyone')

  records = []

  for mc in aux_df['ModularityClass'].unique():
    members = aux_df[aux_df['ModularityClass'] == mc]
    others = aux_df[aux_df['ModularityClass'] != mc]
    print(members)

    records.append(analyze_community(members, others, mc))
  

  # records.append(everyone)
  result = pd.DataFrame.from_records(records)
  print(result.sort_values(by='Members'))

