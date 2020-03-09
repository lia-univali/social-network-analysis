import pandas as pd
import time
import network_utils

def get_reverse(df, row):
  source = row['Source']
  target = row['Target']
  filtered = df[(df['Source'] == target) & (df['Target'] == source)]
  
  if len(filtered) == 0:
    return None

  return filtered.iloc[0]

def compute_distributions(df):
  distributions = dict()

  for level in df['Level'].unique():
    level_df = df[df['Level'] == level]

    # The outgoing level is *level*. what is the distribution of the incoming levels?
    incoming = {x: 0 for x in range(5 + 1)}

    for i, row in level_df.iterrows():
      reverse = network_utils.get_reverse_row(df, row)

      if reverse is not None:
        incoming[reverse['Level']] += 1
    
    total = sum(incoming.values())
    distributions[level] = {k: v / total for k, v in incoming.items()}
  
  print(distributions)
  return distributions
