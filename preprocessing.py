import pandas as pd
import numpy as np
import reciprocity_distribution
import network_utils
import time

def remove_irrelevant_links(df, auxiliary_df):
  """
  Removes links TARGETING individuals who: (i) did not answer and (ii) are unknown
  to most of the sample, i. e. whose in-degree is less than the first quartile.

  """

  participants = df['Target'].unique()

  # Let's compute the in degrees of everyone
  in_degrees = []

  # Store the in degree and document ID
  for participant in participants:
    row = auxiliary_df[auxiliary_df['docid'] == participant].iloc[0]

    reverse = df[df['Target'] == participant]
    in_degree = reverse['Level'].sum()
    in_degrees.append((participant, in_degree))

  # Calculate the first quartile from the rebuilt data frame
  deg_df = pd.DataFrame.from_records(in_degrees, columns=['person', 'in_degree'])
  q1 = deg_df['in_degree'].quantile(0.25)
  q3 = deg_df['in_degree'].quantile(0.75)
  # print("Mean: ", deg_df['in_degree'].mean())
  # print("Stddev: ", deg_df['in_degree'].std())
  # print("Threshold (Q1): ", q1)
  # print("IQR: ", (q3 - q1))

  # deg_df.to_csv('__temp_degrees.csv', index=None)

  # Now let's see who should be excluded from the sample
  targets_to_remove = []

  for participant in participants:
    row = auxiliary_df[auxiliary_df['docid'] == participant].iloc[0]

    # If this person didn't answer. (As a consequence, they may only appear in the Target column)
    if row['answered'] != 'yes':
      # print(row)
      in_degree = deg_df[deg_df['person'] == participant].iloc[0]['in_degree']
      # print('In degree of this person /\:', in_degree)

      # If too few people know them, they can be removed from the sample.
      if in_degree <= q1:
        # print("Remove this one!")
        targets_to_remove.append(participant)

      print('\n')

  # Keep the ones that shouldn't be removed
  new_df = df[df['Target'].isin(targets_to_remove) == False]

  return new_df


def fill_missing_relationships(df, auxiliary_df):
  everyone = df['Target'].unique()
  distributions = reciprocity_distribution.compute_distributions(df)

  # print("Distributions!")
  # for k, v in distributions.items():
  #   print(k)

  #   print(v.keys())
  #   vals = list(map(lambda v: "%.2f" % v, v.values()))
  #   print(' & '.join(vals), '\\\\')
    

  # time.sleep(10)

  new_rows = []

  for ego in everyone:
    ego_row = auxiliary_df[auxiliary_df['docid'] == ego].iloc[0]

    if ego_row['answered'] != 'yes':
      # print("Analyzing this person:")
      print(ego_row)

      
      for alter in everyone:
        if ego != alter:
          ego_name = ego_row['docid']
          alter_name = auxiliary_df[auxiliary_df['docid'] == alter].iloc[0]['docid']

          reverse = network_utils.get_reverse(df, ego, alter)
          filled_value = None

          if reverse is None:
            print("I can't deduce the relationship between", ego, "-", ego_name, "and", alter, "-", alter_name, "!")
            raw_value = input("What should it be? ")

            if len(raw_value) > 0:
              filled_value = int(raw_value)
            else:
              print("Assuming 0.")
              filled_value = 0

          else:
            # print("The reverse level is: ", reverse['Level'])

            level_distribution = distributions[reverse['Level']]

            filled_value = np.random.choice(list(level_distribution.keys()), p=list(level_distribution.values()))
            # print("I'm guessing a value of: ", filled_value)
          
          new_rows.append({
            'Source': ego,
            'Target': alter,
            'Level': filled_value,
            'Change': 'stable' #whatever
          })
  
  deduced_df = pd.DataFrame.from_records(new_rows)
  complete_df = pd.concat([df, deduced_df], ignore_index=True)

  # print(complete_df)

  return complete_df


def fill_missing_scores(df, auxiliary_df):
  members = df['Source'].unique()

  members_df = auxiliary_df[auxiliary_df['docid'].isin(members)]
  unanswered = auxiliary_df[(auxiliary_df['docid'].isin(members)) & (auxiliary_df['answered'] != 'yes')]
  
  new_auxiliary_df = auxiliary_df.copy(deep=True)

  for key, row in unanswered.iterrows():
    # new_auxiliary_df.at[key, 'answered'] = 'yes'
    new_auxiliary_df.at[key, '2018/1'] = members_df['2018/1'].mean()
    new_auxiliary_df.at[key, '2018/2'] = members_df['2018/2'].mean()
    new_auxiliary_df.at[key, '2019/1'] = members_df['2019/1'].mean()
    new_auxiliary_df.at[key, 'average'] = members_df['average'].mean()



  return new_auxiliary_df


def remove_excluded_individuals(df, auxiliary_df):
  """
  Remove individuals who absolutely cannot be part of the sample, due to a specific
  exclusion reason found in the auxiliary dataset.

  """

  not_excluded = auxiliary_df[auxiliary_df['exclusion_reason'].isnull()]['docid']

  new_df = df[df['Source'].isin(not_excluded)]
  print("Excluded individuals removed from Source. Rows removed: ", len(df) - len(new_df))
  newer_df = new_df[new_df['Target'].isin(not_excluded)]
  print("Excluded individuals removed from Target. Rows removed: ", len(new_df) - len(newer_df))
  print("Total rows removed: ", len(df) - len(newer_df))

  return newer_df


def add_weights(df):
  '''Adds a column called Weight that is equal to the Level column.'''

  df['Weight'] = df['Level']
  return df

def remove_null_links(df):
  '''Removes entries with a friendship level of zero.'''
  
  original_individuals = len(df['Source'].unique())

  new_df = df[df['Level'] > 0]

  new_individuals = len(new_df['Source'].unique())
  print("Zeroes removed. Before, there were ", original_individuals, " individuals. Now, there are ", new_individuals)

  return new_df


def numericize_changes(df):
  '''Converts textual representations of changes to numeric values in {-1, 0, 1}.'''

  new_df = df.copy(deep=True)

  for key, row in df.iterrows():
    new_df.at[key, 'Change'] = network_utils.change_to_value(row['Change'])

  return new_df


def preprocess(rel_df, aux_df, save_to=None, remove_excluded=True, remove_irrelevant=True, fill_missing=True, remove_zeroes=True, convert_changes=True):
  np.random.seed(9999)

  if remove_excluded:
    rel_df = remove_excluded_individuals(rel_df, aux_df)
  
  if remove_irrelevant:
    rel_df = remove_irrelevant_links(rel_df, aux_df)
  
  if fill_missing:
    rel_df = fill_missing_relationships(rel_df, aux_df)
  
  if remove_zeroes:
    rel_df = remove_null_links(rel_df)

  if convert_changes:
    rel_df = numericize_changes(rel_df)


  rel_df = add_weights(rel_df)

  if save_to is not None:
    rel_df.to_csv(save_to, index=None)
    print("Results written to: ", save_to)

  return rel_df