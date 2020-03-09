import pandas as pd
import numpy as np
import copy

def select_neighbors(node, level, rel_df):
  # Outgoing links only!
  links = rel_df[(rel_df['Source'] == node) & (rel_df['Level'] == level)]
  return links['Target'].values


def select_community_members(node, level, rel_df, aux_df):
  everyone_but_node = rel_df[rel_df['Source'] != node]['Source'].unique()

  community = aux_df[aux_df['docid'] == node].iloc[0]['ModularityClass']
  members = aux_df[(aux_df['docid'].isin(everyone_but_node)) & (aux_df['ModularityClass'] == community)]

  member_identifiers = members['docid']

  filtered_rel_df = rel_df[(rel_df['Source'] == node) & (rel_df['Target'].isin(member_identifiers))]

  # print("Members of the community:")
  # print(members)
  # print("Relationships")
  # print(filtered_rel_df)
  # print("Are they the same thing? ", len(members) == len(filtered_rel_df))

  if level > 0:
    return select_neighbors(node, level, filtered_rel_df)
  else:
    # People who belong to the community but whom node does not know.

    # remove from member_identifiers the students that appear in filtered_rel_df.
    unknown_members = []
    for member in member_identifiers:
      if member not in filtered_rel_df['Target'].values:
        unknown_members.append(member)

    return unknown_members




def change_to_value(change):
  if change == 'increased':
      return 1
  elif change == 'decreased':
      return -1
  else:
      return 0

def get_reverse_row(df, row):
  source = row['Source']
  target = row['Target']
  
  return get_reverse(df, source, target)


def get_reverse(df, source, target):
  filtered = df[(df['Source'] == target) & (df['Target'] == source)]
  
  if len(filtered) == 0:
    return None

  return filtered.iloc[0]


def apply_null_transformation(df):
  # we're assuming this is a "ready-to-use" network

  students = df['Source'].unique()

  weight_distribution = dict(df['Weight'].value_counts())
  total_weight = sum(weight_distribution.values())
  weight_probabilities = [v / total_weight for v in weight_distribution.values()]
  
  null_model = df.copy(deep=True)

  for student in students:
    filtered_df = df[df['Source'] == student]
    filtered_students = students[students != student]

    replacements = np.random.choice(filtered_students, len(filtered_df), replace=False)

    # print(student)
    for i, (key, row) in enumerate(filtered_df.iterrows()):
      # print(i, key)
      # print(filtered_students)
      null_model.at[key, 'Target'] = replacements[i]

      new_weight = np.random.choice(list(weight_distribution.keys()), 1, p=weight_probabilities)
      null_model.at[key, 'Level'] = new_weight[0]
      null_model.at[key, 'Weight'] = new_weight[0]


  print(null_model)
  return null_model

def to_undirected(df, save_to=None):
  records = []

  for i, row in df.iterrows():
    reverse = get_reverse_row(df, row)
    has_reverse = True
    
    if reverse is None:
      # print("This person:")
      # print(row)
      # print("has no reverse link. Assuming zero.")

      has_reverse = False
      reverse = {
        'Level': 0,
        'Change': 0,
      }
    
    average_level = (row['Level'] + reverse['Level']) / 2.0
    average_change = (row['Change'] + reverse['Change']) / 2.0

    row_dict = row.to_dict()
    row_dict['Level'] = average_level
    row_dict['Weight'] = average_level
    row_dict['Change'] = average_change
    records.append(row_dict)

    if has_reverse == False:
      reverse_row_dict = copy.deepcopy(row_dict)
      reverse_row_dict['Source'] = row_dict['Target']
      reverse_row_dict['Target'] = row_dict['Source']

      # print("Adding reverse:")
      # print(reverse_row_dict)
      records.append(reverse_row_dict)



  undirected_df = pd.DataFrame.from_records(records)
  # print("Undirected")
  # print(undirected_df)

  if save_to is not None:
    undirected_df.to_csv(save_to, index=None)
    print("Results written to: ", save_to)

  return undirected_df