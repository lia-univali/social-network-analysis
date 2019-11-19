import network_utils
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model, metrics
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time

def to_ranking(rel_df, auxiliary_df, column, ignore_rank=False):
  students = rel_df['Source'].unique()

  ranking_dict = dict()
  filtered_df = auxiliary_df[auxiliary_df['docid'].isin(students)]
  sorted_df = filtered_df.sort_values(by=column, ascending=False)
  # print(sorted_df)

  for rank, (key, row) in enumerate(sorted_df.iterrows()):
    student = row['docid']
    # print(student)
    # print(network_utils.select_neighbors(student, 1, rel_df))

    if ignore_rank == False:
      ranking_dict[student] = (len(students) - rank) / len(students)
    else:
      ranking_dict[student] = row[column]
    # print(student, row['nome'], ranking_dict[student])

  return ranking_dict

def make_centrality_dict(auxiliary_df, metric):
  centrality_dict = dict()

  for key, row in auxiliary_df.iterrows():
    student = row['docid']
    centrality_dict[student] = row[metric]

  return centrality_dict


def characterize_social_circle(node, level, ranking, rel_df, auxiliary_df, centrality_metric=None):
  neighbors = network_utils.select_neighbors(node, level, rel_df)
  # neighbors = network_utils.select_community_members(node, level, rel_df, auxiliary_df)

  mapper = None

  if centrality_metric is not None:
    centrality_dict = make_centrality_dict(auxiliary_df, centrality_metric)

    mapper = lambda neighbor: ranking[neighbor] * (centrality_dict[neighbor] - centrality_dict[node])
  else:
    mapper = lambda neighbor: ranking[neighbor]

  
  neighbor_rankings = list(map(mapper, neighbors))
  average_rankings = 0

  if len(neighbors) > 0:
    average_rankings = sum(neighbor_rankings) / len(neighbor_rankings)

  return average_rankings - ranking[node]


def flatten_array(arr):
  return list(map(lambda x: x[0], arr))


def execute_regression(X, y):
  model = linear_model.LinearRegression().fit(X, y)

  y_true = y.values
  y_pred = model.predict(X)

  # print(y_true)
  # print(y_pred)

  print("Coefficients")
  print(model.coef_)
  print("Intercept")
  print(model.intercept_)
  print("R^2")
  print(model.score(X, y))
  print("Pearson's R")
  print(stats.pearsonr(y_true, y_pred))

  return model, y_true, y_pred


def test_multicollinearity(df, x_features):
  print('\n=============== MULTICOLLINEARITY TEST ===============\n')

  length = len(x_features)
  mat = np.zeros((length, length))

  for i, feature1 in enumerate(x_features):
    for j, feature2 in enumerate(x_features):
      correlation = stats.pearsonr(df[feature1], df[feature2])

      mat[i, j] = correlation[0]

  print(mat)



def multiple_regression(df, x_features = ['g', 'x1', 'x2', 'x3', 'x4', 'x5']):
  print('\n=============== MULTIPLE REGRESSION ===============\n')

  y_feature = 'evolution'

  # test_multicollinearity(df, x_features)

  X = df[x_features]
  y = df[y_feature]

  model, y_true, y_pred = execute_regression(X, y)

  X2 = sm.add_constant(X)
  model2 = sm.OLS(y, X2)
  result = model2.fit()
  print(result.summary())

  variables = X
  vif = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
  print("VIF:")
  print(variables)
  print(vif)

  return result


def univariate_regression(df):
  print('\n=============== UNIVARIATE REGRESSION ===============\n')

  x_features = ['g', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5']

  for x_feature in x_features:
    y_feature = 'evolution'

    X = df[x_feature].values.reshape(-1, 1)
    y = df[y_feature]

    model, y_true, y_pred = execute_regression(X, y)

    X_line = np.linspace(min(flatten_array(X)), max(flatten_array(X)), 100).reshape(-1, 1)

    y_limit = max(abs(y.values)) * 1.2

    print(x_feature)
    df.plot.scatter(x=x_feature, y=y_feature)
    plt.plot(X_line, model.predict(X_line), '-', color='red')
    plt.ylim((-y_limit, y_limit))
    plt.show()

    X2 = sm.add_constant(X)
    model2 = sm.OLS(y, X2)
    result = model2.fit()
    print(result.summary())



def brute_force_regression(df):    
  s = ['g', 'x1', 'x2', 'x3', 'x4', 'x5']
  c = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))

  for d in c:
    x_features = list(d)

    one_hot = list(map(lambda feature: feature in x_features, s))
    print(one_hot)

    result = multiple_regression(df, x_features)
    print("Testing with: ", x_features, " -> F-test p-value:", result.f_pvalue, "rsquared:", result.rsquared, "adj rsquared:", result.rsquared_adj)



def analyze(rel_df, auxiliary_df):
  students = rel_df['Source'].unique()

  ignore_rank = False

  ranking_prev = to_ranking(rel_df, auxiliary_df, '2018/1', ignore_rank=ignore_rank)
  ranking_next = to_ranking(rel_df, auxiliary_df, '2019/1', ignore_rank=ignore_rank)

  # print(auxiliary_df)
  # time.sleep(10)

  records = []
  centrality_metric = None
  # centrality_metric = 'out_degree'
  # centrality_metric = 'in_degree'
  # centrality_metric = 'undirected_degree'
  # centrality_metric = 'out_eigenvector'
  # centrality_metric = 'in_eigenvector'
  # centrality_metric = 'undirected_eigenvector'
  # centrality_metric = 'out_closeness'
  # centrality_metric = 'in_closeness'
  # centrality_metric = 'undirected_closeness'
  # centrality_metric = 'out_betweenness'
  # centrality_metric = 'in_betweenness'
  # centrality_metric = 'undirected_betweenness'
  # centrality_metric = 'out_katz'
  # centrality_metric = 'in_katz'
  # centrality_metric = 'undirected_katz'

  for student in students:
    records.append({
      'student': student,
      'g': ranking_prev[student],
      'x0': characterize_social_circle(student, 0, ranking_prev, rel_df, auxiliary_df, centrality_metric),
      'x1': characterize_social_circle(student, 1, ranking_prev, rel_df, auxiliary_df, centrality_metric),
      'x2': characterize_social_circle(student, 2, ranking_prev, rel_df, auxiliary_df, centrality_metric),
      'x3': characterize_social_circle(student, 3, ranking_prev, rel_df, auxiliary_df, centrality_metric),
      'x4': characterize_social_circle(student, 4, ranking_prev, rel_df, auxiliary_df, centrality_metric),
      'x5': characterize_social_circle(student, 5, ranking_prev, rel_df, auxiliary_df, centrality_metric),
      'evolution': ranking_next[student] - ranking_prev[student]
    })

  df = pd.DataFrame.from_records(records)
  # print(df)

  # brute_force_regression(df)

  # univariate_regression(df)
  
  
  return multiple_regression(df)

  # x_features = ['g', 'x1', 'x2', 'x3', 'x4', 'x5']

  # for x_feature in x_features:
  #   y_feature = 'evolution'

  #   X = df[x_feature].values.reshape(-1, 1)
  #   y = df[y_feature]

  #   model = linear_model.LinearRegression().fit(X, y)
  #   print("Coefficients")
  #   print(model.coef_)
  #   print("Intercept")
  #   print(model.intercept_)
  #   print("R^2")
  #   print(model.score(X, y))

  #   y_true = df[y_feature].values
  #   y_pred = model.predict(X)

  #   print(y_true)
  #   print(y_pred)

  #   # print(comparison)
  #   # comparison_df = pd.DataFrame.from_dict(comparison)
  #   # print(comparison_df)

  #   print("Pearson's R")
  #   print(stats.pearsonr(y_true, y_pred))

  #   X_line = np.linspace(min(flatten_array(X)), max(flatten_array(X)), 100).reshape(-1, 1)

  #   y_limit = max(abs(y.values)) * 1.2

  #   df.plot.scatter(x=x_feature, y=y_feature)
  #   plt.plot(X_line, model.predict(X_line), '-', color='red')
  #   plt.ylim((-y_limit, y_limit))
  #   plt.show()

  #   X2 = sm.add_constant(X)
  #   model2 = sm.OLS(y, X2)
  #   result = model2.fit()
  #   print(result.summary())
