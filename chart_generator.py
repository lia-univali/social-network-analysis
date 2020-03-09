import pandas as pd
import numpy as np
import preprocessing
import matplotlib.pyplot as plt

def generate_indegree(scores):
  answered = scores[['docid', 'answered']]
  answered = answered.rename(columns={'docid': 'person'})

  df = pd.read_csv('__temp_degrees.csv')

  df = pd.merge(answered, df, on='person')
  df = df.sort_values(by='in_degree', inplace=False, ascending=False)
  print(df)

  heights = df['in_degree'].values
  x = np.arange(len(heights))

  x_yes = []
  x_no = []

  for i in x:
    if df.iloc[i]['answered'] == 'yes':
      x_yes.append(i)
    else:
      x_no.append(i)

  heights_yes = list(map(lambda i: heights[i], x_yes))
  heights_no = list(map(lambda i: heights[i], x_no))

  plt.bar(x_yes, heights_yes, 0.8, label="Respondeu")
  plt.bar(x_no, heights_no, 0.8, label="Absteve-se")
  plt.legend()
  tick_spacing = 3
  plt.xticks(np.arange(0, len(heights), tick_spacing), np.arange(1, len(heights) + 1, tick_spacing))
  plt.title("Participantes da pesquisa")
  plt.xlabel("Indivíduo")
  plt.ylabel("Grau de entrada ponderado")
  plt.show()
  

def generate(rel_df, scores):
  # generate_indegree(scores)
  # return 0
  scores = scores[scores['exclusion_reason'].isnull()]
  scores = scores[~(scores['2018/1'].isnull())]

  for i, (key, row) in enumerate(scores.iterrows()):
    x = ['2018/1', '2019/1']
    y = row[x].values

    plt.plot(x, y, label=row['docid'], marker='o', alpha=0.8)
    print(i + 1)

  plt.xlabel("Semestre")
  plt.ylabel("Nota média")
  plt.title("Evolução de notas ao longo de um ano")
  plt.show()
  print(scores)



def make_boxplot_part(data, column, ax, title, ylabel):
  original = data[data['transformed'] == 'no']
  x_original = [1]
  y_original = original[column].values

  nulls = data[data['transformed'] == 'yes']

  ax.boxplot(nulls[column])
  ax.set_title(title)
  ax.set_ylabel(ylabel)
  ax.set_xticklabels([])
  ax.scatter(x_original, y_original, label='Rede original', marker='X', zorder=999, s=110)
  ax.legend(loc='upper right')

def make_boxplots(filename, title):
  df = pd.read_csv(filename)

  fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(5, 4.4))
  fig.suptitle(title)

  make_boxplot_part(df, 'rsquared_adj', axes[0], '$R^2$ ajustado', '$R_a^2$')
  make_boxplot_part(df, 'f_pvalue', axes[1], 'Valor-$p$ (teste $F$)', '$p$')

  # plt.legend()
  plt.show()

def run():
  b_title = 'Primeira definição de círculos sociais: $B_k(u)$'
  b_data = 'null_model_results_blansky.csv'

  make_boxplots(b_data, b_title)

  c_title = 'Segunda definição de círculos sociais: $C_k(u)$'
  c_data = 'null_model_results_comm.csv'

  make_boxplots(c_data, c_title)

run()