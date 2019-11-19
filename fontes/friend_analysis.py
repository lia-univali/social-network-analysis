import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("caelondia2.csv")

def in_score(person):
    fdf = df.loc[df['Target'] == person]
    # print("How people view ", person)
    
    levels = fdf['Weight']
    return levels.sum()
        
def out_score(person):
    fdf = df.loc[df['Source'] == person]
    # print("How", person, "views people")
    
    levels = fdf['Weight']
    return levels.sum()


person_cols = ['person', 'in_score', 'out_score', 'out_minus_in', 'sum']
person_rows = []

for k in df['Source'].unique():
    s_in = in_score(k)
    s_out = out_score(k)
    person_rows.append((k, s_in, s_out, s_out - s_in, s_in + s_out))

pdf = pd.DataFrame.from_records(person_rows, columns=person_cols)
pdf = pdf.sort_values(by='sum')

print(pdf)


bar_width = 0.27
index = np.arange(len(person_rows))

plt.bar(index, pdf['in_score'], bar_width, label='Pessoas gostam dele(a)')
plt.bar(index + bar_width, pdf['out_score'], bar_width, label='Gosta das pessoas')

plt.legend()
plt.xlabel("Pessoa")
plt.ylabel("Nota")
plt.xticks(index + bar_width / 2, index + 1)

plt.show()
