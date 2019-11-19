import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv("cc9_rel.csv")

individuals = df['Source'].unique()
levels = sorted(df['Level'].unique())

print("Individuals: ", len(individuals))

def count_for_change(df, change):
    print("Change", change)
    
    change_rows = df[df['Change'] == change]
    print("Total for this change:", len(change_rows))

    change_groups = change_rows.groupby('Level').count()
    counts = np.zeros(len(levels), dtype=int)
    
    for level in levels:
        print("Level", level)
        if level in change_groups.index:
            counts[level] = change_groups.loc[level].values[0]
        else:
            counts[level] = 0
        print("Amt.:", counts[level])
    
    return counts

counts_dict = {
    'Estável': count_for_change(df, 'stable'),
    'Diminuiu': count_for_change(df, 'decreased'),
    'Aumentou': count_for_change(df, 'increased'),
}

counts_df = pd.DataFrame.from_dict(counts_dict)
print(counts_df)

counts_df.plot.bar(stacked=True)

plt.xlabel('Nível de amizade')
plt.ylabel('# Respostas')
plt.title('Ciência da Computação, 9º período')
plt.grid(alpha=0.5)
plt.xticks(rotation='horizontal')
plt.show()
