import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv("cc9_rel_processed.csv")

individuals = df['Source'].unique()
print("Individuals: ", len(individuals))

tuples = []

def get_level(row):
    return row['Level'].values[0]

for individual in individuals:
    for friend in individuals:
        if individual != friend:
            # outgoing: from "individual" to "friend"
            # incoming: from "friend" to "individual"
            
            outgoing_row = df[(df['Source'] == individual) & (df['Target'] == friend)]
            incoming_row = df[(df['Target'] == individual) & (df['Source'] == friend)]
            name = '%s-%s' % (individual, friend)
            
            outgoing = get_level(outgoing_row)
            incoming = get_level(incoming_row)
            
            #print(individual, friend)
            #print(outgoing_row)
            #print(incoming_row)
            
            tuples.append((name, outgoing, incoming))
            
            
reciprocity = pd.DataFrame.from_records(tuples, columns=['pair', 'outgoing', 'incoming'])
print(reciprocity)

#reciprocity.plot(kind='scatter', x='outgoing', y='incoming')
#plt.show()
pivot = reciprocity.pivot_table(index=['outgoing', 'incoming'], aggfunc='size')
#print(pivot)

tuples_with_size = []
for outgoing, df_l0 in pivot.groupby(level=0):
    for incoming, df_l1 in df_l0.groupby(level=1):
        size = df_l1.values[0]
        tuples_with_size.append((outgoing, incoming, size))

reciprocity_with_size = pd.DataFrame.from_records(tuples_with_size, columns=['outgoing', 'incoming', 'size'])

print(reciprocity_with_size)

reciprocity_with_size.plot(kind='scatter', x='outgoing', y='incoming', s=reciprocity_with_size['size'] * 30)
plt.xlabel('Ego para alter')
plt.ylabel('Alter para ego')
plt.title('Reciprocidade')
plt.margins(0.15)
plt.grid(alpha=0.5)
plt.show()
