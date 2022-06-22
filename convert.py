from cgi import print_arguments
import csv
import pandas as pd
data = pd.read_csv("labels.csv")
labels = data.label
names = data.name
LABELS_NAMES = {}
# with open("labels.csv", mode='w', encoding='utf-8', newline='') as train:
#     writer = csv.writer(train, delimiter=',')
#     writer.writerow(['label', 'name'])
#     for (label, name) in zip(labels, names):
#         newline = [label,name]
#         writer = csv.writer(train, delimiter=',')
#         writer.writerow(newline)
for (label, name) in zip(labels, names):
    LABELS_NAMES[label] = name
print(LABELS_NAMES)
