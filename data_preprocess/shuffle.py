import os
import random
import shutil
import csv
import pandas as pd

train = pd.read_csv("train.csv")
train_files = list(train.filename)

files = os.listdir("train")
with open("test.csv", mode='w', encoding='utf-8', newline='') as test:
    writer = csv.writer(test, delimiter=',')
    writer.writerow(["common_name", "filename"])
    testList = random.sample(range(0, len(files)), int(len(files) / 5))
    for i in testList:
        filename = files[i]
        index = train_files.index(filename)
        newline = [train.common_name[index], train.filename[index]]
        writer = csv.writer(test, delimiter=',')
        writer.writerow(newline)
        shutil.copy("train/" + filename, "test/" + filename)
