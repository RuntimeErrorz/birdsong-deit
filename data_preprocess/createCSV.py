import os
import shutil
import csv
from tqdm import tqdm

with open("train.csv", mode='w', encoding='utf-8', newline='') as train:
    writer = csv.writer(train, delimiter=',')
    writer.writerow(["common_name", "filename"])
    for dir in tqdm(os.listdir("File")):
        for file in os.listdir("File/" + dir):
            if 'XC' not in file:
                continue
            list = file.split('-')
            newline = []
            if (len(list) > 1):
                os.rename("File/" + dir + "/" + file,
                          "File/" + dir + "/" + list[0]+".mp3")
                file = list[0] + ".mp3"
                newline = [dir, file]
            else:
                newline = [dir, file]
            writer = csv.writer(train, delimiter=',')
            writer.writerow(newline)
            shutil.copy("File/" + dir + "/" + file, "train/" + file)
