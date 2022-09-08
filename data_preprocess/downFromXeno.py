import requests
import json
from multiprocessing.pool import ThreadPool
import os
import shutil
from tqdm import tqdm


def txtToList(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def download(urlFilename):
    url, filename = urlFilename
    r = requests.get(url)
    with open(filename, 'wb') as f:
        for ch in r:
            f.write(ch)


birdsName = txtToList("names.txt")
for name in tqdm(birdsName):
    name = name.replace('\n', '')
    if os.path.exists(name):
        shutil.rmtree(name)
    os.makedirs(name)
    temp = name
    name = name.replace(' ', '+')
    html = "https://xeno-canto.org/api/2/recordings?query={}".format(name)
    urlsFilenames = []
    json_txt = requests.get(html).text
    flag = False
    for recording in json.loads(json_txt)["recordings"][:100]:
        filename = temp + os.sep + "XC" + recording["id"] + ".mp3"
        url = recording["file"]
        if (url == ''):
            flag = True
            print("\nDiscarded species:",temp)
            break
        urlsFilenames.append((url, filename))
    if (not flag):
        print("\n",temp)
        results = ThreadPool(8).imap_unordered(download, urlsFilenames)
        for i in results:
            pass
