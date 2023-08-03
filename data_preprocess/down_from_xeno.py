import requests
import json
from multiprocessing.pool import ThreadPool
import os
import shutil
from tqdm import tqdm


def txt_to_list(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def download(url_filename):
    url, filename = url_filename
    r = requests.get(url)
    with open(filename, 'wb') as f:
        for ch in r:
            f.write(ch)


birds_name = txt_to_list("names.txt")
for name in tqdm(birds_name):
    name = name.replace('\n', '')
    if os.path.exists(name):
        shutil.rmtree(name)
    os.makedirs(name)
    temp = name
    name = name.replace(' ', '+')
    html = "https://xeno-canto.org/api/2/recordings?query={}".format(name)
    urls_filenames = []
    json_txt = requests.get(html).text
    flag = False
    for recording in json.loads(json_txt)["recordings"][:100]:
        filename = temp + os.sep + "xc" + recording["id"] + ".mp3"
        url = recording["file"]
        if (url == ''):
            flag = True
            print("\n_discarded species:",temp)
            break
        urls_filenames.append((url, filename))
    if (not flag):
        print("\n",temp)
        results = ThreadPool(8).imap_unordered(download, urls_filenames)
        for i in results:
            pass
