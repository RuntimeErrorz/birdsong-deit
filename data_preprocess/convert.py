import os
import magic
from tqdm import tqdm
from pydub import AudioSegment


for dir in tqdm(os.listdir("File")):
    for file in tqdm(os.listdir("File/" + dir)):
        if 'XC' not in file:
            os.remove("File/" + dir + "/" + file)
            continue
        list = file.split('.')[0].split('-')
        if magic.from_file("File/" + dir + "/" + file).split(',')[0].strip() == 'RIFF (little-endian) data':
            os.rename("File/" + dir + "/" + file,
                      "File/" + dir + "/" + list[0]+".wav")
            continue
        newName = "File/" + dir + "/" + list[0] + ".mp3"
        if (len(list) > 1):
            os.rename("File/" + dir + "/" + file, newName)
        sound = AudioSegment.from_mp3(newName)
        os.remove(newName)
        sound.export("File/" + dir + "/" + list[0]+".wav", format="wav")
