from pydub import AudioSegment
song = AudioSegment.from_ogg("a.ogg")
start_time = "0:00"
stop_time = "1:30"
start_time = (int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])) * 1000
stop_time = (int(stop_time.split(':')[0]) * 60 + int(stop_time.split(':')[1])) * 1000
song[start_time:stop_time].export('a_slice.ogg',format="ogg") 