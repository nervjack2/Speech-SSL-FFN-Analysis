import sys 
import os 
import glob

data_dir = sys.argv[1]
wav_pths = glob.glob(data_dir+'/**/*.flac', recursive=True)
for pth in wav_pths:
    save_pth = pth.split('.')[0]+'.wav'
    os.system(f'ffmpeg -i {pth} {save_pth}')