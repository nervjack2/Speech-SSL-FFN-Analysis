import os 
import sys
import glob
import json
import textgrid
import numpy as np 
from tqdm import tqdm

def parse_num(k):
    return ''.join([x for x in k if not x.isdigit()])

fl_path, pl_path, out_file, frame_period, merge_or_not = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
frame_period = int(frame_period)
merge_or_not = int(merge_or_not)

file_list = glob.glob(os.path.join(fl_path, "**/*.TextGrid"), recursive=True)
# Load phone label pair
with open(pl_path, 'r') as fp:
    phone_label_dict = json.load(fp)
phone_label_dict = {key: value[0] for key, value in phone_label_dict.items()}

record = {}
for file_path in tqdm(file_list):
    key = file_path.split('/')[-1].split('.')[0]
    tg = textgrid.TextGrid.fromFile(file_path)
    phone_align = tg[1]
    minTime, maxTime = phone_align.minTime, phone_align.maxTime
    # Calculate the number of frame when frame period is 10ms
    n_frame = int(maxTime * 1000 / frame_period)
    info = []
    for pa in phone_align:
        start, end, phone = pa.minTime*1000, pa.maxTime*1000, pa.mark
        info.append((start, end, phone))
  
    pre_phone_idx = 0
    frame_phone_align = []
    for i in range(n_frame):
        frame_start, frame_end = i*frame_period, (i+1)*frame_period
        best_idx, longest_overlap = -1, 0
        for idx in range(pre_phone_idx, len(info)):
            if frame_end > info[idx][0] and frame_start < info[idx][1]:
                overlap = min(frame_end, info[idx][1]) - max(frame_start, info[idx][0])
                if overlap > longest_overlap:
                    longest_overlap = overlap
                    best_idx = idx 
        pre_phone_idx = best_idx
        phone_name = parse_num(info[best_idx][2]) if merge_or_not else info[best_idx][2]
        frame_phone_align.append(phone_label_dict[phone_name])
    record[key] = frame_phone_align


with open(out_file, 'w') as fp:
    json.dump(record, fp)

        