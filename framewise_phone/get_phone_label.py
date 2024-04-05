import os 
import sys
import glob
import json
import textgrid
from tqdm import tqdm 

fl_path, out_file = sys.argv[1], sys.argv[2]
file_list = glob.glob(os.path.join(fl_path, "**/*.TextGrid"), recursive=True)

phone_dict = {}
for file_path in tqdm(file_list):
    tg = textgrid.TextGrid.fromFile(file_path)
    phone_align = tg[1]
    for pa in phone_align:
        phone = pa.mark
        if phone in phone_dict:
            phone_dict[phone][1] += 1 
        else:
            phone_dict[phone] = [len(phone_dict), 1]

with open(out_file, 'w') as fp:
    json.dump(phone_dict, fp, indent=6)