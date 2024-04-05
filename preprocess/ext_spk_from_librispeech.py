import sys 
import json

libri_spk_info = sys.argv[1] # SPEAKER.txt
save_pth = sys.argv[2]
dataset_name = sys.argv[3]

keys = {}
with open(libri_spk_info, 'r') as fp:
    for i in range(12):
        fp.readline() # Ignore the header
    for x in fp:
        ID, gender, dataset, _, _ = x.strip().split('|')
        ID = ID.strip()
        gender = gender.strip()
        dataset = dataset.strip()
        if dataset == dataset_name:
            keys[ID] = gender

with open(save_pth, 'w') as fp:
    json.dump(keys, fp, indent=3)



    