# Merge primary stress and secondary stress phoneme
import argparse
import json 

def parse_num(k):
    return ''.join([x for x in k if not x.isdigit()])

def main(phone_label, save_pth):
    with open(phone_label, 'r') as fp:
        p = json.load(fp)
    keys = {}
    idx = 0 
    for k, v in p.items():
        i, num = v 
        n = parse_num(k)
        if n not in keys:
            keys[n] = [idx, num]
            idx += 1 
        else:
            ori_idx, ori_num = keys[n]
            keys[n] = [ori_idx, ori_num+num]

    with open(save_pth, 'w') as fp:
        json.dump(keys, fp, indent=6)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phone-label', help='Phone label path')
    parser.add_argument('-s', '--save-pth', help='Save path')
    args = parser.parse_args()
    main(**vars(args))