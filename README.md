# Speech-SSL-FFN-Analysis

# Preprocessing 
1. Generate framewise phoneme label with [Montreal Force aliger](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
2. Get phoneme label set 
```
python3 get_phone_label.py ~/LibriSpeechMFA/dev-clean/ ./data/pl_pth.json 
```
3. Merge phoneme to total number of 39 if needed
```
python3 merge_phone.py ./data/pl_pth.json ./data/pl_pth_merge.json 
```
4. Get phoneme label (index) aligning to frame in json format 
```
python3 get_phone_align_to_frame.py ~/LibriSpeechMFA/dev-clean/ ./data/pl_pth_merge.json ./data/dev-clean-framewise-phone.json 20 1 
```

# Calculate matching probability of phonemes and keys 
HuBERT Base on LibriSpeech dev-clean subset as example (for gender)
```
python3 match_phone_s3prl.py -m hubert_base -f info/dev-clean-framewise-phone-merge-20ms.json -s match_prob_hubert.pkl -d ~/LibriSpeech/dev-clean/ -t mid-phone -c gender
```
