
montreal_force_aligner_dir=$1 # /livingrooms/ray_chen/LibriSpeechMFA/dev-clean on BattleShip
librispeech_root=$2
stage=$3

# Stage 0: Generate framewise phoneme label 
if [[ $stage -le 0 ]]; then
echo "Stage 0: Generate framewise phoneme label"
python3 framewise_phone/get_phone_label.py $montreal_force_aligner_dir phone-label.json
python3 framewise_phone/merge_phone.py -p phone-label.json -s phone-label-merge.json
python3 framewise_phone/get_phone_align_to_frame.py $montreal_force_aligner_dir phone-label-merge.json framewise-phone-merge.json 20 1
fi
if [[ $stage -le 1 ]]; then
# Stage 1: Extract gender information 
echo "Stage 1: Extract gender information from dataset"
python3 preprocess/ext_spk_from_librispeech.py $librispeech_root/SPEAKERS.TXT speaker.json dev-clean
fi 
if [[ $stage -le 2 ]]; then
# Stage 2: Calculate matching probability matrix
echo "Stage 2: Calculate matching probability matrix for gender for HuBERT base"
python3 match_phone_s3prl.py -m hubert_base -f framewise-phone-merge.json -s hubert-gender.pkl -d $librispeech_root/dev-clean/ -t mid-phone -c gender --merge --extra-info speaker.json
fi 
# Stage 3: Find property-specific keys 
echo "Stage 3: Find property-specific keys"
python3 find_ps_keys.py -p hubert-gender.pkl -s hubert-gender-ps-keys.json -l phone-label-merge.json