
mfa_json=$1

for i in 2944 2816 2688 2560 2048 1536 1024 512;
do
    mkdir ./npy-r/phone-uniform-pruned-$i
    mkdir ./npy-r/phone-uniform-$i
    for j in phone-type gender pitch duration;
    do 
        python3 match_phone.py  --model-pth ~/Desktop/checkpoints/Journal_checkpoint/row-pruning/states_prune_"$i"_pruned.ckpt \
                                --mfa-json $mfa_json \
                                --save-pth ./npy-r/phone-uniform-pruned-$i/$j.npy \
                                --fp 20 \
                                --mean-std-pth mean-std-dir/libri-960-mean-std.npy \
                                --data-pth ~/Desktop/dataset/LibriSpeech/test-clean/ \
                                --phone-type mid-phone \
                                --extra-class $j

        python3 match_phone.py  --model-pth ~/Desktop/checkpoints/Journal_checkpoint/row-pruning/states_prune_"$i"_tuned.ckpt \
                                --mfa-json $mfa_json \
                                --save-pth ./npy-r/phone-uniform-$i/$j.npy \
                                --fp 20 \
                                --mean-std-pth mean-std-dir/libri-960-mean-std.npy \
                                --data-pth ~/Desktop/dataset/LibriSpeech/test-clean/ \
                                --phone-type mid-phone \
                                --extra-class $j
    done 
done 