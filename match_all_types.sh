
mfa_json=$1
data_pth=$2
save_dir=$3

for i in 2944 2816 2688 2560 2048 1536 1024 512;
do
    mkdir -p $save_dir/phone-uniform-pruned-$i
    mkdir -p $save_dir/phone-uniform-$i
    for j in phone-type gender pitch duration;
    do 
        python3 match_phone.py  --model-pth ~/Desktop/checkpoints/Journal_checkpoint/row-pruning/states_prune_"$i"_pruned.ckpt \
                                --mfa-json $mfa_json \
                                --save-pth $save_dir/phone-uniform-pruned-$i/$j.pkl \
                                --fp 20 \
                                --mean-std-pth mean-std-dir/libri-960-mean-std.npy \
                                --data-pth $data_pth \
                                --phone-type mid-phone \
                                --extra-class $j \
                                --merge

        python3 match_phone.py  --model-pth ~/Desktop/checkpoints/Journal_checkpoint/row-pruning/states_prune_"$i"_tuned.ckpt \
                                --mfa-json $mfa_json \
                                --save-pth $save_dir/phone-uniform-$i/$j.pkl \
                                --fp 20 \
                                --mean-std-pth mean-std-dir/libri-960-mean-std.npy \
                                --data-pth $data_pth \
                                --phone-type mid-phone \
                                --extra-class $j \
                                --merge 
    done 
done 