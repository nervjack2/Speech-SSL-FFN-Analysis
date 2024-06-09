
mfa_json=$1
data_pth=$2
save_dir=$3
model_dir=$4
python_pth=$5

#for i in 2944 2816 2688 2560 2048 1536 1024 512;
for i in 2997 2950 2918 2898;
do
    mkdir -p $save_dir/phone-uniform-pruned-$i
    mkdir -p $save_dir/phone-uniform-$i
    # for j in phone-type gender pitch duration;
    for j in phone-type gender;
    do 
        $python_pth match_phone.py  --model-pth $model_dir/states_prune_"$i"_pruned.ckpt \
                                --mfa-json $mfa_json \
                                --save-pth $save_dir/phone-uniform-pruned-$i/$j.pkl \
                                --fp 20 \
                                --mean-std-pth mean-std-dir/libri-960-mean-std.npy \
                                --data-pth $data_pth \
                                --phone-type mid-phone \
                                --extra-class $j \
                                --merge

        $python_pth match_phone.py  --model-pth $model_dir/states_prune_"$i"_tuned.ckpt \
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
