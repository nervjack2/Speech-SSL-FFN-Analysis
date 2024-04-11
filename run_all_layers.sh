
pkl_pth=$1
save_dir=$2
phone_label_pth=$3
mode=$4


mkdir $save_dir 

for i in {1..12}
do 
python3 visualize_phone.py --pkl-pth $pkl_pth \
                        --save-pth $save_dir/layer-$i.png \
                        --phone-label-pth $phone_label_pth \
                        --mode $mode \
                        --layer-n $i
done 