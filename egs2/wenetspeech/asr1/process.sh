scp_path=/home/v-shenkai/wenet/dump/raw/train_L/wav.scp
output_path=/blob/v-shenkai/data/wenet/dataset/wenet_exp/wenet/dump/raw/infer_train_
# output_path=/home/v-shenkai/wenet/dump/raw/infer_train_
split_scps=""
_nj=95

for n in $(seq "${_nj}"); do
    mkdir -p ${output_path}${n}
    split_scps+=" ${output_path}${n}/wav.scp"
done
utils/split_scp.pl "${scp_path}" ${split_scps}