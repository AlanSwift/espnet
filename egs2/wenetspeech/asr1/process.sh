

scp_path=/home/v-shenkai/wenet/dump/raw/train_L/wav.scp
output_path=/blob/v-shenkai/data/wenet/inference/train/
split_scps=""
_nj=200

for n in $(seq "${_nj}"); do
    mkdir -p ${output_path}/${n}
    split_scps+=" ${output_path}/${n}/wav.scp"
done
utils/split_scp.pl "${scp_path}" ${split_scps}
