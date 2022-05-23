# inference environment for P40 envorinment

# install environment
export LANG=en_US.UTF-8
export LANGUAGE=
export LC_CTYPE="en_US.UTF-8"
export LC_NUMERIC=zh_CN.UTF-8
export LC_TIME=zh_CN.UTF-8
export LC_COLLATE="en_US.UTF-8"
export LC_MONETARY=zh_CN.UTF-8
export LC_MESSAGES="en_US.UTF-8"
export LC_PAPER=zh_CN.UTF-8
export LC_NAME=zh_CN.UTF-8
export LC_ADDRESS=zh_CN.UTF-8
export LC_TELEPHONE=zh_CN.UTF-8
export LC_MEASUREMENT=zh_CN.UTF-8
export LC_IDENTIFICATION=zh_CN.UTF-8
export LC_ALL=
sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
sudo locale-gen zh_CN.UTF-8

job_id=$1
echo $job_id
exit 0


PHILLY_USER=${USER}
echo "PHILLY_USER:${PHILLY_USER}"

#path="/var/storage/shared/msrmt/v-jinx/ASR_NAS/espnet/tools/venv/bin:/var/storage/shared/msrmt/xuta/v-yileng/asr/se4asr/SCTK/bin"
#path="/opt/conda/envs/pytorch-py3.6/bin:/opt/conda/bin"

# sudo rm /etc/sudoers.d/${PHILLY_USER}
sudo touch /etc/sudoers.d/${PHILLY_USER}
sudo chmod 777 /etc/sudoers.d/${PHILLY_USER}
sudo echo "Defaults        secure_path=\"$path:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" > /etc/sudoers.d/${PHILLY_USER}
sudo chmod 0440 /etc/sudoers.d/${PHILLY_USER}

sudo chmod -R 777 /opt/conda




sudo apt-get install -y vim

export PATH=$PATH:/opt/conda/bin

sudo apt-get install nano tmux libsndfile1 bci jq screen vim -y
sudo apt-get install libsndfile1 -y

pip uninstall apex


echo "set default github names"

git config --global user.name "AlanSwift"
git config --global user.email "shenkai200861@gmail.com"


cp -r /blob/v-shenkai/envs/.tmux /home/v-shenkai/
cp -r /blob/v-shenkai/envs/.tmux.conf /home/v-shenkai/
cp -r /blob/v-shenkai/envs/.bashrc /home/v-shenkai/
# echo "alias ts=\"tmux new -s\"" >> /home/v-shenkai/.bashrc
# echo "alias ta=\"tmux attach -t\"" >> /home/v-shenkai/.bashrc


echo "Finish installing"


sudo apt-get install cmake -y
sudo apt-get install sox -y
sudo apt-get install libsndfile1-dev -y
sudo apt-get install ffmpeg -y
sudo apt-get install flac -y


# install espnet
source /home/v-shenkai/.bashrc

echo "create conda env: espnet"
conda create -n espnet python=3.7 -y
source activate espnet
# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch




echo "Download espnet"
HOME_DIR=/home/v-shenkai/
DOWNLOAD_DIR=/home/v-shenkai/download/
mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR
git clone https://github.com/AlanSwift/espnet.git
cd espnet

echo "Install espnet"
cd tools
make

echo "Copy features to native storage to speed up inference"
mkdir -p $HOME_DIR/wenet
cp -r /blob/v-shenkai/data/wenet/dataset/wenet_exp/wenet/dump $HOME_DIR/wenet/
cp -r /blob/v-shenkai/data/wenet/dataset/wenet_exp_new/exp $HOME_DIR/wenet/
ln -s /blob/v-shenkai/data/wenet/dataset/wenet_exp/wenet/untar /home/v-shenkai/wenet/untar

echo "Inference"
cd $DOWNLOAD_DIR/espnet/egs2/wenetspeech/asr1
./asr_inference.sh --job_id $job_id

