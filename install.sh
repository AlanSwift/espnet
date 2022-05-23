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
sudo locale-gen en_US.UTF-8
sudo locale-gen zh_CN.UTF-8

sudo apt-get install cmake
sudo apt-get install sox
sudo apt-get install libsndfile1-dev
sudo apt-get install ffmpeg
sudo apt-get install flac


# install espnet
source /home/v-shenkai/.bashrc

echo "create conda env: espnet"
conda create -n espnet python=3.7 -y
source activate espnet
# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch




echo "Download espnet"
WORK_DIR=/home/v-shenkai/download/
mkdir -p $WORK_DIR
cd $WORK_DIR
git clone https://github.com/AlanSwift/espnet.git
cd espnet

echo "Install espnet"
cd tools
make
