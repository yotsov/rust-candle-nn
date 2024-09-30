# rust-candle-nn
Example of how to train a neural network using Rust and Candle.

# How to install CUDA-toolkit
In order to run the example in this code, cuda-toolkit needs to be installed.
That requires an Nvidia graphics card.

To do the installation on Fedora 39, following instructions on https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#fedora:

sudo dnf remove *nvidia*
sudo dnf module disable nvidia-driver
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
sudo dnf --disablerepo="rpmfusion-nonfree*" module install nvidia-driver:latest-dkms
sudo dnf update
sudo reboot
nvidia-smi --query-gpu=compute_cap --format=csv

sudo dnf --disablerepo="rpmfusion-nonfree*" install cuda-toolkit
/usr/local/cuda-12.6/bin/nvcc --version

In ~/.bashrc:

export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"

Then in new terminal window:
nvcc --version