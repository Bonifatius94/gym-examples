
# GPU Setup (Ubuntu 20.04)

## Install the proprietary NVIDIA driver

1) Go to 'Software & Updates' -> 'Additional Drivers'
2) Select a proprietary NVIDIA driver (e.g. nvidia-driver-470)

## Install the NVIDIA Docker Runtime

```sh
# install cURL for sending HTTP requests
sudo apt-get update && sudo apt-get install -y curl

# register the NVIDIA Docker PPA as apt source
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
        | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# install NVIDIA Docker from the official NVIDIA PPA
sudo apt-get update && sudo apt-get install -y nvidia-docker2
```
