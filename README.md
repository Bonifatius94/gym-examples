
# Reinforcement Learning - Some Gym Examples

## About
This project is meant to provide some beginner reinforcement learning
trainings for popular OpenAI Gym games using TensorFlow models.

## System Setup

### Install Docker and Docker-Compose

```sh
sudo apt-get update && sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER && reboot
# CAUTION: last command reboots your machine
```

### Install NVIDIA GPU Support For Docker (Optional)
See this [tutorial](./GPU_SETUP.md) for further setup information.

## Run Training

```sh
PYTHONUNBUFFERED=1 docker-compose -f compose/training-nogpu-compose.yml up --build
# info: if you want to train with GPU support, use the corresponding
#       yml file named *gpu* instead of *nogpu*
```
