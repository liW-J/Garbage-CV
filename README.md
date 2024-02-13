
## Documentation

An automatic garbage classification software Based on YOLOv5.

Code is running on **Jetson Xavier NX** with Servos, Microcontrollers and A Camera.

Provide trained models in `weight/best.pt`, can sort garbage into four categories: Hazardous/Recyclable/FoodWaste/Others.

After sort is completed, device will control the servo to deliver the garbage into corresponding trash can.

It's OK that maybe you need modify Servo rotation parameters or train your own model.

This document helps you successfully run code on NVIDIA Jetson platform.

## Quick Start Examples

### Install Necessary Packages

Access the terminal of Jetson device, install pip and upgrade it.

```bash
sudo apt update
sudo apt install -y python3-pip
pip3 install --upgrade pip
```

Clone repo and install [requirements.txt](https://github.com/liW-J/Garbage-CV/main/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/liW-J/Garbage-CV  # clone
cd Garbage-CV
pip install -r requirements.txt  # install
```

### Install PyTorch and Torchvision

We cannot install PyTorch and Torchvision from pip because they are not compatible to run on Jetson platform which is based on **ARM aarch64 architecture**. Therefore, we need to manually install pre-built PyTorch pip wheel and compile/ install Torchvision from source.

Visit [this page](https://forums.developer.nvidia.com/t/pytorch-for-jetson) to access all the PyTorch and Torchvision links.

Please install torch according to your JetPack version and install torchvision depending on the version of PyTorch that you have installed.

### Inference with main.py

`main.py` runs inference on a variety of sources.

Trained model has been placed in `weight/best.pt`  or you can use your own best model. 

results will save to `runs/detect`.

Run with following command to inference and control modules:
```bash
python main.py --source 0 --weight weight/best.pt
```
