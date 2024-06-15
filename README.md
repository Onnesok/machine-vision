##  Tensorflow
![copy](https://www.tensorflow.org/images/tf_logo_horizontal.png)
 
 This repository contains basic codes of tensorflow object detection. It's maintained for a baseline code start.

[![Compatibility](https://img.shields.io/badge/python-3.12-brightgreen.svg)](https://www.python.org/)
[![Modified](https://img.shields.io/badge/Coverage-working-red)](machine-vision)
[![Hits](https://hits.sh/github.com/Onnesok/machine-vision.svg)](https://hits.sh/github.com/Onnesok/machine-vision/)
## Installation

First create a virtual environment named tf_vision to avoid package conflicts. use ``cmd or bash`` to do it.

```bash
python3 -m venv tf_vision
```
Now source or activate environment.
#### For linux

```bash
source ./tf_vision/bin/activate
```

#### For windows cmd

```bash
tf_vision\Scripts\activate
```

## Install tensorflow and opencv
For latest version......
```bash
pip install tensorflow
```
```bash
pip install opencv-python
```

wait.... select interpreter of vs code to your selected virtual environment from the bottom right corner.

But it will run using cpu resources. So, lets run using ``gpu`` so that it can use more cores.
But, native windows does not support gpu on tensorflow version greater than ``2.10.0``..........Also, package seems to be removed right now for ``tensorflow-gpu==2.10.0`` using ``pip``. 

For compatiblity see here ==> https://www.tensorflow.org/install/source_windows

So, lets do it using ``conda``

Create virtual env using conda first and activate.
```bash
conda create -n conda_env python==3.9
conda activate conda_env
```
Now, select interpreter of vs code to your selected virtual environment from the bottom right corner.

Ok, now install cudatoolkit and cudnn for gpu support.
```bash
conda install cudatoolkit=11.2 cudnn=8.1 -c=conda-forge
```
Now install tensorflow-gpu version 2.10.0 and opencv-python
```bash
pip install tensorflow-gpu==2.10.0
pip install opencv-python
```
Done.
