## Livox Detection for ITS applications

This repository is built based on the previous work [livox detection](https://github.com/Livox-SDK/livox_detection).

We further deploy the pre-trained model on our own devices and accelerate by TensorRT with an FPS of `15Hz` on GTX 1080Ti out of the original [pretrained model](https://github.com/Livox-SDK/livox_detection) whose FPS merely `5Hz` .

We deploy deepfusion methods on [MOT](https://github.com/wangxiyang2022/DeepFusionMOT).

<!--
## Demo
We fine-tune the model on our own dataset for demonstration as below:
![demo1](demo/its1%2000_00_00-00_00_30.gif) 
![demo2](demo/its2%2000_00_00-00_00_30.gif)
-->



# Dependencies
- `python3.6+`
- `tensorflow1.13+` (tested on 1.13.0)
- `pybind11`
- `ros`

# Installation

1. Clone this repository.
2. Clone `pybind11` from [pybind11](https://github.com/pybind/pybind11).
```bash
$ cd utils/lib_cpp
$ git clone https://github.com/pybind/pybind11.git
```
3. Compile C++ module in utils/lib_cpp by running the following command.
```bash
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
4. copy the `lib_cpp.so` to root directory:
```bash
$ cp lib_cpp.so ../../../
```

