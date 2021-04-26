# WinML inference time

## Problem statement

It seems that WinML inference duration is corelated with the frequency with which it is called. The more frequent called the faster it is.

## Experiment

This repository contains a simple application which calls in a loop the WinML-based inference with various sleep times between iterations (`0ms`, `50ms`, `100ms`, `150ms`, `200ms`) and records the computed durations in a text file (`winml_durations.txt`). The ONNX model shall be provided as a first command line argument. The model's input is randomly generated, the output is retrieved and ignored. The application at startup enumerates available GPU devices in order to determine whether inference should be run on GPU (DirectX) or CPU (default), if there is no GPU device available. There is also a simple python script [visualize_durations.py](visualize_durations.py) visualizing the collected durations.

### Running the app

 * Open Visual Studio and load the [winml_inference_time.sln](winml_inference_time.sln) solution.
 * Build the solution in the `Release` mode for your architecture
 * Run the generated `winml_inference_time.exe` providing path to the ONNX model as the command line argument, for instance use the [yolov2-coco-9.onnx](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx) model, i.e.:
```
$ ./x64/Release/winml_inference_time.exe ~/Downloads/yolov2-coco-9.onnx
Using model C:/Users/tomaszmi/Downloads/yolov2-coco-9.onnx
GPU device: description: NVIDIA GeForce GTX 1050, type: default, vendor_id: 4318, device_id: 7297, dedicated_video_memory: 2073419776, dedicated_system_memory: 0, shared_system_memory: 17159557120
GPU device: description: NVIDIA GeForce GTX 1050, type: high performance, vendor_id: 4318, device_id: 7297, dedicated_video_memory: 2073419776, dedicated_system_memory: 0, shared_system_memory: 17159557120
GPU device: description: NVIDIA GeForce GTX 1050, type: minimum power, vendor_id: 4318, device_id: 7297, dedicated_video_memory: 2073419776, dedicated_system_memory: 0, shared_system_memory: 17159557120
Running 100 inference iterations with 0 [ms] sleep in between
Running 100 inference iterations with 50 [ms] sleep in between
Running 100 inference iterations with 100 [ms] sleep in between
Running 100 inference iterations with 150 [ms] sleep in between
Running 100 inference iterations with 200 [ms] sleep in between
The inference durations have been saved in ./winml_durations.txt
Run `python ./winml_durations.txt` to visualize the results
$
```

The inference durations are stored to the `winml_durations.txt` file, generated in the directory the app has been run from. In order to visualize the collected results run the [visualize_durations.py](visualize_durations.py) script:

```
$ ~/Anaconda3/python visualize_durations.py winml_durations.txt
```

