# TF-TRT Optimization for InceptionV3

This repository demonstrates the optimization of the InceptionV3 model using TensorFlow-TensorRT (TF-TRT) for various precision modes. It includes code to convert a TensorFlow SavedModel into an optimized TF-TRT graph and perform benchmarking for different precision modes.

## Benchmark Results

| Precision Mode | Average Inference Time (ms) | Throughput (images/s) |
|----------------|--------------------------|-----------------------|
| Original InceptionV3 | 91.9 | 44 |
| FP32 Optimized InceptionV3 | 23.9 | 168 |
| FP16 Optimized InceptionV3 | 9.8 | 408 |
| INT8 Optimized InceptionV3 | 9.2 | 432 |

## Getting Started

### Prerequisites

- TensorFlow (2.x)
- TensorFlow-TensorRT (TF-TRT)

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/tf-trt-inceptionv3.git
   cd tf-trt-inceptionv3

2. Run the provided noetbook preferably in colab.

## Observations

FP16 showed significant improvement from FP32 however, there seems to be ~10% throughput improvement between FP16 and INT8. This might be due to inefficiencies in the TensorRT engine to introduce INT8 instructions for all segments in the model graph. 

## TODO

Tune INT8 compilation and re-check calibration

