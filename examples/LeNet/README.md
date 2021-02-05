# LeNet Example

This example showcases the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)-based handwritten digits classification by WebNN API.

This example leverages the network topology of [the LeNet example of Caffe*](https://github.com/BVLC/caffe/tree/master/examples/mnist), the weights (`lenet.bin`) and `MnistUByte` reader of [the LeNet example of OpenVINO*](https://github.com/openvinotoolkit/openvino/tree/master/inference-engine/samples/ngraph_function_creation_sample). This example uses one-channel UByte picture as an input. The sample UByte pictures (`*.idx`) are extracted from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Usage:

Running the example with the -h option:
```sh
out/Release/LeNet -h
```
yields the following usage message:
```sh
LeNet Example [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Path to image.
    -m "<path>"             Required. Path to a .bin file with weights for the trained model.
```

The location of the weights file and input images: [lenet.bin](/examples/LeNet/lenet.bin), [*.idx](/examples/LeNet/images).

## Example Output

For example, to classify an UByte image, please run the following command:

```sh
out/Release/LeNet -i ./examples/LeNet/images/9.idx -m ./examples/LeNet/lenet.bin
```

By default, the example outputs top-3 classification results as below:

```
Info: Compute done, inference time: 1.49335 ms

Top 3 results:

#   Label Probability

0   9     100.00%
1   2     0.00%
2   0     0.00%
```
