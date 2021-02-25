# LeNet Example

This example showcases the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)-based handwritten digits classification by WebNN API.

This example leverages the network topology of [the LeNet example of Caffe*](https://github.com/BVLC/caffe/tree/master/examples/mnist), the weights (`lenet.bin`) and `MnistUByte` reader of [the LeNet example of OpenVINO*](https://github.com/openvinotoolkit/openvino/tree/master/inference-engine/samples/ngraph_function_creation_sample). This example uses one-channel UByte picture as an input. The sample UByte pictures (`*.idx`) are extracted from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Usage

```sh
> out/Release/LeNet -h

LeNet [OPTIONs]

Options:
    -h                      Print this message.
    -i "<path>"             Required. Path to an image.
    -m "<path>"             Required. Path to a .bin file with trained weights.

```

## Example Output

```sh
> out/Release/LeNet -m examples/LeNet/lenet.bin -i examples/LeNet/images/9.idx
Info: Compilation Time: 27.3588 ms
Info: Execution Time: 0.919068 ms

Prediction Result:
#   Label Probability
0   9     100.00%
1   2     0.00%
2   0     0.00%

Info: Done.
```

You can also try other example [images](/examples/LeNet/images).
