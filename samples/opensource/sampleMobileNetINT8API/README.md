# Performing Inference In INT8 Precision (modified for pre-quantized MobileNetV1)


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

This is a modified sample, originally based on sampleINT8API. It performs INT8 inference without using the INT8 calibrator; using the user provided per activation tensor dynamic range. INT8 inference is available only on GPUs with compute capability 6.1 or 7.x and supports Image Classification ONNX models such as ResNet-50, VGG19, and MobileNet. **This sample has been adjusted to using a pre-quantized MobileNetV1 by default** (instead of originally ResNet-50).

## How does this sample work?

In order to perform INT8 inference, you need to provide TensorRT with the dynamic range for each network tensor, including network input and output tensor. One way to choose the dynamic range is to use the TensorRT INT8 calibrator. But if you don't want to go that route (for example, let’s say you used quantization-aware training or you just want to use the min and max tensor values seen during training), you can skip the INT8 calibration and set custom per-network tensor dynamic ranges. This sample implements INT8 inference for a pre-quantized ONNX model of MobileNetV1 using per-tensor dynamic ranges specified in an input file.

This modified sample takes the ONNX MobileNetV1 model that was used for [NVIDIA's MLPerf Inference 0.5 submission](https://github.com/mlperf/inference_results_v0.5/tree/master/closed/NVIDIA/code/mobilenet/tensorrt#model-source).

## Prerequisites

In addition to the model file and input image, you will need per-tensor dynamic range stored in a text file along with the ImageNet label reference file.

The following required files are included in the package and are located in the `data/int8_api` directory.

- `reference_labels.txt`
The ImageNet reference label file.

- `airliner.ppm`
The image to be inferred.

- `mobilenet_last_dynamic_range.txt`
The MobileNetV1 per-tensor dynamic ranges file. Note on modifications to original `sampleINT8API`: If a tensor is not specified in this text file, the ranges will default to `[-127, 127]`. Since we assume to use a pre-quantized model by default, we assume dynamic ranges `[-127, 127]` for all tensors except for the output tensor, being the only entry here:
```
	[Contents of mobilenet_last_dynamic_range.txt]
	169:150559.92626953125
```
This value is derived from NVIDIA's [calibration cache file for MobileNetV1](https://github.com/mlperf/inference_results_v0.5/blob/master/closed/NVIDIA/code/mobilenet/tensorrt/calibrator.cache#L85) on the MLPerf Inference 0.5 repo, namely line `85`: `fc_replaced_output: 4494305c`. `4494305c` is the hexadecimal scaling factor for the outputs of the penultimate Gemm node in the original ONNX file. The float equivalent is `1185.51f`. To obtain the corresponding (symmetric) dynamic range value, we multiply this value with `127.0f` and achieve  `1185.51f * 127.0f = 150559.92626953125`.


- `mobilenet_quantized_opt.onnx`
The pre-quantized MobileNetV1 model, created as shown below:

1.  Download the ONNX MobileNetV1 model that was used for [NVIDIA's MLPerf Inference 0.5 submission](https://github.com/mlperf/inference_results_v0.5/tree/master/closed/NVIDIA/code/mobilenet/tensorrt#model-source):
    ```
    wget https://zenodo.org/record/3353417/files/Quantized%20MobileNet.zip
    ```
    For further details, see the included `.txt` and `.pdf` files included [in the archive]( https://zenodo.org/record/3353417).

2.  Unpackage the downloaded archive:
    ```
    unzip 'Quantized MobileNet.zip'
    ```

3.  Recommended to run on x86 (for ease of installing ONNX): Make sure your installed ONNX Python version is the required one (see `requirements.txt`). Then run the script `opt_mobilenet_onnx.py` with the following expected console output:
	```console
	user@machine:/path/to/TensorRT/samples/opensource/sampleMobileNetINT8API$ python3 opt_mobilenet_onnx.py
	Reading the original MobileNetV1 model from ./Quantized MobileNet/mobilenet_sym_no_bn.onnx...
	Removing Constant node with inputs [] and outputs ['167'] from the graph...
	Removing Reshape node with inputs ['166', '167'] and outputs ['168'] from the graph...
	Removing Gemm node with inputs ['168', '82', '83'] and outputs ['169'] from the graph...
	Adding 1x1 Conv node last_conv_fc with inputs ['166', '82', '83'] and outputs ['169'] to the graph...
	Saving modified MobileNetV1 model to mobilenet_quantized_opt.onnx...
	Done.
	```
4.  Copy `mobilenet_quantized_opt.onnx` to the existing `data/int8_api/` directory (on the device you want to run this sample on). By default when installing TensorRT with a `.deb` file, this folder is located at `/usr/src/tensorrt/data/int8_api/`.

## Running the sample

1.  Copy this folder to the `<TensorRT root directory>/samples/` directory. Compile this sample by running `make` in that directory. The binary named `sample_mobilenet_int8_api` will be created in the `<TensorRT root directory>/bin` directory.

	```
	cd <TensorRT root directory>/samples/sampleMobileNetINT8API
	make
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run the sample to perform INT8 inference on a classification network, for example here by default, MobileNetV1 (run on DRIVE AGX):
	`./sample_mobilenet_int8_api [-v or --verbose]`

	```console
	nvidia@tegra-ubuntu:/usr/src/tensorrt/bin$ ./sample_mobilenet_int8_api -h
	Usage: ./sample_mobilenet_int8_api [-h or --help] [--model=model_file] [--ranges=per_tensor_dynamic_range_file]
	[--image=image_file] [--reference=reference_file] [--write_tensors] [--network_tensors_file=network_tensors_file]
	[--data=/path/to/data/dir] [--useDLACore=<int>] [--topBottomK=<int>] [--fp32] [--safeGpuInt8]
	[-v or --verbose]
	```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following (run on DRIVE AGX):

	```console
	nvidia@tegra-ubuntu:/usr/src/tensorrt/bin$ ./sample_mobilenet_int8_api
	&&&& RUNNING TensorRT.sample_mobilenet_int8_api # ./sample_mobilenet_int8_api
	[02/16/2020-18:41:45] [I] Please follow README.md to generate missing input files.
	[02/16/2020-18:41:45] [I] Validating input parameters. Using following input files for inference.
	[02/16/2020-18:41:45] [I]     Model File: ../data/int8_api/mobilenet_quantized_opt.onnx
	[02/16/2020-18:41:45] [I]     Image File: ../data/int8_api/airliner.ppm
	[02/16/2020-18:41:45] [I]     Reference File: ../data/int8_api/reference_labels.txt
	[02/16/2020-18:41:45] [I]     Dynamic Range File: ../data/int8_api/mobilenet_last_dynamic_range.txt
	[02/16/2020-18:41:45] [I] Building and running a INT8 inference engine on GPU for ../data/int8_api/mobilenet_quantized_opt.onnx
	[02/16/2020-18:41:47] [I] Setting Per Tensor Dynamic Range
	[02/16/2020-18:41:47] [W] [TRT] Calibrator is not being used. Users must provide dynamic range for all tensors that are not Int32.
	[02/16/2020-18:41:47] [I] [TRT] 
	[02/16/2020-18:41:47] [I] [TRT] --------------- Layers running on DLA: 
	[02/16/2020-18:41:47] [I] [TRT] 
	[02/16/2020-18:41:47] [I] [TRT] --------------- Layers running on GPU: 
	[02/16/2020-18:41:47] [I] [TRT] (Unnamed Layer* 0) [Convolution] + (Unnamed Layer* 2) [Activation], (Unnamed Layer* 3) [Convolution] + (Unnamed Layer* 5) [Activation] + (Unnamed Layer* 6) [Convolution] + (Unnamed Layer* 8) [Activation], (Unnamed Layer* 9) [Convolution] + (Unnamed Layer* 11) [Activation], (Unnamed Layer* 12) [Convolution] + (Unnamed Layer* 14) [Activation], (Unnamed Layer* 15) [Convolution] + (Unnamed Layer* 17) [Activation] + (Unnamed Layer* 18) [Convolution] + (Unnamed Layer* 20) [Activation], (Unnamed Layer* 21) [Convolution] + (Unnamed Layer* 23) [Activation], (Unnamed Layer* 24) [Convolution] + (Unnamed Layer* 26) [Activation], (Unnamed Layer* 27) [Convolution] + (Unnamed Layer* 29) [Activation], (Unnamed Layer* 30) [Convolution] + (Unnamed Layer* 32) [Activation], (Unnamed Layer* 33) [Convolution] + (Unnamed Layer* 35) [Activation], (Unnamed Layer* 36) [Convolution] + (Unnamed Layer* 38) [Activation], (Unnamed Layer* 39) [Convolution] + (Unnamed Layer* 41) [Activation], (Unnamed Layer* 42) [Convolution] + (Unnamed Layer* 44) [Activation], (Unnamed Layer* 45) [Convolution] + (Unnamed Layer* 47) [Activation], (Unnamed Layer* 48) [Convolution] + (Unnamed Layer* 50) [Activation], (Unnamed Layer* 51) [Convolution] + (Unnamed Layer* 53) [Activation], (Unnamed Layer* 54) [Convolution] + (Unnamed Layer* 56) [Activation], (Unnamed Layer* 57) [Convolution] + (Unnamed Layer* 59) [Activation], (Unnamed Layer* 60) [Convolution] + (Unnamed Layer* 62) [Activation], (Unnamed Layer* 63) [Convolution] + (Unnamed Layer* 65) [Activation], (Unnamed Layer* 66) [Convolution] + (Unnamed Layer* 68) [Activation], (Unnamed Layer* 69) [Convolution] + (Unnamed Layer* 71) [Activation], (Unnamed Layer* 72) [Convolution] + (Unnamed Layer* 74) [Activation], (Unnamed Layer* 75) [Convolution] + (Unnamed Layer* 77) [Activation], (Unnamed Layer* 78) [Convolution] + (Unnamed Layer* 80) [Activation], (Unnamed Layer* 81) [Pooling], (Unnamed Layer* 82) [Convolution], 
	[02/16/2020-18:45:27] [W] [TRT] No implementation of layer (Unnamed Layer* 82) [Convolution] obeys the requested constraints in strict mode. No conforming implementation was found i.e. requested layer computation precision and output precision types are ignored, using the fastest implementation.
	[02/16/2020-18:45:28] [W] [TRT] No implementation obeys reformatting-free rules, at least 1 reformatting nodes are needed, now picking the fastest path instead.

	[02/16/2020-18:47:20] [I] [TRT] Detected 1 inputs and 1 output network tensors.
	[02/16/2020-18:47:21] [I] Top-1 predicted class, activation value: airliner, 117816
	[02/16/2020-18:47:21] [I] Top-2 predicted class, activation value: warplane, 97372.9
	[02/16/2020-18:47:21] [I] Top-3 predicted class, activation value: wing, 91125.6
	[02/16/2020-18:47:21] [I] Top-4 predicted class, activation value: space shuttle, 88287.7
	[02/16/2020-18:47:21] [I] Top-5 predicted class, activation value: projectile, 84140.4
	[02/16/2020-18:47:21] [I] Bottom-1 predicted class, activation value: mosquito net, -34046.6
	[02/16/2020-18:47:21] [I] Bottom-2 predicted class, activation value: yellow lady's slipper, -32296.7
	[02/16/2020-18:47:21] [I] Bottom-3 predicted class, activation value: Komodo dragon, -30606.4
	[02/16/2020-18:47:21] [I] Bottom-4 predicted class, activation value: chocolate sauce, -30140.4
	[02/16/2020-18:47:21] [I] Bottom-5 predicted class, activation value: overskirt, -29748.3
	[02/16/2020-18:47:21] [I] SampleINT8API result - Detected:
	[02/16/2020-18:47:21] [I] [1]  airliner
	[02/16/2020-18:47:21] [I] [2]  warplane
	[02/16/2020-18:47:21] [I] [3]  wing
	[02/16/2020-18:47:21] [I] [4]  space shuttle
	[02/16/2020-18:47:21] [I] [5]  projectile
	&&&& PASSED TensorRT.sample_mobilenet_int8_api # ./sample_mobilenet_int8_api
	```

	This output shows that the sample ran successfully; `PASSED`.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
	Usage: ./sample_mobilenet_int8_api [-h or --help] [--model=model_file] [--ranges=per_tensor_dynamic_range_file]
	[--image=image_file] [--reference=reference_file] [--write_tensors] [--network_tensors_file=network_tensors_file]
	[--data=/path/to/data/dir] [--useDLACore=<int>] [--topBottomK=<int>] [--fp32] [--safeGpuInt8]
	[-v or --verbose]
	-h or --help. Display This help information
	--model=model_file.onnx or /absolute/path/to/model_file.onnx. Generate model file using README.md in case
	it does not exists. Defaults to mobilenet_quantized_opt.onnx.
	--image=image.ppm or /absolute/path/to/image.ppm. Image to infer. Defaults to airliner.ppm.
	--reference=reference.txt or /absolute/path/to/reference.txt. Reference labels file. Defaults to
	reference_labels.txt.
	--ranges=ranges.txt or /absolute/path/to/ranges.txt. Specify custom per tensor dynamic range for the
	network. Defaults to mobilenet_last_dynamic_range.txt.
	--write_tensors. Option to generate file containing network tensors name. By default writes network_tensors.txt.
	To provide user defined file name use additional option --network_tensors_file. See --network_tensors_file option
	usage for more detail.
	--network_tensors_file=network_tensors.txt or /absolute/path/to/network_tensors.txt. This option
	needs to be used with --write_tensors option. Specify file name (will write to current execution
	directory) or absolute path to file name to write network tensor names file. Dynamic range
	corresponding to each network tensor is required to run the sample. Defaults to network_tensors.txt.
	--data=/path/to/data/dir. Specify data directory to search for above files in case absolute paths to
	files are not provided. Defaults to data/samples/int8_api/ or data/int8_api/.
	--useDLACore=N. Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1,
	where n is the number of DLA engines on the platform.
	--topBottomK=K. Specify how many Top-K results shall be output. Both the Top-K and the Bottom-K predictions
	will be printed with their output activation values. Defaults to 5 (for Top-5 and Bottom-5 results).
	--fp32. Run inference at FP32 precision on GPU. Cannot be combined with --useDLACore=N (N>=0).
	Defaults to running inference at INT8 precision (--fp32 not set).
	--safeGpuInt8. Run inference in safe mode on GPU at INT8. Cannot be combined with --useDLACore=N (N>=0)
	and/or --fp32. Defaults to running inference in unsafe mode (--safeGpuInt8 not set).
	--verbose. Outputs per tensor dynamic range and layer precision info for the network.
```


# Additional resources
See `README.md` of the original `sampleINT8API` [here](https://github.com/NVIDIA/TensorRT/blob/release/6.0/samples/opensource/sampleINT8API/README.md).


# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

**March 2020**: This `README.md` was modified for `sampleMobileNetINT8API`, showing inference for a pre-quantized MobileNetV1.

**March 2019** (based on the original `sampleINT8API`): This `README.md` file was recreated, updated and reviewed.


# Known issues

It is not recommended to use the pre-quantized MobileNetV1 model at fp16 precision, due to potential accumulation overflows of the integer weight values used in the Convolution nodes.
