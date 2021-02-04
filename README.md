# webnn-native

## Get the code
Create a `.gclient` file with the following content:
```
solutions = [
  { "name"        : ".",
    "url"         : "https://github.com/otcshare/webnn-native.git",
    "deps_file"   : "DEPS",
    "managed"     : False,
  },
]
```

Sync the dependencies.
```sh
> gclient sync
```
## Manually build step by step
### Setting up the build
Generate projects with OpenVINO backend
```sh
> gn gen out/Default --args="webnn_enable_openvino = true"
```

Or generate projects with DirectML backend
```sh
> gn gen out/Default --args="webnn_enable_dml = true"
```

Or generate projects with default null backend
```sh
> gn gen out/Default
```
### Build by ninja
```sh
> ninja -C out/Default
```

## Build with Batch Scripts under [./build_script](./build_script) folder
Please refer to [./build_script/README](./build_script/README.md).

## Test
For OpenVINO build, please [set the environment variables](https://docs.openvinotoolkit.org/2021.1/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables) first.
```sh
> <out_dir>/webnn_end2end_tests
```
or
```sh
> <out_dir>\webnn_end2end_tests.exe
```