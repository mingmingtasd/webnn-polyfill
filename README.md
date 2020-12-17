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

## Generate projects
Build with OpenVINO backend
```sh
> gn gen out/Default --args="dawn_enable_ie = true"
```

Build with DirectML backend
```sh
> gn gen out/Default --args="dawn_enable_dml = true"
```

## Build
```sh
> ninja -C out/Default
```

## Test
For OpenVINO build, please [set the environment variables](https://docs.openvinotoolkit.org/2021.1/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables) first.
```sh
> out/Default/add
```
