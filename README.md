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
## Build manually step by step
### Setting up the build
Generate projects with OpenVINO backend
```sh
> gn gen out/Default --args="dawn_enable_ie = true"
```

Or generate projects with DirectML backend
```sh
> gn gen out/Default --args="dawn_enable_dml = true"
```

Or generate projects with default null backend
```sh
> gn gen out/Default
```
### Build by ninja
```sh
> ninja -C out/Default
```

## Build with Node scripts
### Installation
```sh
> npm install
```
### Fast build with latest code
```sh
> npm run build all
```

### Only gclient sync to latest code
```sh
> npm run build sync
```

### Only git pull to latest code
```sh
> npm run build pull
```

### Only build with current code
Defualt
```sh
> npm run build build
```
or with backen / config options
```sh
> npm run build build -- --backend=[ie|dml]
```
or
```sh
> npm run build build -- ---config=<path>
```
or
```sh
> npm run build build -- --backend=[ie|dml] --config=<path>
```

Please refer ./build_script/README.md to find details commands.

## Test
For OpenVINO build, please [set the environment variables](https://docs.openvinotoolkit.org/2021.1/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables) first.
```sh
> out/Default/add
```
