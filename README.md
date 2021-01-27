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

## Node Commands
### Installation
```sh
> npm install
```
### Build
##### 1. Only sync dependencies
```sh
> npm run build sync
```

##### 2. Only fetch latest code
```sh
> npm run build pull
```

**Modify config file for following commands 3 and 4, if you don't want to use default config file:**
Please refer to "Configure config json file" part of [./build_script/README.md](./build_script/README.md)

##### 3. Only build with current code
* Build with enabling the compilation of Dawn's Null backend and default arguments from configurations in [./build_script/bot_config.json](./build_script/bot_config.json)
```sh
> npm run build build
```
* Or build with enabling the compilation of OpenVINO Inference Engine backend (ie) or DirectML backend (dml) and default arguments from configurations in ./build_script/bot_config.json
```sh
> npm run build build -- --backend=[ie|dml]
```
* Or build with enabling the compilation of Dawn's Null backend and default arguments from configurations in given config json file
```sh
> npm run build build -- ---config=<path>
```
* Or build with enabling the compilation of OpenVINO Inference Engine backend (ie) or DirectML backend (dml) and arguments from configurations in given config json file
```sh
> npm run build build -- --backend=[ie|dml] --config=<path>
```

##### 4. All in one command to sync dependencies, fetch and build with latest code
Commands like above 3.
```sh
> npm run build all
```
```sh
> npm run build all -- --backend=[ie|dml]
```
```sh
> npm run build all -- ---config=<path>
```
```sh
> npm run build all -- --backend=[ie|dml] --config=<path>
```

### Lint
```sh
> npm run linter
```

## Test
For OpenVINO build, please [set the environment variables](https://docs.openvinotoolkit.org/2021.1/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables) first.
```sh
> out/Default/add
```
