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
##### 1. (Optional) Sync dependencies
```sh
> npm run build sync
```

**Modify config file for following commands 2 and 3, if you don't want to use default config file:**
Please refer to "Configure config json file" part of [./build_script/README.md](./build_script/README.md)

##### 2. Only build with current code
* Build with enabling the compilation of Dawn's Null backend and default arguments from configurations in [./build_script/bot_config.json](./build_script/bot_config.json)
```sh
> npm run build
```
* Or build with enabling the compilation of OpenVINO Inference Engine backend (ie) or DirectML backend (dml) and default arguments from configurations in ./build_script/bot_config.json
```sh
> npm run build -- --backend=[ie|dml]
```
* Or build with enabling the compilation of Dawn's Null backend and default arguments from configurations in given config json file
```sh
> npm run build -- ---config=<path>
```
* Or build with enabling the compilation of OpenVINO Inference Engine backend (ie) or DirectML backend (dml) and arguments from configurations in given config json file
```sh
> npm run build -- --backend=[ie|dml] --config=<path>
```

##### 3. All in one command to sync dependencies, fetch and build with latest code
Commands like above 3.
```sh
> npm run all
```
```sh
> npm run all -- --backend=[ie|dml]
```
```sh
> npm run all -- ---config=<path>
```
```sh
> npm run all -- --backend=[ie|dml] --config=<path>
```

### Lint
```sh
> npm run linter
```
Currnet this command only runs ESLint checking node scripts under [./build_script](./build_script) folder, will enchance later.  

## Test
For OpenVINO build, please [set the environment variables](https://docs.openvinotoolkit.org/2021.1/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables) first.
```sh
> out/Default/add
```
