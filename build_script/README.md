# Description

Node script [bin/build_webnn](./bin/build_webnn) enables you to build [webnn-native](https://github.com/otcshare/webnn-native) easily.

### Configure config json file
Here's [bot_config.json](./bot_config.json) config file, its content is

```
{
  "target-os": "",
  "target-cpu": "",
  "gnArgs": {
    "is-clang": true,
    "is-component": true,
    "is-debug": false
  },
  "clean-build": true,
  "logging": {
    "level": "",
    "file": ""
  },
  "archive-server": {
    "host": "",
    "dir": "",
    "ssh-user": ""
  },
  "email-service":{
    "user": "",
    "host": "",
    "from": "",
    "to": ""
  }
}
```
- target-os: The desired operating system for the build. Defualt "", use current system OS for build. Values
  - ""
  - "linux"
  - "win"

- target-cpu: The desired cpu architecture for the build. Defualt "", use current system CPU architecture for build. values
  - ""
  - "x86"
  - "x64"
  - "arm"
  - "arm64"

- clean-build: Whether remove output directory. Default true, remove output directory before build.

Seetings in **"gnArgs"**: Specifies build arguments overrides, will save in args.gn file under output directory after running *gn gen --args="args"* command.
- is-clang: Set to true when compiling with the Clang compiler. Please set false and install [Visual Studio 2019 IDE](https://visualstudio.microsoft.com/downloads/) to build Node Addon on Windows platform.
- is-component: Component build. Setting to true compiles targets declared as "components" as shared libraries loaded dynamically. This speeds up development time. Default true for compilation of libc++ library. When false, components will be linked statically.
- is-debug: Debug build. Enabling official builds automatically sets is_debug to false.

Seetings in **"logging"**: Specifies directory for storing output packages on a dedicated server.
- level: Logging level. Default "debug". Values
  - "info"
  - "warn"
  - "error"
  - "debug"
- file: Save logging message into given file. Default "", save logging message into /tmp/webnn_\<target-os\>\_\<target-cpu\>\_\<backend\>\_\<YYYY-MM-DD\>.log on Linux platform or C:\Users\\<username\>\AppData\Local\Temp\\<target-os\>\_\<target-cpu\>\_\<backend\>\_\<YYYY-MM-DD\>.log on Windows platform.

Seetings in **"archive-server"**: Specifies directory for storing output packages and logging files on a dedicated server.
- host: IP or domain name.
- dir: Path of directory for storing output packages, likes /path .
- ssh-user: A account you want to access to host.

Seetings in **"email-service"**: Specifies directory for storing output packages and logging files on a dedicated server.
- user: User name for logging into SMTP Serve.
- host: SMTP host.
- from: Sender of the format (address or name \<address\> or "name" \<address\>).
- to: Recipients (same format as above), multiple recipients are separated by a comma.

### Install
```sh
> npm install.
```

### Use Node Scripts
```sh
$ ./bin/build_webnn --help

Usage: build_webnn [options] [command]

Options:
  -V, --version    output the version number
  -h, --help       display help for command

Commands:
  sync
  pull
  build [options]
  package
  upload
  notify
  all [options]
  help [command]   display help for command
```

Options for *./bin/build_webnn build* or *./bin/build_webnn all* commands are same, likes:
```
$ ./bin/build_webnn build --help
Usage: build_webnn build [options]

Options:
  -b, --backend <backend>  Build with target backend (default: "null")
  -c, --conf <config>      Configuration file (default: "bot_config.json")
  -h, --help               display help for command
```

### Use Batch Wrapper Scripts
[build_webnn](./build_webnn) script is for build on Linux platform
[build_webnn.bat](./build_webnn.bat) script is for build on Windows platform.

### Sync dependencies
```sh
> ./build_webnn sync
```
```sh
> build_webnn.bat sync
```

### Fetch code
```sh
> ./build_webnn pull
```
```sh
> build_webnn.bat pull
```

### Build
##### Build with enabling the compilation of Dawn's Null backend and default arguments from configurations in [./bot_config.json](./bot_config.json)
```sh
> ./build_webnn
```
```sh
> build_webnn.bat
```

##### Or build with enabling the compilation of OpenVINO Inference Engine backend (ie) or DirectML backend (dml) and default arguments from configurations in ./bot_config.json
```sh
> ./build_webnn --backend=[ie|dml]
```
```sh
> build_webnn.bat "--backend=[ie|dml]"
```
##### Or build with enabling the compilation of Dawn's Null backend and default arguments from configurations in given config json file
```sh
> ./build_webnn --config=<path>
```
```sh
> build_webnn.bat "--config=<path>"
```

##### Or build with enabling the compilation of OpenVINO Inference Engine backend (ie) or DirectML backend (dml) and arguments from configurations in given config json file
```sh
> ./build_webnn --backend=[ie|dml] --config=<path>
```
```sh
> build_webnn.bat "--backend=[ie|dml] --config=<path>"
```

### Package
```sh
> ./build_webnn package
```
```sh
> build_webnn.bat package
```

### Upload package and log file to Store Server
```sh
> ./build_webnn upload
```
```sh
> build_webnn.bat upload
```

### Notify by sending Email
```sh
> ./build_webnn notify
```
```sh
> build_webnn.bat notify
```

### All above commands in one command
The usage of backend and config arguments are same with above **Build** command

```sh
> ./build_webnn all --backend=[ie|dml] --config=<path>
```
```sh
> build_webnn.bat all "--backend=[ie|dml] --config=<path>"
```

## BKMs
### To support to upload via SSH
1. On your client, follow [Github SSH page](https://help.github.com/articles/connecting-to-github-with-ssh/) to generate SSH keys and add to ssh-agent. (If you've done that, ignore)
2. On upload server, config [Authorized keys](https://www.ssh.com/ssh/authorized_keys/) with above client public keys.

### Coding style
We're following the [Google JavaScript coding style](https://google.github.io/styleguide/jsguide.html) in general. And there is pre-commit checking `tools/linter.js` to ensure styling before commit code.