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
    "level": "debug",
    "file": ""
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

### Help
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
  all [options]
  help [command]   display help for command
```

Options for *./bin/build_webnn build* or *./bin/build_webnn all* commands are same, likes:
```
$ ./bin/build_webnn build --help
Usage: build_webnn build [options]

Options:
  -b, --backend <backend>  Build with target backend (default: "null")
  -c, --conf <config>      Configuration file (default: "build_script/bot_config.json")
  -h, --help               display help for command
```

## BKMs
### To support to upload via SSH
1. On your client, follow [Github SSH page](https://help.github.com/articles/connecting-to-github-with-ssh/) to generate SSH keys and add to ssh-agent. (If you've done that, ignore)
2. On upload server, config [Authorized keys](https://www.ssh.com/ssh/authorized_keys/) with above client public keys.

### Coding style
We're following the [Google JavaScript coding style](https://google.github.io/styleguide/jsguide.html) in general. And there is pre-commit checking `tools/linter.js` to ensure styling before commit code.