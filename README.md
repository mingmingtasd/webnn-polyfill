# webnn-native

## Build

Create a `.gclient` file with the following content:
```
solutions = [
  { "name"        : ".",
    "url"         : "https://github.com/huningxin/webnn-native.git",
    "deps_file"   : "DEPS",
    "managed"     : False,
  },
]
```

Sync the dependencies.
```sh
> gclient sync
```

Generate projects.
```sh
Use Release mode when running OpenVINO backend.
> gn gen out/Release --args="is_debug = false"
```

Build
```sh
> ninja -C out/Release
```

## Test

```sh
> out/Release/matmul
```
