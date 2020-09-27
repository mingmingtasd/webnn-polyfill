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
> gn gen out/Release
```

Build
```sh
> ninja -C out/Release
```

## Test

```sh
> out/Release/matmul
```
