{
  "variables": {
    "platform": "<(OS)",
    "build": "<@(module_root_dir)/build",
    "release": "<(build)/Release",
    "webnn": "<@(module_root_dir)/../",
  },
  "conditions": [
    [ "platform == 'win'",   { "variables": { "platform": "win" } } ],
    [ "platform == 'linux'", { "variables": {
      "platform": "linux",
      "rel_release": "<!(echo <(release) | sed 's/.*generated/generated/')",
      } } ],
    [ "platform == 'mac'",   { "variables": { "platform": "darwin" } } ]
  ],
  "make_global_settings": [
    ['CXX','/usr/bin/clang++'],
    ['LINK','/usr/bin/clang++'],
  ],
  "targets": [
    {
      "target_name": "action_after_build",
      "type": "none",
      "conditions": []
    },
    {
      "conditions": [
        [
          "OS=='win'",
          {
            "sources": [
              "src/*.cpp","src/ops/*.cc"
            ],
            "target_name": "addon-win32",
            "cflags!": [
              "-fno-exceptions"
            ],
            "cflags_cc!": [
              "-fno-exceptions"
            ],
            "include_dirs": [
              "<!@(node -p \"require('node-addon-api').include\")",
              "<(webnn)/src/include",
              "<(webnn)/out/Shared/gen/src/include",
            ],
            "library_dirs": [
              "<(webnn)/out/Shared",
            ],
            "link_settings": {
              "libraries": [
                "-lwebnn_native.dll.lib",
                "-lwebnn_proc.dll.lib",
                #"-lwebnn_wire.dll.lib",
              ]
            },
            "defines": [
              "WIN32_LEAN_AND_MEAN",
              "NOMINMAX",
              "_UNICODE",
              "UNICODE",
              #"DAWN_ENABLE_BACKEND_D3D12",
              "DAWN_NATIVE_SHARED_LIBRARY",
              "DAWN_NATIVE_IMPLEMENTATION",
              "NAPI_CPP_EXCEPTIONS"
            ],
            "msvs_settings": {
              "VCCLCompilerTool": {
                "AdditionalOptions": ["/MP /EHsc"],
                "ExceptionHandling": 1
              },
              "VCLibrarianTool": {
                "AdditionalOptions" : []
              },
              "VCLinkerTool": {
                "AdditionalLibraryDirectories": [
                  "../@PROJECT_SOURCE_DIR@/lib/<(platform)/<(target_arch)",
                  "<(build)/"
                ]
              }
            }
          },
          "OS=='linux'",
          {
            "sources": [
              "<!@(node -p \"require('fs').readdirSync('./src').map(f=>'src/'+f).join(' ')\")",
               "<!@(node -p \"require('fs').readdirSync('./src/ops').map(f=>'src/ops/'+f).join(' ')\")"
            ],
            "target_name": "addon-linux",
            "defines": [
              #"DAWN_ENABLE_BACKEND_NULL",
              #"DAWN_ENABLE_BACKEND_VULKAN",
              "DAWN_NATIVE_SHARED_LIBRARY",
              "DAWN_NATIVE_IMPLEMENTATION",
              #"DAWN_WIRE_SHARED_LIBRARY",
              #"WGPU_SHARED_LIBRARY",
              "NAPI_CPP_EXCEPTIONS"
            ],
            "include_dirs": [
              "<!@(node -p \"require('node-addon-api').include\")",
              "<(webnn)/src/include",
              "<(webnn)/out/Shared/gen/src/include",
            ],
            "cflags": [
             "-std=c++14",
              "-fexceptions",
              "-Wno-switch",
              "-Wno-unused",
              "-Wno-uninitialized"
            ],
            "cflags_cc": [
             "-std=c++14",
              "-fexceptions",
              "-Wno-switch",
              "-Wno-unused",
              "-Wno-uninitialized"
            ],
            "library_dirs": [
              "<@(module_root_dir)/build",
              "<@(module_root_dir)/build/Release",
              "<(module_root_dir)/../../../build/<(platform)",
              "<(webnn)/out/Shared",
            ],
            "libraries": [
              "-Wl,-rpath,./<(rel_release)",
              "-lwebnn_native",
              "-lwebnn_proc",
              "-lXrandr",
              "-lXi",
              "-lX11",
              "-lXinerama",
              "-lXcursor",
              "-ldl",
              "-pthread"  
            ]
          }
        ]
      ]
    }
  ]
}
