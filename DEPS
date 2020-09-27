use_relative_paths = True

gclient_gn_args_file = 'build/config/gclient_args.gni'
gclient_gn_args = [
  'mac_xcode_version',
]

vars = {
  'chromium_git': 'https://chromium.googlesource.com',
  'dawn_git': 'https://dawn.googlesource.com',
  'github_git': 'https://github.com',
  'swiftshader_git': 'https://swiftshader.googlesource.com',

  'dawn_standalone': True,

  # This can be overridden, e.g. with custom_vars, to download a nonstandard
  # Xcode version in build/mac_toolchain.py instead of downloading the
  # prebuilt pinned revision.
  'mac_xcode_version': 'default',
}

deps = {
  # Dependencies required to use GN/Clang in standalone
  'build': {
    'url': '{chromium_git}/chromium/src/build@c10077be589bfb5151ebcd923928898a89ab33b8',
    'condition': 'dawn_standalone',
  },
  'buildtools': {
    'url': '{chromium_git}/chromium/src/buildtools@ff93f3ea1a7f033d3caf8f60ec1937cc71351419',
    'condition': 'dawn_standalone',
  },
  'tools/clang': {
    'url': '{chromium_git}/chromium/src/tools/clang@fcef86e30a0ab061b982b5c9d91bb060df8f5269',
    'condition': 'dawn_standalone',
  },
  'tools/clang/dsymutil': {
    'packages': [
      {
        'package': 'chromium/llvm-build-tools/dsymutil',
        'version': 'M56jPzDv1620Rnm__jTMYS62Zi8rxHVq7yw0qeBFEgkC',
      }
    ],
    'condition': 'checkout_mac or checkout_ios',
    'dep_type': 'cipd',
  },

  # Testing, GTest and GMock
  'testing': {
    'url': '{chromium_git}/chromium/src/testing@e5ced5141379ee8ae28b4f93d3c02df039d2b052',
    'condition': 'dawn_standalone',
  },
  'third_party/googletest': {
    'url': '{chromium_git}/external/github.com/google/googletest@a09ea700d32bab83325aff9ff34d0582e50e3997',
    'condition': 'dawn_standalone',
  },

  # Jinja2 and MarkupSafe for the code generator
  'third_party/jinja2': {
    'url': '{chromium_git}/chromium/src/third_party/jinja2@b41863e42637544c2941b574c7877d3e1f663e25',
    'condition': 'dawn_standalone',
  },
  'third_party/markupsafe': {
    'url': '{chromium_git}/chromium/src/third_party/markupsafe@8f45f5cfa0009d2a70589bcda0349b8cb2b72783',
    'condition': 'dawn_standalone',
  },
}

hooks = [
  # Pull the compilers and system libraries for hermetic builds
  {
    'name': 'sysroot_x86',
    'pattern': '.',
    'condition': 'checkout_linux and ((checkout_x86 or checkout_x64) and dawn_standalone)',
    'action': ['python', 'build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x86'],
  },
  {
    'name': 'sysroot_x64',
    'pattern': '.',
    'condition': 'checkout_linux and (checkout_x64 and dawn_standalone)',
    'action': ['python', 'build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x64'],
  },
  {
    # Update the Windows toolchain if necessary. Must run before 'clang' below.
    'name': 'win_toolchain',
    'pattern': '.',
    'condition': 'checkout_win and dawn_standalone',
    'action': ['python', 'build/vs_toolchain.py', 'update', '--force'],
  },
  {
    # Note: On Win, this should run after win_toolchain, as it may use it.
    'name': 'clang',
    'pattern': '.',
    'action': ['python', 'tools/clang/scripts/update.py'],
    'condition': 'dawn_standalone',
  },
  {
    # Pull rc binaries using checked-in hashes.
    'name': 'rc_win',
    'pattern': '.',
    'condition': 'checkout_win and (host_os == "win" and dawn_standalone)',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--no_auth',
                '--bucket', 'chromium-browser-clang/rc',
                '-s', 'build/toolchain/win/rc/win/rc.exe.sha1',
    ],
  },
  # Update build/util/LASTCHANGE.
  {
    'name': 'lastchange',
    'pattern': '.',
    'condition': 'dawn_standalone',
    'action': ['python', 'build/util/lastchange.py',
               '-o', 'build/util/LASTCHANGE'],
  },
]

recursedeps = [
  # buildtools provides clang_format, libc++, and libc++abi
  'buildtools',
]
