/*
 *  Copyright (c) 2010 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "late_binding_symbol_table.h"

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

// #include "base/files/file_path.h"
#include "common/Log.h"
#if defined(__APPLE__)
#include "base/mac/bundle_locations.h"
#include "base/mac/foundation_util.h"
#endif
// #include "base/path_service.h"

namespace dawn_native {

inline static const char* GetDllError() {
#if defined(__linux__) || defined(__APPLE__)
  char* err = dlerror();
  if (err) {
    return err;
  }
#endif
  return "No error";
}

DllHandle InternalLoadDll(const char dll_name[]) {
  DllHandle handle = nullptr;
#if defined(__linux__)
  // base::FilePath module_path;
  // if (!base::PathService::Get(base::DIR_MODULE, &module_path)) {
  //   LOG(ERROR) << "Can't get module path";
  //   return nullptr;
  // }

  // base::FilePath dll_path = module_path.Append(dll_name);
  // handle = dlopen(dll_name.MaybeAsASCII().c_str(), RTLD_NOW);
  handle = dlopen(dll_name, RTLD_NOW);
#elif defined(__APPLE__)
  // base::FilePath base_dir;
  // if (base::mac::AmIBundled()) {
  //   base_dir = base::mac::FrameworkBundlePath().Append("Libraries");
  // } else {
  //   if (!base::PathService::Get(base::FILE_EXE, &base_dir)) {
  //     LOG(ERROR) << "PathService::Get failed.";
  //     return nullptr;
  //   }
  //   base_dir = base_dir.DirName();
  // }
  // base::FilePath dll_path = base_dir.Append(dll_name);
  handle = dlopen(dll_name.MaybeAsASCII().c_str(), RTLD_NOW);
#elif defined(_WIN32) || defined(_WIN64)
  handle = LoadLibraryA(dll_name);
#endif
  if (handle == kInvalidDllHandle) {
    dawn::ErrorLog() << "Can't load " << dll_name << " : " << GetDllError();
  }
  return handle;
}

void InternalUnloadDll(DllHandle handle) {
#if !defined(ADDRESS_SANITIZER)
#if defined(__linux__) || defined(__APPLE__)
  if (dlclose(handle) != 0) {
    dawn::ErrorLog() << GetDllError();
  }
#elif defined(_WIN32) || defined(_WIN64)
  FreeLibrary(static_cast<HMODULE>(handle));
#endif
#endif  // !defined(ADDRESS_SANITIZER)
}

static bool LoadSymbol(DllHandle handle,
                       const char* symbol_name,
                       void** symbol) {
#if defined(__linux__) || defined(__APPLE__)
  *symbol = dlsym(handle, symbol_name);
  char* err = dlerror();
  if (err) {
    dawn::ErrorLog() << "Error loading symbol " << symbol_name << " : " << err;
    return false;
  } else if (!*symbol) {
    dawn::ErrorLog() << "Symbol " << symbol_name << " is NULL";
    return false;
  }
#elif defined(_WIN32) || defined(_WIN64)
  *symbol = reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(handle), symbol_name));
#endif
  return true;
}

// This routine MUST assign SOME value for every symbol, even if that value is
// NULL, or else some symbols may be left with uninitialized data that the
// caller may later interpret as a valid address.
bool InternalLoadSymbols(DllHandle handle,
                         int num_symbols,
                         const char* const symbol_names[],
                         void* symbols[]) {
  // Clear any old errors.
  GetDllError();
  for (int i = 0; i < num_symbols; ++i) {
    if (!LoadSymbol(handle, symbol_names[i], &symbols[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace dawn_native
