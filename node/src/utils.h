#ifndef __UTILS_H__
#define __UTILS_H__

#define NAPI_EXPERIMENTAL
#include <napi.h>

inline char* getNAPIStringCopy(const Napi::Value& value) {
  std::string utf8 = value.ToString().Utf8Value();
  int len = utf8.length() + 1; // +1 NULL
  char *str = new char[len];
  strncpy(str, utf8.c_str(), len);
  return str;
};

inline void nextJSProcessTick(Napi::Env& env) {
  Napi::Object process = env.Global().Get("process").As<Napi::Object>();
  Napi::Function nextTick = process.Get("nextTick").As<Napi::Function>();
  nextTick.Call(env.Global(), {});
};

inline void* getTypedArrayData(const Napi::Value& value, size_t* len = nullptr) {
  if (len) *len = 0;
  if (!value.IsTypedArray()) {
    Napi::Env env = value.Env();
    Napi::Error::New(env, "Argument must be a 'ArrayBufferView'").ThrowAsJavaScriptException();
    return nullptr;
  }
  Napi::TypedArray arr = value.As<Napi::TypedArray>();
  Napi::ArrayBuffer buffer = arr.ArrayBuffer();
  if (len) *len = arr.ByteLength();
  return buffer.Data();
};

#endif // __UTILS_H__
