#ifndef __UTILS_H__
#define __UTILS_H__

#define NAPI_EXPERIMENTAL
#include <napi.h>
#include <unordered_map>

inline char* getNAPIStringCopy(const Napi::Value& value) {
  std::string utf8 = value.ToString().Utf8Value();
  int len = utf8.length() + 1; // +1 NULL
  char *str = new char[len];
  strncpy(str, utf8.c_str(), len);
  return str;
};

template<typename T> inline T* getTypedArrayData(const Napi::Value& value, size_t* len = nullptr) {
  T* data = nullptr;
  if (len) *len = 0;
  if (!value.IsTypedArray()) {
    Napi::Env env = value.Env();
    Napi::Error::New(env, "Argument must be a 'ArrayBufferView'").ThrowAsJavaScriptException();
    return data;
  }
  Napi::TypedArray arr = value.As<Napi::TypedArray>();
  Napi::ArrayBuffer buffer = arr.ArrayBuffer();
  if (len) *len = arr.ByteLength();
  data = reinterpret_cast<T*>(reinterpret_cast<uint64_t>(buffer.Data()) + arr.ByteOffset());
  return data;
};

static std::unordered_map<std::string, WebnnOperandType> s_operand_type_map = {
  { "float32", WebnnOperandType_Float32 },
  { "float16", WebnnOperandType_Float16 },
  { "int32", WebnnOperandType_Int32 },
  { "uint32", WebnnOperandType_Uint32 },
};

inline WebnnOperandType getOperandType(const Napi::Value& value) {
  if (!value.IsString()) {
    Napi::Env env = value.Env();
    Napi::Error::New(env, "Argument must be a 'String'").ThrowAsJavaScriptException();
    return WebnnOperandType_Force32;
  }
  return s_operand_type_map[value.As<Napi::String>().Utf8Value()];
};

#endif // __UTILS_H__
