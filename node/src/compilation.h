#ifndef __Compilation_H__
#define __Compilation_H__

#include "Base.h"

template <typename T>
T* ParseOperand(Napi::Object item, std::string& name) {
  Napi::Array property_names = item.GetPropertyNames();
  // assert(property_names.Length() == 1);
  for (size_t j = 0; j < property_names.Length(); ++j) {
    name = property_names.Get(j).As<Napi::String>().Utf8Value();
  }
  
  Napi::Object obj = item.Get(name).As<Napi::Object>();
  // The Buffer can't be set with DescriptorDecoder
  Napi::TypedArray array = obj.Get("buffer").As<Napi::TypedArray>();
  Napi::ArrayBuffer buffer = array.ArrayBuffer();
  T* operand = new T();
  operand->buffer = reinterpret_cast<void*>(buffer.Data());
  operand->size = buffer.ByteLength();
  return operand;
}

class Compilation : public Napi::ObjectWrap<Compilation> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  explicit Compilation(const Napi::CallbackInfo &info);
  ~Compilation();
  void SetCompilation(WNNCompilation);
  WNNCompilation GetCompilation();
  void FreeUnusedData();

  Napi::Value Compute(const Napi::CallbackInfo &info);

private:
  WNNCompilation compilation_;
  // The WNNInput and WNNOutput struct need to be kept until compute.
  std::vector<WNNInput*> inputs_;
  std::vector<WNNOutput*> outputs_;
};

#endif // __Compilation_H__
