#ifndef __Compilation_H__
#define __Compilation_H__

#include <iostream>
#include <map>

#include "Base.h"

template <typename T>
std::map<std::string, T> GetNamedOperands(Napi::Object obj) {
    std::map<std::string, T> namedOperands;
    Napi::Array property_names = obj.GetPropertyNames();
    for (size_t i = 0; i < property_names.Length(); ++i) {
        std::string name = property_names.Get(i).As<Napi::String>().Utf8Value();
        Napi::Object item = obj.Get(name).As<Napi::Object>();
        Napi::TypedArray array = item.Get("buffer").As<Napi::TypedArray>();
        Napi::ArrayBuffer buffer = array.ArrayBuffer();
        T operand;
        operand.buffer = reinterpret_cast<void*>(buffer.Data());
        operand.size = buffer.ByteLength();
        namedOperands[name] = operand;
    }
    return namedOperands;
}

class ComputeAsyncWorker;
class Compilation : public Napi::ObjectWrap<Compilation> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  explicit Compilation(const Napi::CallbackInfo &info);
  ~Compilation();
  void SetCompilation(WebnnCompilation);
  WebnnCompilation GetCompilation();

  Napi::Value Compute(const Napi::CallbackInfo &info);

private:
  WebnnCompilation compilation_;
  Napi::ObjectReference model_object_;
  ComputeAsyncWorker* compute_worker_;
};

#endif // __Compilation_H__
