#ifndef __Compilation_H__
#define __Compilation_H__

#include "Base.h"

template <typename T>
T ParseOperand(Napi::Object item, const std::string& name) {
  Napi::Object obj = item.Get(name).As<Napi::Object>();
  // The Buffer can't be set with DescriptorDecoder
  Napi::TypedArray array = obj.Get("buffer").As<Napi::TypedArray>();
  Napi::ArrayBuffer buffer = array.ArrayBuffer();
  T operand;
  operand.buffer = reinterpret_cast<void*>(buffer.Data());
  operand.size = buffer.ByteLength();
  return operand;
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