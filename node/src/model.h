#ifndef __Model_H__
#define __Model_H__

#include "Base.h"

class Model : public Napi::ObjectWrap<Model> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  explicit Model(const Napi::CallbackInfo &info);
  ~Model();
  void SetModel(WNNModel);
  WNNModel GetModel();

  Napi::Value Compile(const Napi::CallbackInfo &info);
  void SetWNNCompilation(WNNCompilation);
  std::vector<std::string> &GetOutputName();

private:
  WNNModel model_;
  WNNCompilation compilation_;
  std::vector<std::string> output_name_;
};

#endif // __Model_H__
