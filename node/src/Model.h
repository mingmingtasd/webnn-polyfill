#ifndef __Model_H__
#define __Model_H__

#include "Base.h"

class Model : public Napi::ObjectWrap<Model> {
public:
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);
  static Napi::FunctionReference constructor;

  explicit Model(const Napi::CallbackInfo &info);
  ~Model();
  void SetModel(WebnnModel);
  WebnnModel GetModel();

  Napi::Value Compile(const Napi::CallbackInfo &info);
  void SetWebnnCompilation(WebnnCompilation);
  std::vector<std::string> &GetOutputName();

private:
  WebnnModel model_;
  WebnnCompilation compilation_;
  std::vector<std::string> output_name_;
};

#endif // __Model_H__
