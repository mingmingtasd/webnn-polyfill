#include "Conv2d.h"

#include <iostream>
#include <unordered_map>

#include "../operand.h"

namespace op {

std::vector<int32_t> GetTypedArray(Napi::Object &obj, std::string name) {
  Napi::Array array = obj.Get(name).As<Napi::Array>();
  uint32_t len = array.Length();
  std::vector<int32_t> typed_array;
  for (uint32_t i = 0; i < len; i++) {
    typed_array.push_back(
        static_cast<Napi::Value>(array[i]).As<Napi::Number>().Int32Value());
  }
  return typed_array;
}

static std::unordered_map<std::string, uint32_t> OperandLayoutMap = {
    {"nchw", 0},
    {"nhwc", 1},
};

uint32_t OperandLayout(std::string name) { return OperandLayoutMap[name]; };

Conv2d::Conv2d(const Napi::CallbackInfo &info) : Node(info) {
  options_ = {};
  if (info.Length() == 2)
    return;

  Napi::Object obj = info[2].As<Napi::Object>();
  Napi::Array property_names = obj.GetPropertyNames();
  for (size_t j = 0; j < property_names.Length(); ++j) {
    std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
    if (name == "padding") {
      padding_ = GetTypedArray(obj, name);
      options_.padding = padding_.data();
      options_.paddingCount = padding_.size();
    } else if (name == "strides") {
      stride_ = GetTypedArray(obj, name);
      options_.strides = stride_.data();
      options_.stridesCount = stride_.size();
    } else if (name == "dilations") {
      dilations_ = GetTypedArray(obj, name);
      options_.dilations = dilations_.data();
      options_.dilationsCount = dilations_.size();
    } else if (name == "groups") {
      options_.groups = static_cast<Napi::Value>(obj.Get(name))
                            .As<Napi::Number>()
                            .Int32Value();
    } else if (name == "layout") {
      options_.layout = static_cast<WNNOperandLayout>(
          OperandLayout(obj.Get(name).As<Napi::String>().Utf8Value()));
    }
  }
}

WNNConv2dOptions *Conv2d::GetOptions() { return &options_; }

} // namespace op
