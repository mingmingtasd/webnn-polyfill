#include "pool2d.h"

#include <iostream>
#include <unordered_map>

#include "../operand.h"
#include "conv2d.h"

namespace op {

Pool2d::Pool2d(const Napi::CallbackInfo &info) : Node(info) {
  options_ = {};
  if (info.Length() == 1)
    return;

  Napi::Object obj = info[1].As<Napi::Object>();
  Napi::Array property_names = obj.GetPropertyNames();
  for (size_t j = 0; j < property_names.Length(); ++j) {
    std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
    if (name == "windowDimensions") {
      window_dimensions_ = GetTypedArray(obj, name);
      options_.windowDimensions = window_dimensions_.data();
      options_.windowDimensionsCount = window_dimensions_.size();
    } else if (name == "padding") {
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
    } else if (name == "layout") {
      options_.layout = static_cast<WNNOperandLayout>(
          OperandLayout(obj.Get(name).As<Napi::String>().Utf8Value()));
    }
  }
}

WNNPool2dOptions *Pool2d::GetOptions() { return &options_; }

} // namespace op
