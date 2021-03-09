#include "Transpose.h"

#include <iostream>

#include "../Operand.h"
#include "Conv2d.h"

namespace op {

Transpose::Transpose(const Napi::CallbackInfo &info) : Node(info) {
  options_ = {};
  if (info.Length() == 1)
    return;

  Napi::Object obj = info[1].As<Napi::Object>();
  Napi::Array property_names = obj.GetPropertyNames();
  for (size_t j = 0; j < property_names.Length(); ++j) {
    std::string name = property_names.Get(j).As<Napi::String>().Utf8Value();
    if (name == "permutation") {
      permutation_ = GetTypedArray(obj, name);
      options_.permutation = permutation_.data();
      options_.permutationCount = permutation_.size();
    }
  }
}

WebnnTransposeOptions *Transpose::GetOptions() { return &options_; }

} // namespace op
