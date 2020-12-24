// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef WEBNN_NATIVE_OPS_TRANSPOSE_H_
#define WEBNN_NATIVE_OPS_TRANSPOSE_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native {

namespace op {

class Transpose final : public OperandBase {
public:
  Transpose(OperandBase *input, TransposeOptions const *options)
      : OperandBase({input}) {
    if (options) {
      permutation_.assign(options->permutation,
                          options->permutation + options->permutationCount);
      options_.permutation = permutation_.data();
      options_.permutationCount = permutation_.size();
    }
  }
  ~Transpose() override = default;

  MaybeError AddToModel(ModelBase *model) const override {
    return model->AddTranspose(this);
  }

  TransposeOptions const *Options() const { return &options_; }

private:
  TransposeOptions options_;
  std::vector<int32_t> permutation_;
};

} // namespace op

} // namespace dawn_native

#endif // WEBNN_NATIVE_OPS_TRANSPOSE_H_
