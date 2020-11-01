// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef IE_MODEL_H
#define IE_MODEL_H

#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_nn_c_api.h"
#include "ngraph/node_output.hpp"
#include "ngraph/op/parameter.hpp"
#include "utils.h"

namespace InferenceEngine {

class Model {
public:
  Model() = default;
  ~Model() = default;

  ie_operand_t *AddConstant(ie_operand_descriptor_t const *desc,
                            void const *value, size_t size);
  ie_operand_t *AddInput(ie_operand_descriptor_t const *desc);
  void AddOutput(ie_operand_t *operand);
  ie_operand_t *AddMatMul(ie_operand_t *a, ie_operand_t *b);
  ie_operand_t *AddBinary(ie_binary_type type, ie_operand_t *a,
                          ie_operand_t *b);
  ie_operand_t *AddConv2d(ie_operand_t *input, ie_operand_t *filter,
                          ie_conv2d_options_t *options);
  ie_operand_t *AddPool2d(ie_pool_type type, ie_operand_t *input,
                          ie_pool2d_options_t *options);
  ie_operand_t *AddRelu(ie_operand_t *input);
  void Finish();

private:
  friend class Compilation;
  std::map<std::string, ngraph::Output<ngraph::Node>> name_node_map_;
  std::vector<std::shared_ptr<ngraph::op::v0::Parameter>> ngraph_inputs_;
  std::vector<std::shared_ptr<ngraph::op::v0::Result>> ngraph_outputs_;
  std::unique_ptr<CNNNetwork> network_;

  DISALLOW_COPY_AND_ASSIGN(Model);
};

} // namespace InferenceEngine

#endif // IE_MODEL_H
