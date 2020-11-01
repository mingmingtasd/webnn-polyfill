// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ie_model.h"

#include <gna/gna_config.hpp>
#include <string>
#include <utility>

#include "ngraph/ngraph.hpp"
#include "ngraph/node.hpp"
#include "utils.h"

namespace InferenceEngine {

using namespace ngraph;

namespace {

SizeVector ToVector(int32_t const *value, uint32_t count) {
  SizeVector data;
  data.reserve(count);
  for (int i = 0; i < count; ++i) {
    data.push_back(value[i]);
  }
  return data;
}

ie_operand_t *CreateOperand(std::string &name) {
  ie_operand_t *operand = new ie_operand_t();
  std::unique_ptr<char[]> node_name(new char[name.length() + 1]);
  operand->name = node_name.release();
  memcpy(operand->name, name.c_str(), name.length() + 1);
  return operand;
}
} // namespace

ie_operand_t *Model::AddConstant(ie_operand_descriptor_t const *desc,
                                 void const *value, size_t size) {
  SizeVector dims = ToVector(desc->dimensions, desc->dimensionsCount);
  // Generally, FP16 is preferable as it is most ubiquitous and performant
  // documented in
  // https://docs.openvinotoolkit.org/2021.1/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html.
  bool fp32_precision = true;
  Blob::Ptr blob;
  if (fp32_precision) {
    // GNA only accepts FP32 precision, cpu/gpu use FP32 currently.
    blob = make_shared_blob<float>({Precision::FP32, dims, Layout::ANY});
  } else {
    // MYRIAD only accepts FP16 precision.
    blob = make_shared_blob<int16_t>({Precision::FP16, dims, Layout::ANY});
  }
  blob->allocate();
  const float *src = reinterpret_cast<const float *>(value);
  std::shared_ptr<op::Constant> node;
  uint32_t result;
  if (fp32_precision) {
    float *dst = blob->buffer().as<float *>();
    CopyDataToBuffer<float>(dst, src, size);
    node = std::make_shared<op::Constant>(element::f32, Shape(dims), dst);
  } else {
    int16_t *dst = blob->buffer().as<int16_t *>();
    CopyDataToBuffer<int16_t>(dst, src, size);
    node = std::make_shared<op::Constant>(element::f16, Shape(dims), dst);
  }

  std::string node_name = node->get_name();
  name_node_map_[node_name] = node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t *Model::AddInput(ie_operand_descriptor_t const *desc) {
  SizeVector dims = ToVector(desc->dimensions, desc->dimensionsCount);
  auto input_node =
      std::make_shared<op::v0::Parameter>(element::f32, Shape(dims));
  ngraph_inputs_.push_back(input_node);

  std::string node_name = input_node->get_name();
  name_node_map_[node_name] = input_node->output(0);
  return CreateOperand(node_name);
}

void Model::AddOutput(ie_operand_t *operand) {
  auto node_name = std::string(operand->name);
  auto output_node = std::make_shared<op::Result>(name_node_map_[node_name]);
  ngraph_outputs_.push_back(output_node);

  return;
}

ie_operand_t *Model::AddMatMul(ie_operand_t *a, ie_operand_t *b) {
  auto primary_node = name_node_map_[a->name];
  auto secondary_node = name_node_map_[b->name];
  auto matmul_node = std::make_shared<op::v0::MatMul>(
      primary_node, secondary_node, false, false);

  std::string node_name = matmul_node->get_name();
  name_node_map_[node_name] = matmul_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t *Model::AddBinary(ie_binary_type type, ie_operand_t *a,
                               ie_operand_t *b) {
  auto primary_node = name_node_map_[a->name];
  auto secondary_node = name_node_map_[b->name];
  std::shared_ptr<ngraph::Node> binary_node;
  switch (type) {
  case ie_binary_type::ADD:
    binary_node = std::make_shared<op::v1::Add>(primary_node, secondary_node);
    break;
  case ie_binary_type::MUL:
    binary_node =
        std::make_shared<op::v1::Multiply>(primary_node, secondary_node);
    break;
  default:
    assert(0);
  }

  std::string node_name = binary_node->get_name();
  name_node_map_[node_name] = binary_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t *Model::AddConv2d(ie_operand_t *input, ie_operand_t *filter,
                               ie_conv2d_options_t *options) {
  CoordinateDiff pad_begin = {options->padding[0], options->padding[2]};
  CoordinateDiff pad_end = {options->padding[1], options->padding[3]};
  Strides strides = {static_cast<size_t>(options->strides[0]),
                     static_cast<size_t>(options->strides[1])};
  Strides dilations = {static_cast<size_t>(options->dilations[0]),
                       static_cast<size_t>(options->dilations[1])};

  auto input_node = name_node_map_[input->name];
  auto filter_node = name_node_map_[filter->name];
  auto conv2d_node = std::make_shared<op::v1::Convolution>(
      input_node, filter_node, strides, pad_begin, pad_end, dilations);

  std::string node_name = conv2d_node->get_name();
  name_node_map_[node_name] = conv2d_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t *Model::AddPool2d(ie_pool_type type, ie_operand_t *input,
                               ie_pool2d_options_t *options) {
  Shape window_dimensions = {static_cast<size_t>(options->windowDimensions[0]),
                             static_cast<size_t>(options->windowDimensions[1])};
  Shape pad_begin = {static_cast<size_t>(options->padding[0]),
                     static_cast<size_t>(options->padding[2])};
  Shape pad_end = {static_cast<size_t>(options->padding[1]),
                   static_cast<size_t>(options->padding[3])};
  Strides strides = {static_cast<size_t>(options->strides[0]),
                     static_cast<size_t>(options->strides[1])};
  Shape dilations = {static_cast<size_t>(options->dilations[0]),
                     static_cast<size_t>(options->dilations[1])};

  auto input_node = name_node_map_[input->name];
  std::shared_ptr<ngraph::Node> pool2d_node;
  switch (type) {
  case ie_pool_type::AVERAGE_POOL:
    pool2d_node = std::make_shared<op::v1::AvgPool>(
        input_node, strides, pad_begin, pad_end, window_dimensions, true,
        op::RoundingType::FLOOR, op::PadType::EXPLICIT);
    break;
  case ie_pool_type::MAX_POOL:
    pool2d_node = std::make_shared<op::v1::MaxPool>(
        input_node, strides, pad_begin, pad_end, window_dimensions,
        op::RoundingType::FLOOR, op::PadType::EXPLICIT);
    break;
  default:
    assert(0);
  }

  std::string node_name = pool2d_node->get_name();
  name_node_map_[node_name] = pool2d_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t *Model::AddRelu(ie_operand_t *input) {
  auto input_node = name_node_map_[input->name];
  auto relu_node = std::make_shared<op::v0::Relu>(input_node);

  std::string node_name = relu_node->get_name();
  name_node_map_[node_name] = relu_node->output(0);
  return CreateOperand(node_name);
}

ie_operand_t *Model::AddReshape(ie_operand_t *input, int32_t const *new_shape,
                                uint32_t new_shape_count) {
  auto input_node = name_node_map_[input->name];
  SizeVector shape = ToVector(new_shape, new_shape_count);
  auto target_shape_node =
      std::make_shared<op::Constant>(element::i64, Shape{shape.size()}, shape);
  auto reshape_node = std::make_shared<op::v1::Reshape>(
      input_node, target_shape_node->output(0), true);

  std::string node_name = reshape_node->get_name();
  name_node_map_[node_name] = reshape_node->output(0);
  return CreateOperand(node_name);
}

void Model::Finish() {
  auto ngraph_function =
      std::make_shared<Function>(ngraph_outputs_, ngraph_inputs_);
  network_ = std::make_unique<CNNNetwork>(ngraph_function);
  InputsDataMap input_info(network_->getInputsInfo());
  for (auto itr : input_info) {
    itr.second->setPrecision(Precision::FP32);
  }
  OutputsDataMap output_info(network_->getOutputsInfo());
  for (auto itr : output_info) {
    itr.second->setPrecision(Precision::FP32);
  }
}

} // namespace InferenceEngine
