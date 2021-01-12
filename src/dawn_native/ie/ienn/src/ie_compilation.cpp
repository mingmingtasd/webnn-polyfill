// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ie_compilation.h"

#include <gna/gna_config.hpp>
#include <string>
#include <utility>

#include "ngraph/ngraph.hpp"
#include "ngraph/node.hpp"
#include "utils.h"

namespace InferenceEngine {

// TODO(Junwei): GNA device only be opened for one instance of
// ExecutableNetwork, there will be memory leak for these static objects.
static std::unique_ptr<Core> s_ie_core = nullptr;
static std::unique_ptr<ExecutableNetwork> s_gna_execution = nullptr;
static std::unique_ptr<InferRequest> s_gna_infer_request = nullptr;

Compilation::Compilation(std::shared_ptr<Model> model)
    : preference_(PREFER_FAST_SINGLE_ANSWER) {
  std::string device_name;
  if (preference_ == prefer_t::PREFER_FAST_SINGLE_ANSWER) {
    device_name = "CPU";
  } else if (preference_ == prefer_t::PREFER_SUSTAINED_SPEED) {
    device_name = "GPU";
  } else if (preference_ == prefer_t::PREFER_LOW_POWER) {
    device_name = "MYRIAD";
  } else if (preference_ == prefer_t::PREFER_ULTRA_LOW_POWER) {
    device_name = "GNA";
    // Release in squence to avoid crash. Close GNA device befere re-open,
    s_gna_infer_request.reset(nullptr);
    s_gna_execution.reset(nullptr);
    s_ie_core.reset(nullptr);
  }
  std::unique_ptr<InferRequest> infer_request;
  std::unique_ptr<Core> ie_core;
  std::unique_ptr<ExecutableNetwork> execution;
  ie_core.reset(new Core());
  std::map<std::string, std::string> plugin_Config = {};
  if (preference_ == prefer_t::PREFER_ULTRA_LOW_POWER) {
    // TODO(Junwei): The SCALE_FACTOR need to be set.
    plugin_Config[GNAConfigParams::KEY_GNA_DEVICE_MODE] = "GNA_AUTO";
    // Note that it is not always possible to use 8-bit weights due to GNA
    // hardware limitations. For example, convolutional layers always use
    // 16-bit weights (GNA harware verison 1 and 2). This limitation will be
    // removed in GNA hardware version 3 and higher.
    // gnaPluginConfig[GNAConfigParams::KEY_GNA_PRECISION] = "I8";
  }
  execution.reset(new ExecutableNetwork(static_cast<IExecutableNetwork::Ptr &>(
      ie_core->LoadNetwork(*(model->network_), device_name, plugin_Config))));
  infer_request.reset(new InferRequest(
      static_cast<IInferRequest::Ptr>(execution->CreateInferRequest())));

  if (preference_ == prefer_t::PREFER_ULTRA_LOW_POWER) {
    s_gna_infer_request = std::move(infer_request);
    s_gna_execution = std::move(execution);
    s_ie_core = std::move(ie_core);
  } else {
    infer_request_ = std::move(infer_request);
    execution_ = std::move(execution);
    ie_core_ = std::move(ie_core);
  }
}

InferRequest *Compilation::GetInferenceRequest() {
  return preference_ == prefer_t::PREFER_ULTRA_LOW_POWER
             ? s_gna_infer_request.get()
             : infer_request_.get();
}

StatusCode Compilation::SetInput(ie_operand_t *operand, const void *buffer,
                                 uint32_t length) {
  InferRequest *infer_request = GetInferenceRequest();
  if (!infer_request) {
    return StatusCode::NETWORK_NOT_LOADED;
  }

  Blob::Ptr input_blob = infer_request->GetBlob(operand->name);
  memcpy(input_blob->buffer(), buffer, length);

  return StatusCode::OK;
}

StatusCode Compilation::GetOutput(ie_operand_t *operand, void *buffer,
                                  uint32_t length) {
  InferRequest *infer_request = GetInferenceRequest();
  if (!infer_request) {
    return StatusCode::NETWORK_NOT_LOADED;
  }

  Blob::Ptr output_blob = infer_request->GetBlob(operand->name);
  memcpy(buffer, output_blob->buffer(), length);

  return StatusCode::OK;
}

StatusCode Compilation::GetBuffer(const char *name, void **buffer,
                                  size_t *byte_length) {
  InferRequest *infer_request = GetInferenceRequest();
  if (!infer_request) {
    return StatusCode::NETWORK_NOT_LOADED;
  }
  Blob::Ptr output_blob = infer_request->GetBlob(name);
  *byte_length = output_blob->byteSize();
  *buffer = malloc(*byte_length);
  memcpy(*buffer, output_blob->buffer(), *byte_length);

  return StatusCode::OK;
}

StatusCode Compilation::Compute(ie_complete_call_back_t *callback) {
  InferRequest *infer_request = GetInferenceRequest();
  if (!infer_request) {
    return StatusCode::NETWORK_NOT_LOADED;
  }

  auto fun = [=]() { callback->completeCallBackFunc(callback->args); };
  infer_request->SetCompletionCallback(fun);
  infer_request->StartAsync();

  return StatusCode::OK;
}

} // namespace InferenceEngine
