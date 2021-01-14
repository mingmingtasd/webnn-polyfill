// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef IE_COMPILATION_H
#define IE_COMPILATION_H

#include <inference_engine.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_model.h"
#include "ie_nn_c_api.h"
#include "ngraph/node_output.hpp"
#include "ngraph/op/parameter.hpp"
#include "utils.h"

namespace InferenceEngine {

class Compilation {
public:
  explicit Compilation(std::shared_ptr<Model> model);
  ~Compilation() = default;

  StatusCode SetInput(ie_operand_t *operand, const void *buffer,
                      uint32_t length);
  StatusCode GetOutput(ie_operand_t *operand, void *buffer, uint32_t length);
  StatusCode Compute(ie_complete_call_back_t *callback);
  StatusCode GetBuffer(const char *name, void **buffer, size_t *byte_length);
  StatusCode GetDimensions(const char *name, ie_dimensions_t *dimensions);

private:
  InferRequest *GetInferenceRequest();
  prefer_t preference_;

  std::unique_ptr<InferRequest> infer_request_;
  std::unique_ptr<ExecutableNetwork> execution_;
  std::unique_ptr<Core> ie_core_;

  DISALLOW_COPY_AND_ASSIGN(Compilation);
};

} // namespace InferenceEngine

#endif // IE_COMPILATION_H
