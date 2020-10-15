// Copyright 2018 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dawn_native/DawnNative.h"

#include <memory>

#include "common/Assert.h"
#include "dawn_native/Inputs.h"
#include "dawn_native/Outputs.h"

// Contains the entry-points into dawn_native
namespace dawn_native {
namespace ie {
ModelBuilderBase *Create();
}
// Context
NeuralNetworkContext::NeuralNetworkContext() = default;

NeuralNetworkContext::~NeuralNetworkContext() = default;

WNNModelBuilder NeuralNetworkContext::CreateModelBuilder() {
  return reinterpret_cast<WNNModelBuilder>(ie::Create());
}

DawnProcTable GetProcsAutogen();

DawnProcTable GetProcs() { return GetProcsAutogen(); }

WNNInputs CreateInputs() {
  return reinterpret_cast<WNNInputs>(new InputsBase());
}

WNNOutputs CreateOutputs() {
  return reinterpret_cast<WNNOutputs>(new OutputsBase());
}

} // namespace dawn_native
