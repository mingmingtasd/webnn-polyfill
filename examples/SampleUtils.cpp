// Copyright 2017 The Dawn Authors
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

#include "SampleUtils.h"

#include <dawn/dawn_proc.h>
#include <dawn/webnn.h>
#include <dawn/webnn_cpp.h>
#include <dawn_native/DawnNative.h>

wnn::ModelBuilder CreateCppModelBuilder() {
  DawnProcTable backendProcs = dawn_native::GetProcs();
  dawnProcSetProcs(&backendProcs);
  dawn_native::NeuralNetworkContext context;
  return wnn::ModelBuilder::Acquire(context.CreateModelBuilder());
}

wnn::Inputs CreateCppInputs() {
  return wnn::Inputs::Acquire(dawn_native::CreateInputs());
}

wnn::Outputs CreateCppOutputs() {
  return wnn::Outputs::Acquire(dawn_native::CreateOutputs());
}
