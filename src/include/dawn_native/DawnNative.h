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

#ifndef DAWNNATIVE_DAWNNATIVE_H_
#define DAWNNATIVE_DAWNNATIVE_H_

#include <dawn/dawn_proc_table.h>
#include <dawn/webnn.h>
#include <dawn_native/dawn_native_export.h>
#include <string>
#include <vector>

namespace dawn_native {

class DAWN_NATIVE_EXPORT NeuralNetworkContext {
public:
  NeuralNetworkContext();
  ~NeuralNetworkContext();

  WNNModelBuilder CreateModelBuilder();
};

// Backend-agnostic API for dawn_native
DAWN_NATIVE_EXPORT DawnProcTable GetProcs();

DAWN_NATIVE_EXPORT WNNInputs CreateInputs();
DAWN_NATIVE_EXPORT WNNOutputs CreateOutputs();

}  // namespace dawn_native

#endif  // DAWNNATIVE_DAWNNATIVE_H_