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
#include "dawn_native/Compilation.h"
#include "dawn_native/ModelBuilder.h"

// Contains the entry-points into dawn_native
namespace dawn_native {
DawnProcTable GetProcsAutogen();

DawnProcTable GetProcs() { return GetProcsAutogen(); }

namespace null {
NeuralNetworkContextBase *Create();
}
namespace ie {
NeuralNetworkContextBase *Create();
}
namespace dml {
NeuralNetworkContextBase *Create();
}

// Should put the default null backend at the end.
WNNNeuralNetworkContext CreateNeuralNetworkContext() {
#if defined(DAWN_ENABLE_BACKEND_IE)
  return reinterpret_cast<WNNNeuralNetworkContext>(ie::Create());
#elif defined(DAWN_ENABLE_BACKEND_DML)
  return reinterpret_cast<WNNNeuralNetworkContext>(dml::Create());
#elif defined(DAWN_ENABLE_BACKEND_NULL)
  return reinterpret_cast<WNNNeuralNetworkContext>(null::Create());
#else
  return nullptr;
#endif
}

WNNNamedInputs CreateNamedInputs() {
  return reinterpret_cast<WNNNamedInputs>(new NamedInputsBase());
}

WNNNamedOperands CreateNamedOperands() {
  return reinterpret_cast<WNNNamedOperands>(new NamedOperandsBase());
}

WNNNamedOutputs CreateNamedOutputs() {
  return reinterpret_cast<WNNNamedOutputs>(new NamedOutputsBase());
}

} // namespace dawn_native
