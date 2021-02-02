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

#ifndef WEBNN_NATIVE_WEBNN_NATIVE_H_
#define WEBNN_NATIVE_WEBNN_NATIVE_H_

#include <webnn/webnn.h>
#include <webnn/webnn_proc_table.h>
#include <webnn_native/webnn_native_export.h>
#include <string>
#include <vector>

namespace webnn_native {

    // Backend-agnostic API for webnn_native
    WEBNN_NATIVE_EXPORT WebnnProcTable GetProcs();

    WEBNN_NATIVE_EXPORT WebnnNeuralNetworkContext CreateNeuralNetworkContext();
    WEBNN_NATIVE_EXPORT WebnnNamedInputs CreateNamedInputs();
    WEBNN_NATIVE_EXPORT WebnnNamedOperands CreateNamedOperands();
    WEBNN_NATIVE_EXPORT WebnnNamedOutputs CreateNamedOutputs();

}  // namespace webnn_native

#endif  // WEBNN_NATIVE_WEBNN_NATIVE_H_
