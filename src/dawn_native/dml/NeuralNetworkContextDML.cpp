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

#include "dawn_native/dml/NeuralNetworkContextDML.h"

#include "common/RefCounted.h"
#include "dawn_native/dml/ModelBuilderDML.h"

namespace dawn_native { namespace dml {

    NeuralNetworkContextBase* Create() {
        Ref<NeuralNetworkContextBase> context = AcquireRef(new NeuralNetworkContext());
        return context.Detach();
    }

    ModelBuilderBase* NeuralNetworkContext::CreateModelBuilderImpl() {
        Ref<ModelBuilderBase> builder = AcquireRef(new ModelBuilder(this));
        return builder.Detach();
    }

}}  // namespace dawn_native::dml
