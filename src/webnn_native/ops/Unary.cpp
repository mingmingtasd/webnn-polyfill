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

#include "webnn_native/ops/Unary.h"

#include "common/Log.h"
#include "webnn_native/Error.h"

namespace webnn_native { namespace op {

    MaybeError Unary::ValidateAndInferTypes() {
        auto input = mInputs[0];
        if (input->IsError()) {
            return DAWN_VALIDATION_ERROR("Argument input is invalid.");
        }

        if (mOpType == UnaryOpType::kSoftmax) {
            if (input->Rank() != 2) {
                return DAWN_VALIDATION_ERROR("Input dimensions is incorrect.");
            }
        }

        // only for softmax and relu
        mType = input->Type();
        mRank = input->Rank();

        return {};
    }

}}  // namespace webnn_native::op
