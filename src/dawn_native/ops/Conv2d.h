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

#ifndef WEBNN_NATIVE_OPS_CONV2D_H_
#define WEBNN_NATIVE_OPS_CONV2D_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native { namespace op {

    class Conv2d final : public OperandBase {
      public:
        Conv2d(ModelBuilderBase* builder,
               OperandBase* input,
               OperandBase* filter,
               Conv2dOptions const* options);
        ~Conv2d() override = default;

        MaybeError AddToModel(ModelBase* model) const override;
        MaybeError Validate() override;

        Conv2dOptions const* GetOptions() const;

      private:
        Conv2dOptions options_;
        std::vector<int32_t> padding_;
        std::vector<int32_t> stride_;
        std::vector<int32_t> dilations_;
    };

}}  // namespace dawn_native::op

#endif  // WEBNN_NATIVE_OPS_CONV2D_H_
