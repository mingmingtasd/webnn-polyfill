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

#ifndef WEBNN_NATIVE_OPS_RESHAPE_H_
#define WEBNN_NATIVE_OPS_RESHAPE_H_

#include "webnn_native/Model.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    class Reshape final : public OperandBase {
      public:
        Reshape(ModelBuilderBase* builder,
                OperandBase* input,
                int32_t const* new_shape,
                size_t new_shape_count)
            : OperandBase(builder, {input}) {
            new_shape_.assign(new_shape, new_shape + new_shape_count);
        }
        ~Reshape() override = default;

        MaybeError AddToModel(ModelBase* model) const override {
            return model->AddReshape(this);
        }
        MaybeError ValidateAndInferTypes() override;
        int32_t const* GetNewShape() const {
            return new_shape_.data();
        }
        size_t GetNewShapeCount() const {
            return new_shape_.size();
        }

      private:
        std::vector<int32_t> new_shape_;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_RESHAPE_H_
