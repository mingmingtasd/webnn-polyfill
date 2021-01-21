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

#ifndef WEBNN_NATIVE_OPS_UNARY_H_
#define WEBNN_NATIVE_OPS_UNARY_H_

#include "dawn_native/Model.h"
#include "dawn_native/Operand.h"

namespace dawn_native { namespace op {

    enum UnaryOpType {
        kRelu = 0,
        kSoftmax,
    };

    class Unary final : public OperandBase {
      public:
        Unary(ModelBuilderBase* builder, UnaryOpType op_type, OperandBase* input)
            : OperandBase(builder, {input}), op_type_(op_type) {
        }
        ~Unary() override = default;

        MaybeError AddToModel(ModelBase* model) const override {
            return model->AddUnary(this);
        }
        MaybeError Validate() override;
        UnaryOpType GetType() const {
            return op_type_;
        }

      private:
        UnaryOpType op_type_;
    };

}}  // namespace dawn_native::op

#endif  // WEBNN_NATIVE_OPS_UNARY_H_
