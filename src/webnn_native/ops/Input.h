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

#ifndef WEBNN_NATIVE_OPS_INPUT_H_
#define WEBNN_NATIVE_OPS_INPUT_H_

#include <memory>
#include <string>

#include "webnn_native/Model.h"
#include "webnn_native/Operand.h"

namespace webnn_native { namespace op {

    class Input final : public OperandBase {
      public:
        Input(ModelBuilderBase* builder, const std::string& name, const OperandDescriptor* desc)
            : OperandBase(builder), name_(name) {
            descriptor_.type = desc->type;
            type_ = desc->type;
            rank_ = desc->dimensionsCount;
            dimensions_.assign(desc->dimensions, desc->dimensions + desc->dimensionsCount);
            descriptor_.dimensions = dimensions_.data();
            descriptor_.dimensionsCount = dimensions_.size();
        }
        ~Input() override = default;

        MaybeError AddToModel(ModelBase* model) const override {
            return model->AddInput(this);
        }
        MaybeError ValidateAndInferTypes() override;

        const std::string& GetName() const {
            return name_;
        }
        const OperandDescriptor* GetOperandDescriptor() const {
            return &descriptor_;
        }

      private:
        std::string name_;
        OperandDescriptor descriptor_;
        std::vector<int32_t> dimensions_;
    };

}}  // namespace webnn_native::op

#endif  // WEBNN_NATIVE_OPS_INPUT_H_
