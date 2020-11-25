#include "node.h"

#include "../operand.h"

namespace op {

Node::Node(const Napi::CallbackInfo &info) {
  for (int i = 0; i < info.Length(); ++i) {
    if (info[i].IsObject()) {
      Napi::Object object = info[i].As<Napi::Object>();
      if (object.InstanceOf(Operand::constructor.Value())) {
        Operand* operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        inputs_.push_back(operand->GetOperand());
      }
    }
  }
}

std::vector<WNNOperand>& Node::GetInputs() {
  return inputs_;
}

void Node::SetOutput(WNNOperand operand) {
  output_ = operand;
}

WNNOperand Node::GetOutput() {
  return output_;
}

Node::~Node() {
  wnnOperandRelease(output_);
}

} // namespace op
