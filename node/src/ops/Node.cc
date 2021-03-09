#include "Node.h"

#include "../Operand.h"

namespace op {

Node::Node(const Napi::CallbackInfo &info) {
  for (int i = 0; i < info.Length(); ++i) {
    if (info[i].IsObject()) {
      Napi::Object object = info[i].As<Napi::Object>();
      if (object.InstanceOf(Operand::constructor.Value())) {
        Operand *operand = Napi::ObjectWrap<Operand>::Unwrap(object);
        inputs_.push_back(operand->GetOperand());
      }
    }
  }
}

std::vector<WebnnOperand> &Node::GetInputs() { return inputs_; }

void Node::SetOutput(WebnnOperand operand) { output_ = operand; }

WebnnOperand Node::GetOutput() { return output_; }

Node::~Node() { webnnOperandRelease(output_); }

} // namespace op
