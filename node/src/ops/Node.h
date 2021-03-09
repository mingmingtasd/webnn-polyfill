#ifndef __OPS_NODE_H_
#define __OPS_NODE_H_

#include <string>
#include <vector>

#include "../Base.h"

namespace op {

class Node {
public:
  Node(const Napi::CallbackInfo &info);
  ~Node();

  std::vector<WebnnOperand> &GetInputs();
  void SetOutput(WebnnOperand);
  WebnnOperand GetOutput();

private:
  std::vector<WebnnOperand> inputs_;
  WebnnOperand output_;
};
} // namespace op

#endif // __OPS_NODE_H_
