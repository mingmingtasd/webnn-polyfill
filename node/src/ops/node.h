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

  std::vector<WNNOperand> &GetInputs();
  void SetOutput(WNNOperand);
  WNNOperand GetOutput();

private:
  std::vector<WNNOperand> inputs_;
  WNNOperand output_;
};
} // namespace op

#endif // __OPS_NODE_H_