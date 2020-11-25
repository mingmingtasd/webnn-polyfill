#ifndef ___OPS_CONV2D_H__
#define ___OPS_CONV2D_H__

#include "node.h"

namespace op {

class Conv2d final : public Node {
public:
  Conv2d(const Napi::CallbackInfo &info);
  ~Conv2d() = default;

  WNNConv2dOptions* GetOptions();

private:
  WNNConv2dOptions options_;
  std::vector<int32_t> padding_;
  std::vector<int32_t> stride_;
  std::vector<int32_t> dilations_;
};

} // namespace op

#endif // ___OPS_CONV2D_H__
