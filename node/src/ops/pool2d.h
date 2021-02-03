#ifndef ___OPS_POOL2D_H__
#define ___OPS_POOL2D_H__

#include "node.h"

namespace op {

class Pool2d final : public Node {
public:
  Pool2d(const Napi::CallbackInfo &info);
  ~Pool2d() = default;

  WebnnPool2dOptions *GetOptions();

private:
  WebnnPool2dOptions options_;
  std::vector<int32_t> window_dimensions_;
  std::vector<int32_t> padding_;
  std::vector<int32_t> stride_;
  std::vector<int32_t> dilations_;
};

} // namespace op

#endif // ___OPS_POOL2D_H__
