#ifndef ___OPS_TRANSPOSE_H__
#define ___OPS_TRANSPOSE_H__

#include <unordered_map>

#include "Node.h"

namespace op {

class Transpose final : public Node {
public:
  Transpose(const Napi::CallbackInfo &info);
  ~Transpose() = default;

  WebnnTransposeOptions *GetOptions();

private:
  WebnnTransposeOptions options_;
  std::vector<int32_t> permutation_;
};

} // namespace op

#endif // ___OPS_TRANSPOSE_H__
