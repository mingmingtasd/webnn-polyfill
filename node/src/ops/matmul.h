#ifndef ___OPS_MAT_MUL_H__
#define ___OPS_MAT_MUL_H__

#include "operand_wrap.h"

namespace op {

class MatMul final : public OperandWrap {
public:
  MatMul(const Napi::CallbackInfo &info);
  ~MatMul() = default;

  void AddToModel(WNNModelBuilder builder);
private:
  WNNOperand primary_;
  WNNOperand secondary_;
};

} // namespace op

#endif // ___OPS_MAT_MUL_H__
