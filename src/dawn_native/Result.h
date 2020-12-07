#ifndef WEBNN_NATIVE_RESULT_H_
#define WEBNN_NATIVE_RESULT_H_

#include <vector>

#include "common/RefCounted.h"
#include "dawn_native/Forward.h"
#include "dawn_native/dawn_platform.h"

namespace dawn_native {

class ResultBase : public RefCounted {
public:
  explicit ResultBase(
    void* buffer,
    uint32_t buffer_size,
    std::vector<uint32_t>& dimensions);
  virtual ~ResultBase() = default;

  // Dawn API
  const void* Buffer() const { return buffer_; }
  uint32_t BufferSize() const { return buffer_size_; }
  const uint32_t* Dimensions() const { return dimensions_.data(); }
  uint32_t DimensionsSize() const {return dimensions_.size(); }

protected:
  void* buffer_;
  uint32_t buffer_size_;
  std::vector<uint32_t> dimensions_;
};
}

#endif  // WEBNN_NATIVE_RESULT_H_