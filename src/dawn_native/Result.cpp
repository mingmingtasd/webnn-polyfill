
#include "dawn_native/Result.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {

    ResultBase::ResultBase(void* buffer, uint32_t buffer_size, std::vector<int32_t>& dimensions)
        : buffer_(buffer), buffer_size_(buffer_size), dimensions_(std::move(dimensions)) {
    }
}  // namespace dawn_native