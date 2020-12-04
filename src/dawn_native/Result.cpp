
#include "dawn_native/Result.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {

ResultBase::ResultBase(void* buffer, uint32_t buffer_size,
                       std::vector<uint32_t>& dimensions) :
    buffer_(buffer),
    buffer_size_(buffer_size), 
    dimensions_(std::move(dimensions)) {
}

}