// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dawn_native/Result.h"

#include "common/Assert.h"
#include "common/Log.h"

namespace dawn_native {

    ResultBase::ResultBase(void* buffer, uint32_t buffer_size, std::vector<int32_t>& dimensions)
        : buffer_(buffer), buffer_size_(buffer_size), dimensions_(std::move(dimensions)) {
    }
}  // namespace dawn_native