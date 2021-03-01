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

#include "src/tests/WebnnTest.h"

class ReshapeTests : public WebnnTest {
  public:
    void testReshape(const std::vector<int32_t>& oldShape,
                     const std::vector<int32_t>& newShape,
                     const std::vector<int32_t>& expectedShape = std::vector<int32_t>()) {
        const webnn::ModelBuilder builder = GetContext().CreateModelBuilder();
        const webnn::Operand a = utils::BuildInput(builder, "a", oldShape);
        const std::vector<float> data({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
        const webnn::Operand b = builder.Reshape(a, newShape.data(), newShape.size());
        const webnn::Model model = utils::CreateModel(builder, {{"b", b}});
        const webnn::Compilation compiledModel = utils::AwaitCompile(model);
        const webnn::Input input = {data.data(), data.size() * sizeof(float)};
        const webnn::Result result = utils::AwaitCompute(compiledModel, {{"a", input}}).Get("b");
        if (expectedShape.empty()) {
            EXPECT_TRUE(utils::CheckShape(result, newShape));
        } else {
            EXPECT_TRUE(utils::CheckShape(result, expectedShape));
        }

        EXPECT_TRUE(utils::CheckValue(result, data));
    }
};

TEST_F(ReshapeTests, ReshapeReorderedAllDims) {
    testReshape({2, 3, 4}, {4, 2, 3});
}

TEST_F(ReshapeTests, ReshapeReorderedLastDims) {
    testReshape({2, 3, 4}, {2, 4, 3});
}

TEST_F(ReshapeTests, ReshapeReducedDims) {
    testReshape({2, 3, 4}, {2, 12});
}

TEST_F(ReshapeTests, ReshapeExtendedDims) {
    testReshape({2, 3, 4}, {2, 3, 2, 2});
}

TEST_F(ReshapeTests, ReshapeOneDim) {
    testReshape({2, 3, 4}, {24});
}

TEST_F(ReshapeTests, ReshapeNegativeDim) {
    testReshape({2, 3, 4}, {2, -1, 2}, {2, 6, 2});
}

TEST_F(ReshapeTests, ReshapeNegativeDim1) {
    testReshape({2, 3, 4}, {-1, 2, 3, 4}, {1, 2, 3, 4});
}