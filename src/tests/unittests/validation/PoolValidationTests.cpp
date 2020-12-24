#include "tests/unittests/validation/ValidationTest.h"

#include <memory>

using namespace testing;

class PoolValidationTest : public ValidationTest {
  protected:
    void SetUp() override {
      ValidationTest::SetUp();
      builder_ = context.CreateModelBuilder();
    }
    wnn::ModelBuilder builder_;
};

TEST_F(PoolValidationTest, CreateByDefaultOptions) {
    std::vector<int32_t> shape = {1, 100, 1000, 1000};
    wnn::OperandDescriptor inputDesc = {wnn::OperandType::Float32, shape.data(),
                                             (uint32_t)shape.size()};
    wnn::Operand input = builder_.Input("input", &inputDesc);
    // Success
    {
      // using default value for options
      wnn::Pool2dOptions pool2dOptions = {};
      wnn::Operand pool = builder_.AveragePool2d(input, &pool2dOptions);
      EXPECT_NE(&pool, nullptr);
    }
    {
      wnn::Operand pool = builder_.MaxPool2d(input);
      EXPECT_NE(&pool, nullptr);
    }
}


TEST_F(PoolValidationTest, InputDimsError) {
    // input is not a 4D tensor
    std::vector<int32_t> shape = {1, 100, 1000, 1000, 1};
    wnn::OperandDescriptor inputDesc = {wnn::OperandType::Float32, shape.data(),
                                             (uint32_t)shape.size()};
    wnn::Operand input = builder_.Input("input", &inputDesc);

    wnn::Pool2dOptions pool2dOptions = {};
    ASSERT_CONTEXT_ERROR(builder_.MaxPool2d(input, &pool2dOptions));
}

TEST_F(PoolValidationTest, windowDimensionsCountError) {
    // windowDimensionsCount is incorrect
    std::vector<int32_t> shape = {1, 100, 1000, 1000};
    wnn::OperandDescriptor inputDesc = {wnn::OperandType::Float32, shape.data(),
                                             (uint32_t)shape.size()};
    wnn::Operand input = builder_.Input("input", &inputDesc);

    wnn::Pool2dOptions options;
    std::vector<int32_t> windowDimensions = {2, 2, 1};
    options.windowDimensions = windowDimensions.data();
    options.windowDimensionsCount = 3;
    options.strides = nullptr;
    options.padding = nullptr;
    options.dilations = nullptr;
    ASSERT_CONTEXT_ERROR(builder_.MaxPool2d(input, &options));
}

TEST_F(PoolValidationTest, paddingCountError) {
    // paddingCount is incorrect
    std::vector<int32_t> shape = {1, 100, 1000, 1000};
    wnn::OperandDescriptor inputDesc = {wnn::OperandType::Float32, shape.data(),
                                             (uint32_t)shape.size()};
    wnn::Operand input = builder_.Input("input", &inputDesc);

    wnn::Pool2dOptions options;
    options.windowDimensions = nullptr;
    options.strides = nullptr;
    std::vector<int32_t> padding = {1, 1};
    options.padding = padding.data();
    options.paddingCount = 2;
    options.dilations = nullptr;
    ASSERT_CONTEXT_ERROR(builder_.MaxPool2d(input, &options));
}

TEST_F(PoolValidationTest, StridesCountError) {
    // stridesCount is incorrect
    std::vector<int32_t> shape = {1, 100, 1000, 1000};
    wnn::OperandDescriptor inputDesc = {wnn::OperandType::Float32, shape.data(),
                                             (uint32_t)shape.size()};
    wnn::Operand input = builder_.Input("input", &inputDesc);

    wnn::Pool2dOptions options;
    options.windowDimensions = nullptr;
    std::vector<int32_t> strides = {1};
    options.strides = strides.data();
    options.stridesCount = 1;
    options.padding = nullptr;
    options.dilations = nullptr;
    ASSERT_CONTEXT_ERROR(builder_.MaxPool2d(input, &options));
}

TEST_F(PoolValidationTest, DilationsCountError) {
    // dilationsCount is incorrect
    std::vector<int32_t> shape = {1, 100, 1000, 1000};
    wnn::OperandDescriptor inputDesc = {wnn::OperandType::Float32, shape.data(),
                                             (uint32_t)shape.size()};
    wnn::Operand input = builder_.Input("input", &inputDesc);

    wnn::Pool2dOptions options;
    options.windowDimensions = nullptr;
    options.strides = nullptr;
    options.padding = nullptr;
    std::vector<int32_t> dilations = {1};
    options.dilations = dilations.data();
    options.dilationsCount = 1;
    ASSERT_CONTEXT_ERROR(builder_.MaxPool2d(input, &options));
}

