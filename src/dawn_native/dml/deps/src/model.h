//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

namespace pydml
{
    struct CompiledModel
    {
        CompiledModel(
            dml::Graph& graph, 
            DML_EXECUTION_FLAGS flags,
            std::vector<dml::Expression>& outputs
            ) : 
            op(graph.Compile(flags, outputs))
        {}

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> op;
    };

    struct TensorData
    {
        TensorData(void * buffer,
                   size_t size,
                   int32_t const * dimensions,
                   uint32_t dimensionsCount) :
            buffer_(buffer),
            size_(size)
        {
            dimensions_.assign(dimensions, dimensions + dimensionsCount);
        }

        TensorData(dml::TensorDesc* desc)
        {
            for (auto size : desc->sizes)
            {
                dimensions_.push_back(static_cast<int32_t>(size));
            }

            size_ = desc->totalTensorSizeInBytes;
            buffer_ = malloc(size_);
        }

        TensorData() {}

        void* Get() const { return buffer_; }

        size_t Size() const { return size_; }

        void* buffer_;
        size_t size_;
        std::vector<int32_t> dimensions_;
    };

    struct Binding
    {
        explicit Binding(dml::Expression& expression, 
                         void * buffer,
                         size_t size,
                         int32_t const * dimensions,
                         uint32_t dimensionsCount)
            :   desc(expression.GetOutputDesc()),
                data(buffer, size, dimensions, dimensionsCount)
        {}

        Binding() = default;

        dml::TensorDesc desc;
        TensorData data;
    };
}
