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
                   size_t size) :
            buffer_(buffer),
            size_(size) {}

        TensorData(dml::TensorDesc* desc) :
            size_(desc->totalTensorSizeInBytes),
            desc_(*desc->AsPtr<DML_BUFFER_TENSOR_DESC>())
        {
            // Free by user code.
            buffer_ = malloc(size_);
        }

        TensorData() {}

        void* Get() const { return buffer_; }

        size_t Size() const { return size_; }

        const dml::TensorDesc* Desc() const { return &desc_; }

        void* buffer_;
        size_t size_;
        dml::TensorDesc desc_;
    };

    struct Binding
    {
        explicit Binding(dml::Expression& expression, 
                         void * buffer,
                         size_t size)
            :   desc(expression.GetOutputDesc()),
                data(buffer, size)
        {}

        Binding() = default;

        dml::TensorDesc desc;
        TensorData data;
    };
}
