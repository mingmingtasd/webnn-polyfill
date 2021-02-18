//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

#pragma warning(push)
#pragma warning(disable:4238)

using namespace pydml;
using Microsoft::WRL::ComPtr;

Device::Device(bool useGpu, bool useDebugLayer) : m_useGpu(useGpu), m_useDebugLayer(useDebugLayer) {}

HRESULT Device::Init()
{
    // 
    // Create D3D12 resources
    //

    if (m_useDebugLayer)
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();
        }
    }

    if (    !m_useGpu 
        ||  FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_d3d12Device))))
    {
        Microsoft::WRL::ComPtr<IDXGIFactory4> dxgiFactory;
        ReturnIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
        Microsoft::WRL::ComPtr<IDXGIAdapter3> pAdapter;
        ReturnIfFailed(dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&pAdapter)));
        ReturnIfFailed(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_d3d12Device)));
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ReturnIfFailed(m_d3d12Device->CreateCommandQueue(&queueDesc, IID_GRAPHICS_PPV_ARGS(m_commandQueue.GetAddressOf())));

    ReturnIfFailed(m_d3d12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_GRAPHICS_PPV_ARGS(m_commandAllocator.GetAddressOf())));

    ReturnIfFailed(m_d3d12Device->CreateCommandList(
        0, // node mask
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        m_commandAllocator.Get(),
        nullptr, // initial pipeline state
        IID_GRAPHICS_PPV_ARGS(m_commandList.GetAddressOf())));

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = 4; // One each for the input, output, persistent, and temporary resources
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ReturnIfFailed(m_d3d12Device->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(m_clearUavDescriptorHeapCpu.GetAddressOf())));

    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ReturnIfFailed(m_d3d12Device->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(m_clearUavDescriptorHeapGpu.GetAddressOf())));


    // 
    // Create DML resources
    //

    if (    !m_useDebugLayer 
        ||  FAILED(DMLCreateDevice(m_d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_DEBUG, IID_PPV_ARGS(&m_dmlDevice))))
    {
        ReturnIfFailed(DMLCreateDevice(m_d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)));
    }

    ReturnIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_commandRecorder)));
    ReturnIfFailed(m_dmlDevice->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&m_operatorInitializer)));
    ReturnIfFailed(m_dmlDevice->CreateBindingTable(nullptr, IID_PPV_ARGS(&m_bindingTable)));
    return S_OK;
}

HRESULT Device::DispatchOperator(
    IDMLCompiledOperator* op,
    const std::vector<pydml::Binding*>& inputs,
    const std::vector<dml::Expression*>& outputs,
    std::vector<pydml::TensorData*>& outputData
    )
{
    std::vector<DmlBufferBinding> inputBindings(inputs.size());
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto input = inputs[i];

        if (!input)
        {
            continue; // null optional tensor
        }

        DmlBufferTensorDesc desc = *input->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        // If OWNED_BY_DML is *not* set, this input must be bound at execution
        // fix clang error: logical not is only applied to the left hand side of this bitwise operator [-Werror,-Wlogical-not-parentheses]
        if (!(desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML))
        {
            uint32_t requiredAlignment = std::max(desc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

            // Bind to the end of the inputs resource (with appropriate alignment)
            inputBindings[i].offset = RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
            inputBindings[i].sizeInBytes = desc.totalTensorSizeInBytes;

            inputsResourceSize = inputBindings[i].offset + desc.totalTensorSizeInBytes;
        }
    }

    std::vector<DmlBufferBinding> outputBindings(outputs.size());
    uint64_t outputsResourceSize = 0;

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto output = outputs[i];

        if (!output)
        {
            continue; // null optional tensor
        }

        dml::TensorDesc desc = output->GetOutputDesc();
        DmlBufferTensorDesc bufferDesc = *desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        uint32_t requiredAlignment = std::max(bufferDesc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

        // Bind to the end of the outputs resource (with appropriate alignment)
        outputBindings[i].offset = RoundUpToMultiple(outputsResourceSize, (uint64_t)requiredAlignment);
        outputBindings[i].sizeInBytes = bufferDesc.totalTensorSizeInBytes;

        outputsResourceSize = outputBindings[i].offset + outputBindings[i].sizeInBytes;
    }

    DML_BINDING_PROPERTIES bindingProps = op->GetBindingProperties();

    ReturnIfFailed(EnsureUploadHeapSize(inputsResourceSize));
    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(inputsResourceSize, m_inputsResource));
    ReturnIfFailed(EnsureReadBackHeapSize(outputsResourceSize));
    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(outputsResourceSize, m_outputsResource));
    ReturnIfFailed(EnsureDefaultBufferSize(bindingProps.TemporaryResourceSize, m_temporaryResource));
    ReturnIfFailed(EnsureDescriptorHeapSize(bindingProps.RequiredDescriptorCount));

    // Set up input and output bindings to point to their respective buffers
    for (auto& binding : inputBindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_inputsResource.Get();
        }
    }

    for (auto& binding : outputBindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_outputsResource.Get();
        }
    }

    // The persistent resource should have already been initialized when the operator was initialized
    assert(m_persistentResource->GetDesc().Width >= bindingProps.PersistentResourceSize);

    // Upload inputs for execution
    std::vector<ID3D12Resource*> buffersToClear =
    {
        m_inputsResource.Get(),
        m_temporaryResource.Get(),
        m_outputsResource.Get()
    };
    
    ReturnIfFailed(ClearGpuBuffers(buffersToClear));

    if (inputsResourceSize)
    {
        // Copy the data into the upload heap
        byte* uploadHeapData = nullptr;

        ReturnIfFailed(m_uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&uploadHeapData)));

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputBindings[i].buffer)
            {
                // This input tensor doesn't need to be bound for initialize
                continue;
            }

            DmlBufferTensorDesc bufferDesc = *inputs[i]->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

            void* dest = uploadHeapData + inputBindings[i].offset;
            const void* src = inputs[i]->data.Get();

            assert(inputs[i]->data.Size() == bufferDesc.totalTensorSizeInBytes);

            memcpy(dest, src, static_cast<size_t>(bufferDesc.totalTensorSizeInBytes));
        }

        m_uploadHeap->Unmap(0, nullptr);

        // Record the copy from the upload heap into the inputs resource
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST)
            );

        m_commandList->CopyBufferRegion(m_inputsResource.Get(), 0, m_uploadHeap.Get(), 0, inputsResourceSize);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            );
    }

    // Bind for execution
    DmlTypeConverter<1024> converter;

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = op;
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = bindingProps.RequiredDescriptorCount;

    ReturnIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    // Bind inputs
    std::vector<DML_BINDING_DESC> inputBindingDescs(inputBindings.size());
    for (size_t i = 0; i < inputBindings.size(); ++i)
    {
        inputBindingDescs[i] = converter.ToBindingDesc(inputBindings[i]);
    }

    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindingDescs.size()), inputBindingDescs.data());

    // Bind outputs
    std::vector<DML_BINDING_DESC> outputBindingDescs(outputBindings.size());
    for (size_t i = 0; i < outputBindings.size(); ++i)
    {
        outputBindingDescs[i] = converter.ToBindingDesc(outputBindings[i]);
    }

    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingDescs.size()), outputBindingDescs.data());

    // Bind persistent/temporary resources
    if (bindingProps.PersistentResourceSize != 0)
    {
        DML_BUFFER_BINDING persistentBinding = { m_persistentResource.Get(), 0, bindingProps.PersistentResourceSize };
        auto bindingDesc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &persistentBinding };
        m_bindingTable->BindPersistentResource(&bindingDesc);
    }

    if (bindingProps.TemporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING temporaryBinding = { m_temporaryResource.Get(), 0, bindingProps.TemporaryResourceSize };
        auto bindingDesc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &temporaryBinding };
        m_bindingTable->BindTemporaryResource(&bindingDesc);
    }

    // Record and execute commands, and wait for completion
    m_commandList->SetDescriptorHeaps(1, m_descriptorHeap.GetAddressOf());
    m_commandRecorder->RecordDispatch(m_commandList.Get(), op, m_bindingTable.Get());
    RecordOutputReadBack(outputsResourceSize);
    ReturnIfFailed(ExecuteCommandListAndWait());

    // Read the output data back from the readback heap
    ReturnIfFailed(DownloadFromReadBackHeap(outputsResourceSize, outputs, outputBindings, outputData));
    return S_OK;
}

void Device::RecordOutputReadBack(uint64_t outputsResourceSize)
{
    // Copy output to readback heap
    if (outputsResourceSize != 0)
    {
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE)
            );

        m_commandList->CopyBufferRegion(m_readbackHeap.Get(), 0, m_outputsResource.Get(), 0, outputsResourceSize);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            );
    }
}

 HRESULT Device::DownloadFromReadBackHeap(
    uint64_t outputsResourceSize, 
    const std::vector<dml::Expression*>& outputs,
    const std::vector<DmlBufferBinding>& outputBindings,
    std::vector<pydml::TensorData*>& outputData
    )
{
    if (outputsResourceSize != 0)
    {
        CD3DX12_RANGE readRange(0, static_cast<size_t>(outputsResourceSize));

        byte* readbackHeapData = nullptr;

        ReturnIfFailed(m_readbackHeap->Map(0, &readRange, reinterpret_cast<void**>(&readbackHeapData)));

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];

            if (!output)
            {
                // This output tensor is optional (and null)
                continue;
            }

            dml::TensorDesc desc = output->GetOutputDesc();
            DmlBufferTensorDesc bufferDesc = *desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

            auto data = new TensorData(&desc);
            void* dest = data->Get();
            const void* src = readbackHeapData + outputBindings[i].offset;

            memcpy(dest, src, static_cast<size_t>(bufferDesc.totalTensorSizeInBytes));

            outputData.push_back(data);
        }

        m_readbackHeap->Unmap(0, nullptr);
    }

    return S_OK;
}

HRESULT Device::InitializeOperator(
    IDMLCompiledOperator* op,
    const std::vector<pydml::Binding*>& inputs
    )
{
    // Allocate resources for initialization
    ReturnIfFailed(m_operatorInitializer->Reset(1, &op));

    DmlBufferArrayBinding inputBinding;
    inputBinding.bindings.resize(inputs.size());

    // Fill in the offsets and sizes for each binding, which will also tell us how big we need to make our buffer
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto input = inputs[i];

        if (!input)
        {
            continue; // null optional tensor
        }

        DmlBufferTensorDesc bufferDesc = *input->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        // If OWNED_BY_DML is set, this input must be bound at initialize
        if (bufferDesc.flags & DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            uint32_t requiredAlignment = std::max(bufferDesc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

            // Bind to the end of the inputs resource (with appropriate alignment)
            inputBinding.bindings[i].offset = RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
            inputBinding.bindings[i].sizeInBytes = bufferDesc.totalTensorSizeInBytes;

            inputsResourceSize = inputBinding.bindings[i].offset + bufferDesc.totalTensorSizeInBytes;
        }
    }

    uint64_t temporaryResourceSize = m_operatorInitializer->GetBindingProperties().TemporaryResourceSize;
    uint64_t persistentResourceSize = op->GetBindingProperties().PersistentResourceSize;
    uint32_t descriptorHeapSize = m_operatorInitializer->GetBindingProperties().RequiredDescriptorCount;

    ReturnIfFailed(EnsureUploadHeapSize(inputsResourceSize));
    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(inputsResourceSize, m_inputsResource));
    ReturnIfFailed(EnsureDefaultBufferSize(temporaryResourceSize, m_temporaryResource));
    ReturnIfFailed(EnsureDefaultBufferSize(persistentResourceSize, m_persistentResource));
    ReturnIfFailed(EnsureDescriptorHeapSize(descriptorHeapSize));

    // Set up the bindings to point to our input resource
    for (auto& binding : inputBinding.bindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_inputsResource.Get();
        }
    }

    // Upload inputs for initialization
    std::vector<ID3D12Resource*> buffersToClear =
    {
        m_inputsResource.Get(),
        m_temporaryResource.Get(),
        m_persistentResource.Get()
    };

    ReturnIfFailed(ClearGpuBuffers(buffersToClear));

    if (inputsResourceSize)
    {
        // Copy the data into the upload heap
        byte* uploadHeapData = nullptr;

        ReturnIfFailed(m_uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&uploadHeapData)));

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputBinding.bindings[i].buffer)
            {
                // This input tensor doesn't need to be bound for initialize
                continue;
            }

            DmlBufferTensorDesc bufferDesc = *inputs[i]->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

            void* dest = uploadHeapData + inputBinding.bindings[i].offset;
            const void* src = inputs[i]->data.Get();

            assert(inputs[i]->data.Size() == bufferDesc.totalTensorSizeInBytes);

            memcpy(dest, src, static_cast<size_t>(bufferDesc.totalTensorSizeInBytes));
        }

        m_uploadHeap->Unmap(0, nullptr);

        // Record the copy from the upload heap into the inputs resource
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST)
            );

        m_commandList->CopyBufferRegion(m_inputsResource.Get(), 0, m_uploadHeap.Get(), 0, inputsResourceSize);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                );
    }

    // Bind for initialization
    DmlTypeConverter<1024> converter;

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_operatorInitializer.Get();
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = descriptorHeapSize;

    ReturnIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    DML_BINDING_DESC inputBindingDesc = converter.ToBindingDesc(inputBinding);
    m_bindingTable->BindInputs(1, &inputBindingDesc);

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING outputBinding = { m_persistentResource.Get(), 0, persistentResourceSize };
        auto desc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &outputBinding };
        m_bindingTable->BindOutputs(1, &desc);
    }

    if (temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING temporaryBinding = { m_temporaryResource.Get(), 0, temporaryResourceSize };
        auto desc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &temporaryBinding };
        m_bindingTable->BindTemporaryResource(&desc);
    }

    // Record and execute commands, and wait for completion
    m_commandList->SetDescriptorHeaps(1, m_descriptorHeap.GetAddressOf());
    m_commandRecorder->RecordDispatch(m_commandList.Get(), m_operatorInitializer.Get(), m_bindingTable.Get());
    ReturnIfFailed(ExecuteCommandListAndWait());
    return S_OK;
}

HRESULT Device::ExecuteCommandListAndWait()
{
    ReturnIfFailed(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

    WaitForQueueToComplete(m_commandQueue.Get());

    ReturnIfFailed(m_commandAllocator->Reset());
    ReturnIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
    return S_OK;
}

HRESULT Device::EnsureUploadHeapSize(uint64_t requestedSizeInBytes)
{
    uint64_t existingSize = m_uploadHeap ? m_uploadHeap->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes);     // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536));  // Minimum size of 64k

    if (newSize != existingSize)
    {
        m_uploadHeap = nullptr;
        ReturnIfFailed(CreateCommittedResource(
            m_d3d12Device.Get(),
            CD3DX12_RESOURCE_DESC::Buffer(newSize),
            CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            m_uploadHeap
            ));
    }
    return S_OK;
}

HRESULT Device::EnsureCpuOrDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<ID3D12Resource>& buffer)
{
    if (m_useCpuCustomHeapResources)
    {
        ReturnIfFailed(EnsureCpuBufferSize(requestedSizeInBytes, buffer));
    }
    else
    {
        ReturnIfFailed(EnsureDefaultBufferSize(requestedSizeInBytes, buffer));
    }
    return S_OK;
}

HRESULT Device::EnsureCpuBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<ID3D12Resource>& buffer)
{
    uint64_t existingSize = buffer ? buffer->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes);     // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536));  // Minimum size of 64k

    if (newSize != existingSize)
    {
        buffer = nullptr;
        ReturnIfFailed(CreateCpuCustomBuffer(m_d3d12Device.Get(), newSize, buffer));
    }
    return S_OK;
}

HRESULT Device::EnsureDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<ID3D12Resource>& buffer)
{
    uint64_t existingSize = buffer ? buffer->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes);     // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536));  // Minimum size of 64k

    if (newSize != existingSize)
    {
        buffer = nullptr;
        ReturnIfFailed(CreateDefaultBuffer(m_d3d12Device.Get(), newSize, buffer));
    }
    return S_OK;
}

HRESULT Device::EnsureDescriptorHeapSize(uint32_t requestedSizeInDescriptors)
{
    uint32_t existingSize = m_descriptorHeap ? m_descriptorHeap->GetDesc().NumDescriptors : 0;
    uint32_t newSize = RoundUpToPow2(requestedSizeInDescriptors); // ensures geometric growth

    if (newSize != existingSize)
    {
        m_descriptorHeap = nullptr;

        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = newSize;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        ReturnIfFailed(m_d3d12Device->CreateDescriptorHeap(&desc, IID_GRAPHICS_PPV_ARGS(m_descriptorHeap.GetAddressOf())));
    }
    return S_OK;
}

HRESULT Device::EnsureReadBackHeapSize(uint64_t requestedSizeInBytes)
{
    uint64_t existingSize = m_readbackHeap ? m_readbackHeap->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes); // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536)); // Minimum size of 64k

    if (newSize != existingSize)
    {
        m_readbackHeap = nullptr;
        ReturnIfFailed(CreateReadBackBuffer(m_d3d12Device.Get(), newSize, m_readbackHeap));
    }
    return S_OK;
}

HRESULT Device::ClearGpuBuffers(dml::Span<ID3D12Resource*> buffers)
{
    static const uint32_t ClearValue = static_cast<uint32_t>(-1);

    // The number of buffers we can clear at once is limited by the size of our descriptor heap
    assert(static_cast<uint32_t>(buffers.size()) <= m_clearUavDescriptorHeapCpu->GetDesc().NumDescriptors);

    uint32_t descriptorOffset = 0;
    for (ID3D12Resource* buffer : buffers)
    {
        if (!buffer)
        {
            // Nothing to clear; these buffers are lazily-initialized
            continue;
        }

        ReturnIfFailed(FillGpuBuffer(
            m_commandList.Get(),
            m_clearUavDescriptorHeapCpu.Get(),
            m_clearUavDescriptorHeapGpu.Get(),
            descriptorOffset,
            buffer,
            ClearValue));

        ++descriptorOffset;
    }

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
    return S_OK;
}

#pragma warning(pop)