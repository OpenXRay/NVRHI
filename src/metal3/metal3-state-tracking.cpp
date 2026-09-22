#include "metal3-backend.h"
#include <algorithm>

namespace nvrhi::metal3
{
    void CommandList::setEnableAutomaticBarriers(bool enable)
    {
        m_EnableAutomaticBarriers = enable;
    }

    void CommandList::setEnableUavBarriersForTexture(ITexture* texture, bool enableBarriers)
    {
        if (texture)
        {
            m_ReferencedStateResources.emplace_back(texture);
            m_StateTracker.setEnableUavBarriersForTexture(&static_cast<Texture*>(texture)->stateExtension, enableBarriers);
        }
    }

    void CommandList::setEnableUavBarriersForBuffer(IBuffer* buffer, bool enableBarriers)
    {
        if (buffer)
        {
            m_ReferencedStateResources.emplace_back(buffer);
            m_StateTracker.setEnableUavBarriersForBuffer(&static_cast<Buffer*>(buffer)->stateExtension, enableBarriers);
        }
    }

    void CommandList::beginTrackingTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits)
    {
        if (texture)
        {
            m_ReferencedStateResources.emplace_back(texture);
            m_StateTracker.beginTrackingTextureState(&static_cast<Texture*>(texture)->stateExtension, subresources, stateBits);
        }
    }

    void CommandList::beginTrackingBufferState(IBuffer* buffer, ResourceStates stateBits)
    {
        if (buffer)
        {
            m_ReferencedStateResources.emplace_back(buffer);
            m_StateTracker.beginTrackingBufferState(&static_cast<Buffer*>(buffer)->stateExtension, stateBits);
        }
    }

    void CommandList::setTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits)
    {
        if (texture)
        {
            m_ReferencedStateResources.emplace_back(texture);
            m_StateTracker.requireTextureState(&static_cast<Texture*>(texture)->stateExtension, subresources, stateBits);
        }
    }

    void CommandList::setBufferState(IBuffer* buffer, ResourceStates stateBits)
    {
        if (buffer)
        {
            m_ReferencedStateResources.emplace_back(buffer);
            m_StateTracker.requireBufferState(&static_cast<Buffer*>(buffer)->stateExtension, stateBits);
        }
    }

    void CommandList::setPermanentTextureState(ITexture* texture, ResourceStates stateBits)
    {
        if (texture)
        {
            m_ReferencedStateResources.emplace_back(texture);
            m_StateTracker.setPermanentTextureState(&static_cast<Texture*>(texture)->stateExtension, AllSubresources, stateBits);
        }
    }

    void CommandList::setPermanentBufferState(IBuffer* buffer, ResourceStates stateBits)
    {
        if (buffer)
        {
            m_ReferencedStateResources.emplace_back(buffer);
            m_StateTracker.setPermanentBufferState(&static_cast<Buffer*>(buffer)->stateExtension, stateBits);
        }
    }

    ResourceStates CommandList::getTextureSubresourceState(ITexture* texture, ArraySlice arraySlice, MipLevel mipLevel)
    {
        return texture ? m_StateTracker.getTextureSubresourceState(&static_cast<Texture*>(texture)->stateExtension, arraySlice, mipLevel)
            : ResourceStates::Unknown;
    }

    ResourceStates CommandList::getBufferState(IBuffer* buffer)
    {
        return buffer ? m_StateTracker.getBufferState(&static_cast<Buffer*>(buffer)->stateExtension) : ResourceStates::Unknown;
    }

    void CommandList::commitBarriers()
    {
        const auto& textureBarriers = m_StateTracker.getTextureBarriers();
        const auto& bufferBarriers = m_StateTracker.getBufferBarriers();
        if (textureBarriers.empty() && bufferBarriers.empty())
            return;
        if (m_AccelEncoder)
            endEncoding();
        else if (m_ComputeEncoder)
        {
            NSUInteger scope = 0;
            if (!bufferBarriers.empty())
                scope |= MTLBarrierScopeBuffers;
            if (!textureBarriers.empty())
                scope |= MTLBarrierScopeTextures;
            [m_ComputeEncoder memoryBarrierWithScope:MTLBarrierScope(scope)];
        }
        else if (m_RenderEncoder && !m_RenderEncoderWrites.empty())
        {
            bool split = false;
            for (const TextureBarrier& barrier : textureBarriers)
                split |= std::find(m_RenderEncoderWrites.begin(), m_RenderEncoderWrites.end(), barrier.texture) != m_RenderEncoderWrites.end();
            for (const BufferBarrier& barrier : bufferBarriers)
                split |= std::find(m_RenderEncoderWrites.begin(), m_RenderEncoderWrites.end(), barrier.buffer) != m_RenderEncoderWrites.end();
            if (split)
                endEncoding();
        }
        m_StateTracker.clearBarriers();
    }

    void CommandList::setResourceStatesForBindingSet(IBindingSet* bindingSet)
    {
        if (!bindingSet || !bindingSet->getDesc())
            return;
        referenceBindingSet(bindingSet);
        const ResourceStates srvState = getShaderResourceStateForBindingLayout(bindingSet->getLayout());
        for (const BindingSetItem& item : bindingSet->getDesc()->bindings)
        {
            if (!item.resourceHandle)
                continue;
            switch (item.type)
            {
            case ResourceType::Texture_SRV:
                setTextureState(static_cast<ITexture*>(item.resourceHandle), item.subresources, srvState);
                break;
            case ResourceType::Texture_UAV:
                setTextureState(static_cast<ITexture*>(item.resourceHandle), item.subresources, ResourceStates::UnorderedAccess);
                break;
            case ResourceType::TypedBuffer_SRV:
            case ResourceType::StructuredBuffer_SRV:
            case ResourceType::RawBuffer_SRV:
                setBufferState(static_cast<IBuffer*>(item.resourceHandle), srvState);
                break;
            case ResourceType::TypedBuffer_UAV:
            case ResourceType::StructuredBuffer_UAV:
            case ResourceType::RawBuffer_UAV:
                setBufferState(static_cast<IBuffer*>(item.resourceHandle), ResourceStates::UnorderedAccess);
                break;
            case ResourceType::ConstantBuffer:
                setBufferState(static_cast<IBuffer*>(item.resourceHandle), ResourceStates::ConstantBuffer);
                break;
            default:
                break;
            }
        }
    }
}
