#include "metal3-backend.h"
#include "nvrhi/common/misc.h"
#include "nvrhi/common/resource.h"
#include "nvrhi/utils.h"
#include <Metal/Metal.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <limits>

namespace nvrhi::metal3
{
    void MTL3Context::error(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Error, message.c_str());
    }
    void MTL3Context::warning(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Warning, message.c_str());
    }

    void MTL3Context::info(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Info, message.c_str());
    }

    void MTL3Context::unsupported(const char* operation) const
    {
        error(std::string("[nvrhi] Metal backend does not support ") + operation + ".");
    }

    ResidencyRegistry::ResidencyRegistry(const MTL3Context& context)
    {
        MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
        desc.initialCapacity = 4096;
        desc.label = @"nvrhi allocations";
        NSError* error = nil;
        m_Set = [context.device newResidencySetWithDescriptor:desc error:&error];
        if (!m_Set)
        {
            context.error(std::string("[nvrhi] Failed to create the Metal residency set: ") +
                (error ? error.localizedDescription.UTF8String : "unknown error"));
            return;
        }
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
        {
            id<MTLCommandQueue> queue = context.queue(CommandQueue(index));
            if (!queue || std::find(m_Queues.begin(), m_Queues.end(), queue) != m_Queues.end())
                continue;
            [queue addResidencySet:m_Set];
            m_Queues.push_back(queue);
        }
    }

    ResidencyRegistry::~ResidencyRegistry()
    {
        for (id<MTLCommandQueue> queue : m_Queues)
            [queue removeResidencySet:m_Set];
    }

    void ResidencyRegistry::add(id<MTLAllocation> allocation)
    {
        if (!m_Set || !allocation)
            return;
        std::lock_guard<std::mutex> lock(m_Mutex);
        [m_Set addAllocation:allocation];
        m_Dirty = true;
    }

    void ResidencyRegistry::remove(id<MTLAllocation> allocation)
    {
        if (!m_Set || !allocation)
            return;
        std::lock_guard<std::mutex> lock(m_Mutex);
        [m_Set removeAllocation:allocation];
        m_Dirty = true;
    }

    void ResidencyRegistry::commit()
    {
        if (!m_Set)
            return;
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (!m_Dirty)
            return;
        [m_Set commit];
        m_Dirty = false;
    }

    // device creation
    DeviceHandle createDevice(const DeviceDesc& desc)
    {
        if (!desc.pDevice || !desc.commonQueue || desc.commonQueue.device != desc.pDevice
            || ![desc.pDevice supportsFamily:MTLGPUFamilyMetal3]
            || desc.pDevice.argumentBuffersSupport < MTLArgumentBuffersTier2)
        {
            if (desc.errorCB)
                desc.errorCB->message(MessageSeverity::Error,
                    "[nvrhi] Native Metal requires a Metal 3 device, argument buffers tier 2, and a matching command queue.");
            return nullptr;
        }
        Device* device = new Device(desc);
        return DeviceHandle::Create(device);
    }
    Device::Device(const DeviceDesc& desc)
        : m_AftermathEnabled(false)
    {
        m_Context.device = desc.pDevice;
        m_Context.logBufferLifetime = desc.logBufferLifetime;
        m_Context.messageCallback = desc.errorCB;

        m_ShaderLibraryCacheDirectory = desc.shaderLibraryCacheDirectory ? desc.shaderLibraryCacheDirectory : "";
        m_PipelineArchivePath = desc.pipelineArchivePath ? desc.pipelineArchivePath : "";
        if (!m_PipelineArchivePath.empty())
            createPipelineArchive();

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if (@available(macOS 26.0, iOS 26.0, *))
        {
            if ([m_Context.device supportsFamily:MTLGPUFamilyApple10])
                m_Context.maxTextureDimension = 32768;
        }
#endif
        const MTLSize threads = m_Context.device.maxThreadsPerThreadgroup;
        m_Context.info("[nvrhi] Metal limits: texture_dimension=" + std::to_string(m_Context.maxTextureDimension)
            + " buffer_bytes=" + std::to_string(m_Context.device.maxBufferLength)
            + " threadgroup_dimensions=" + std::to_string(threads.width) + "x"
            + std::to_string(threads.height) + "x" + std::to_string(threads.depth));

        // queues, resoureces reserve, allocation, etc...
        m_Context.commonQueue = desc.commonQueue;
        m_Context.queues[uint32_t(CommandQueue::Compute)] = desc.computeQueue;
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
        {
            m_DefaultLifetimeTrackers[index] = RefCountPtr<CommandListLifetimeTracker>::Create(
                new CommandListLifetimeTracker(this, CommandQueue(index), true));
            m_Queues[index].event = [m_Context.device newEvent];
            m_Queues[index].fence = [m_Context.device newFence];
        }
        m_Context.residency = std::make_shared<ResidencyRegistry>(m_Context);
    }

    Device::~Device()
    {
        waitForIdle();
    }

    void Device::createPipelineArchive()
    {
        if (@available(macOS 11.0, *))
        {
            NSError* error = nil;
            std::error_code existsError;
            if (std::filesystem::exists(std::filesystem::path(m_PipelineArchivePath), existsError) && !existsError)
            {
                MTLBinaryArchiveDescriptor* descriptor = [[MTLBinaryArchiveDescriptor alloc] init];
                descriptor.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:m_PipelineArchivePath.c_str()]];
                m_PipelineArchive = [m_Context.device newBinaryArchiveWithDescriptor:descriptor error:&error];
                if (!m_PipelineArchive)
                {
                    std::string message = "[metal3] failed to load the pipeline archive '" + m_PipelineArchivePath +
                        "'; starting a new archive";
                    if (error)
                        message += std::string(": ") + [[error localizedDescription] UTF8String];
                    m_Context.warning(message);
                    std::error_code removeError;
                    std::filesystem::remove(m_PipelineArchivePath, removeError);
                }
                else
                {
                    m_Context.info("[metal3] loaded the pipeline archive '" + m_PipelineArchivePath + "'");
                    return;
                }
            }

            MTLBinaryArchiveDescriptor* emptyDescriptor = [[MTLBinaryArchiveDescriptor alloc] init];
            emptyDescriptor.url = nil;
            m_PipelineArchive = [m_Context.device newBinaryArchiveWithDescriptor:emptyDescriptor error:&error];
            if (!m_PipelineArchive)
            {
                std::string message = "[metal3] failed to create an empty pipeline archive";
                if (error)
                    message += std::string(": ") + [[error localizedDescription] UTF8String];
                m_Context.warning(message);
            }
        }
    }

    bool Device::serializePipelineArchive()
    {
        std::lock_guard<std::mutex> lock(m_PipelineArchiveMutex);
        if (!m_PipelineArchive || !m_PipelineArchiveDirty)
            return true;
        if (m_PipelineArchivePath.empty())
            return false;

        std::filesystem::path targetPath(m_PipelineArchivePath);
        std::filesystem::path temporaryPath = targetPath;
        temporaryPath += ".tmp";

        std::error_code errorCode;
        std::filesystem::create_directories(targetPath.parent_path(), errorCode);

        NSString* path = [NSString stringWithUTF8String:temporaryPath.string().c_str()];
        if (!path)
            return false;

        NSError* error = nil;
        if (![m_PipelineArchive serializeToURL:[NSURL fileURLWithPath:path] error:&error])
        {
            std::string message = "[metal3] failed to write the pipeline archive '" + targetPath.string() + "'";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            m_Context.error(message);
            std::filesystem::remove(temporaryPath, errorCode);
            return false;
        }

        std::filesystem::rename(temporaryPath, targetPath, errorCode);
        if (errorCode)
        {
            m_Context.error("[metal3] failed to replace the pipeline archive '" + targetPath.string() + "'");
            std::filesystem::remove(temporaryPath, errorCode);
            return false;
        }

        m_PipelineArchiveDirty = false;
        return true;
    }

    id<MTLRenderPipelineState> Device::createRenderPipelineState(MTLRenderPipelineDescriptor* descriptor, NSError** error)
    {
        std::lock_guard<std::mutex> lock(m_PipelineArchiveMutex);
        if (m_PipelineArchive)
        {
            if (@available(macOS 11.0, *))
            {
                descriptor.binaryArchives = @[m_PipelineArchive];
                if (error)
                    *error = nil;
                id<MTLRenderPipelineState> archived = [m_Context.device newRenderPipelineStateWithDescriptor:descriptor
                                                                                                      options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                                                   reflection:nil
                                                                                                        error:error];
                if (archived)
                    return archived;
            }
        }

        if (error)
            *error = nil;
        id<MTLRenderPipelineState> pipeline = [m_Context.device newRenderPipelineStateWithDescriptor:descriptor error:error];
        if (!pipeline || !m_PipelineArchive)
            return pipeline;

        if (@available(macOS 11.0, *))
        {
            NSError* archiveError = nil;
            if ([m_PipelineArchive addRenderPipelineFunctionsWithDescriptor:descriptor error:&archiveError])
            {
                m_PipelineArchiveDirty = true;
            }
            else if (archiveError)
            {
                m_Context.warning(std::string("[metal3] failed to add a render pipeline to the archive: ") +
                    [[archiveError localizedDescription] UTF8String]);
            }
        }

        return pipeline;
    }

    id<MTLRenderPipelineState> Device::createMeshRenderPipelineState(MTLMeshRenderPipelineDescriptor* descriptor, NSError** error)
    {
        std::lock_guard<std::mutex> lock(m_PipelineArchiveMutex);
        if (m_PipelineArchive)
        {
            if (@available(macOS 15.0, *))
            {
                descriptor.binaryArchives = @[m_PipelineArchive];
                if (error)
                    *error = nil;
                id<MTLRenderPipelineState> archived = [m_Context.device newRenderPipelineStateWithMeshDescriptor:descriptor
                                                                                                          options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                                                       reflection:nil
                                                                                                            error:error];
                if (archived)
                    return archived;
            }
        }

        if (error)
            *error = nil;
        id<MTLRenderPipelineState> pipeline = [m_Context.device newRenderPipelineStateWithMeshDescriptor:descriptor
                                                                                                   options:MTLPipelineOptionNone
                                                                                                reflection:nil
                                                                                                     error:error];
        if (!pipeline || !m_PipelineArchive)
            return pipeline;

        if (@available(macOS 15.0, *))
        {
            NSError* archiveError = nil;
            if ([m_PipelineArchive addMeshRenderPipelineFunctionsWithDescriptor:descriptor error:&archiveError])
            {
                m_PipelineArchiveDirty = true;
            }
            else if (archiveError)
            {
                m_Context.warning(std::string("[metal3] failed to add a mesh pipeline to the archive: ") +
                    [[archiveError localizedDescription] UTF8String]);
            }
        }

        return pipeline;
    }

    id<MTLComputePipelineState> Device::createComputePipelineState(MTLComputePipelineDescriptor* descriptor, NSError** error)
    {
        std::lock_guard<std::mutex> lock(m_PipelineArchiveMutex);
        if (m_PipelineArchive)
        {
            if (@available(macOS 11.0, *))
            {
                descriptor.binaryArchives = @[m_PipelineArchive];
                if (error)
                    *error = nil;
                id<MTLComputePipelineState> archived = [m_Context.device newComputePipelineStateWithDescriptor:descriptor
                                                                                                         options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                                                      reflection:nil
                                                                                                           error:error];
                if (archived)
                    return archived;
            }
        }

        if (error)
            *error = nil;
        id<MTLComputePipelineState> pipeline = [m_Context.device newComputePipelineStateWithDescriptor:descriptor
                                                                                                 options:MTLPipelineOptionNone
                                                                                              reflection:nil
                                                                                                   error:error];
        if (!pipeline || !m_PipelineArchive)
            return pipeline;

        if (@available(macOS 11.0, *))
        {
            NSError* archiveError = nil;
            if ([m_PipelineArchive addComputePipelineFunctionsWithDescriptor:descriptor error:&archiveError])
            {
                m_PipelineArchiveDirty = true;
            }
            else if (archiveError)
            {
                m_Context.warning(std::string("[metal3] failed to add a compute pipeline to the archive: ") +
                    [[archiveError localizedDescription] UTF8String]);
            }
        }

        return pipeline;
    }

    Object Device::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::MTL3_Device:
            return Object((__bridge void*)m_Context.device);
        case ObjectTypes::Nvrhi_Metal3_Device:
            return Object(this);
        default:
            return nullptr;
        }
    }

    GraphicsAPI Device::getGraphicsAPI()
    {
        return GraphicsAPI::METAL3;
    }
    Object Device::getNativeQueue(ObjectType objectType, CommandQueue queue)
    {
        if (objectType == ObjectTypes::MTL3_CommandQueue && uint32_t(queue) < uint32_t(CommandQueue::Count))
            return Object((__bridge void*)m_Context.queue(queue));
        return nullptr;
    }

    bool Device::waitForIdle()
    {
        id<MTLCommandBuffer> markers[uint32_t(CommandQueue::Count)] = {};
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
            {
                if (index != 0 && m_Context.queue(CommandQueue(index)) == m_Context.commonQueue)
                    continue;
                markers[index] = [m_Context.queue(CommandQueue(index)) commandBuffer];
                if (!markers[index])
                {
                    m_Context.error("[nvrhi] Failed to allocate a Metal idle marker.");
                    return false;
                }
                [markers[index] commit];
            }
        }
        bool success = true;
        for (id<MTLCommandBuffer> marker : markers)
        {
            if (!marker)
                continue;
            [marker waitUntilCompleted];
            success = success && marker.status == MTLCommandBufferStatusCompleted;
        }
        std::lock_guard<std::mutex> lock(m_Mutex);
        updateCompletedSubmissions();
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
        {
            success = success && m_Queues[index].firstFailure == 0;
            m_DefaultLifetimeTrackers[index]->collect();
        }
        if (!success)
            m_Context.error("[nvrhi] Metal queue idle wait detected failed GPU work.");
        return success;
    }

    void Device::updateCompletedSubmissions()
    {
        for (QueueState& queue : m_Queues)
        {
            while (!queue.pending.empty())
            {
                const SubmittedCommandBuffer& submitted = queue.pending.front();
                const MTLCommandBufferStatus status = submitted.commandBuffer.status;
                if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError)
                    break;
                if (status == MTLCommandBufferStatusError && queue.firstFailure == 0)
                {
                    queue.firstFailure = submitted.instance;
                    const char* message = submitted.commandBuffer.error.localizedDescription.UTF8String;
                    if (m_Context.messageCallback)
                        m_Context.messageCallback->message(MessageSeverity::Fatal,
                            (std::string("[nvrhi] Metal GPU submission failed: ") + (message ? message : "unknown error")).c_str());
                    m_Context.info("[metal3] failed submission=" + std::to_string(submitted.instance) +
                        " allocated_bytes=" + std::to_string(m_Context.device.currentAllocatedSize) +
                        " recommended_working_set_bytes=" + std::to_string(m_Context.device.recommendedMaxWorkingSetSize));
                    NSArray<id<MTLCommandBufferEncoderInfo>>* encoders =
                        submitted.commandBuffer.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
                    NSUInteger encoderIndex = 0;
                    for (id<MTLCommandBufferEncoderInfo> encoder in encoders)
                    {
                        const char* state = "unknown";
                        switch (encoder.errorState)
                        {
                        case MTLCommandEncoderErrorStateCompleted: state = "completed"; break;
                        case MTLCommandEncoderErrorStateAffected: state = "affected"; break;
                        case MTLCommandEncoderErrorStatePending: state = "pending"; break;
                        case MTLCommandEncoderErrorStateFaulted: state = "faulted"; break;
                        default: break;
                        }
                        const char* label = encoder.label.UTF8String;
                        m_Context.info("[metal3] encoder=" + std::to_string(encoderIndex++) +
                            " state=" + state + " label=" + (label ? label : "<unlabelled>"));
                        for (NSString* signpost in encoder.debugSignposts)
                            m_Context.info(std::string("[metal3] signpost=") + signpost.UTF8String);
                    }
                    if (!encoders.count)
                        m_Context.info("[metal3] No encoder execution details; enable LDV_METAL3_ENCODER_STATUS=1 before launch.");
                }
                if (submitted.lastInBatch)
                    queue.completed = submitted.instance;
                queue.pending.pop_front();
            }
        }
    }

    bool Device::submissionsSucceeded(const std::array<uint64_t, uint32_t(CommandQueue::Count)>& submissions) const
    {
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
        {
            const QueueState& queue = m_Queues[index];
            if (queue.completed < submissions[index]
                || (queue.firstFailure != 0 && queue.firstFailure <= submissions[index]))
                return false;
        }
        return true;
    }

    void Device::runGarbageCollection()
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        updateCompletedSubmissions();
        for (const auto& tracker : m_DefaultLifetimeTrackers)
            tracker->collect();
    }

    CommandListLifetimeTracker::CommandListLifetimeTracker(Device* device, CommandQueue queue, bool isDefault)
        : m_Device(device), m_Queue(queue), m_IsDefault(isDefault)
    {
    }

    CommandListLifetimeTracker::~CommandListLifetimeTracker()
    {
        if (m_IsDefault || m_CommandBuffers.empty())
            return;
        std::lock_guard<std::mutex> lock(m_Device->m_Mutex);
        auto& retained = m_Device->m_DefaultLifetimeTrackers[uint32_t(m_Queue)]->m_CommandBuffers;
        for (auto& commandBuffer : m_CommandBuffers)
        {
            const MTLCommandBufferStatus status = commandBuffer.commandBuffer.status;
            if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError)
                retained.push_back(std::move(commandBuffer));
        }
    }

    Object CommandListLifetimeTracker::getNativeObject(ObjectType objectType)
    {
        return objectType == ObjectTypes::Nvrhi_Metal3_LifetimeTracker ? Object(this) : Object(nullptr);
    }

    void CommandListLifetimeTracker::collect()
    {
        m_ResourceStorageHighWaterMark = std::max(m_ResourceStorageHighWaterMark, m_CommandBuffers.size());
        auto end = std::remove_if(m_CommandBuffers.begin(), m_CommandBuffers.end(), [this](TrackedCommandBuffer& tracked) {
            const MTLCommandBufferStatus status = tracked.commandBuffer.status;
            if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError)
                return false;
            tracked.commandBuffer = nil;
            tracked.bindingSets.clear();
            tracked.buffers.clear();
            tracked.resources.clear();
            tracked.descriptorSnapshots.clear();
            tracked.stateResources.clear();
            if (m_ResourceStorage.size() < m_ResourceStorageHighWaterMark
                && (tracked.bindingSets.capacity() || tracked.buffers.capacity() || tracked.resources.capacity()
                    || tracked.descriptorSnapshots.capacity() || tracked.stateResources.capacity()))
                m_ResourceStorage.push_back(std::move(tracked));
            return true;
        });
        m_CommandBuffers.erase(end, m_CommandBuffers.end());
    }

    TrackedCommandBuffer CommandListLifetimeTracker::acquireResourceStorage()
    {
        collect();
        if (m_ResourceStorage.empty())
            return {};
        TrackedCommandBuffer storage = std::move(m_ResourceStorage.back());
        m_ResourceStorage.pop_back();
        return storage;
    }

    void CommandListLifetimeTracker::runGarbageCollection()
    {
        std::lock_guard<std::mutex> lock(m_Device->m_Mutex);
        m_Device->updateCompletedSubmissions();
        collect();
    }

    HeapHandle Device::createHeap(const HeapDesc& d)
    {
        (void)d;
        m_Context.unsupported(__func__);
        return nullptr;
    }

    MemoryRequirements Device::getTextureMemoryRequirements(ITexture* texture)
    {
        MemoryRequirements result{};
        if (texture)
        {
            Texture* t = checked_cast<Texture*>(texture);
            result.size = t->memSize;
            result.alignment = t->memAlign;
        }
        return result;
    }

    bool Device::bindTextureMemory(ITexture* texture, IHeap* heap, uint64_t offset)
    {
        (void)texture; (void)heap; (void)offset;
        m_Context.unsupported(__func__);
        return false;
    }

    StagingTextureHandle Device::createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess)
    {
        m_Context.unsupported(__func__);
        return nullptr;
    }

    void* Device::mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t* outRowPitch)
    {
        m_Context.unsupported(__func__);
        if (outRowPitch)
            *outRowPitch = 0;
        return nullptr;
    }

    void Device::unmapStagingTexture(IStagingTexture* tex)
    {
        m_Context.unsupported(__func__);
    }

    void Device::getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings)
    {
        (void)texture;
        (void)numTiles;
        (void)desc;
        (void)tileShape;
        (void)subresourceTilingsNum;
        (void)subresourceTilings;

        m_Context.unsupported(__func__);
    }

    void Device::updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue)
    {
        (void)texture; (void)tileMappings; (void)numTileMappings; (void)executionQueue;
        m_Context.unsupported(__func__);
    }

    SamplerFeedbackTextureHandle Device::createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc)
    {
        (void)pairedTexture;
        (void)desc;

        m_Context.unsupported(__func__);
        return nullptr;
    }

    SamplerFeedbackTextureHandle Device::createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture)
    {
        (void)objectType;
        (void)texture;
        (void)pairedTexture;

        m_Context.unsupported(__func__);
        return nullptr;
    }

    void* Device::mapBuffer(IBuffer* b, CpuAccessMode mapFlags)
    {
        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return nullptr;

        if (mapFlags == CpuAccessMode::None)
        {
            utils::InvalidEnum();
            return nullptr;
        }

        if (buffer->desc.cpuAccess == CpuAccessMode::None)
        {
            m_Context.error("[nvrhi] Cannot map Metal buffer without CPU access.");
            return nullptr;
        }

        if (buffer->desc.cpuAccess != mapFlags)
        {
            m_Context.error("[nvrhi] Metal buffer mapped with incompatible CPU access mode.");
            return nullptr;
        }

        if (buffer->buffer.storageMode == MTLStorageModePrivate)
        {
            m_Context.error("[nvrhi] Cannot map private Metal buffer.");
            return nullptr;
        }

        std::array<uint64_t, uint32_t(CommandQueue::Count)> submissions;
        std::array<id<MTLCommandBuffer>, uint32_t(CommandQueue::Count)> commands{};
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            submissions = buffer->lastUseSubmissions;
            updateCompletedSubmissions();
            for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
            {
                const QueueState& queue = m_Queues[index];
                if (submissions[index] <= queue.completed)
                    continue;
                for (const SubmittedCommandBuffer& submitted : queue.pending)
                {
                    if (submitted.instance == submissions[index] && submitted.lastInBatch)
                    {
                        commands[index] = submitted.commandBuffer;
                        break;
                    }
                }
            }
        }
        for (id<MTLCommandBuffer> commandBuffer : commands)
            [commandBuffer waitUntilCompleted];
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            updateCompletedSubmissions();
            if (!submissionsSucceeded(submissions))
                return nullptr;
        }

        return [buffer->buffer contents];
    }

    void Device::unmapBuffer(IBuffer* b)
    {
        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return;

        if (buffer->buffer.storageMode == MTLStorageModeManaged &&
            buffer->desc.cpuAccess == CpuAccessMode::Write)
        {
            [buffer->buffer didModifyRange:NSMakeRange(0, NSUInteger(buffer->desc.byteSize))];
        }
    }

    MemoryRequirements Device::getBufferMemoryRequirements(IBuffer* b)
    {
        MemoryRequirements result{};

        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer)
            return result;

        const BufferDesc& desc = buffer->desc;
        if (desc.byteSize == 0)
            return result;

        MTLResourceOptions options = convertCpuAccess(desc.cpuAccess);
        MTLSizeAndAlign sizeAndAlign =
            [m_Context.device heapBufferSizeAndAlignWithLength:NSUInteger(desc.byteSize)
                                                    options:options];

        result.size = sizeAndAlign.size;
        result.alignment = sizeAndAlign.align;
        return result;
    }

    bool Device::bindBufferMemory(IBuffer *buffer, IHeap *heap, uint64_t offset)
    {
        (void)buffer;
        (void)heap;
        (void)offset;
        m_Context.unsupported(__func__);
        return false;
    }

    BufferHandle Device::createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc &desc)
    {
        if (!buffer.pointer)
            return nullptr;

        if (objectType != ObjectTypes::MTL3_Buffer)
            return nullptr;

        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer.pointer;
        if (!mtlBuffer)
            return nullptr;

        if (desc.byteSize > 0 && desc.byteSize > [mtlBuffer length])
        {
            m_Context.error("[nvrhi] Native Metal buffer is smaller than BufferDesc::byteSize.");
            return nullptr;
        }
        Buffer *result	   = new Buffer();
        result->desc	   = desc;
        result->buffer	   = (__bridge id<MTLBuffer>)buffer.pointer;
        result->ownsBuffer = false;
        result->residency = m_Context.residency;
        m_Context.residency->add(result->buffer);
        return BufferHandle::Create(result);
    }

    ShaderHandle Device::createShaderSpecialization(IShader *baseShader, const ShaderSpecialization *constants,
												uint32_t numConstants)
    {
        (void)constants;
        (void)numConstants;
        m_Context.unsupported(__func__);
        return nullptr;
    }
    EventQueryHandle Device::createEventQuery()
    {
        return EventQueryHandle::Create(new EventQuery(this));
    }

    EventQuery* Device::getEventQuery(IEventQuery* query)
    {
        auto* event = query ? static_cast<EventQuery*>(query->getNativeObject(ObjectTypes::Nvrhi_Metal3_EventQuery).pointer) : nullptr;
        if (!event || event->device != this)
        {
            m_Context.error("[nvrhi] Metal event query belongs to a different device or backend.");
            return nullptr;
        }
        return event;
    }

    void Device::setEventQuery(IEventQuery* query, CommandQueue queue)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (uint32_t(queue) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal event-query queue.");
            return;
        }
        EventQuery* event = getEventQuery(query);
        if (!event)
            return;
        id<MTLCommandBuffer> commandBuffer = [m_Context.queue(queue) commandBuffer];
        if (!commandBuffer)
        {
            m_Context.error("[nvrhi] Failed to allocate a Metal event marker.");
            return;
        }
        event->commandBuffer = commandBuffer;
        event->failureReported = false;
        for (uint32_t index = 0; index < uint32_t(CommandQueue::Count); ++index)
            event->submissions[index] = m_Queues[index].submitted;
        [commandBuffer commit];
    }

    bool Device::pollEventQuery(IEventQuery* query)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        EventQuery* event = getEventQuery(query);
        if (!event)
            return false;
        if (!event->commandBuffer)
            return true;
        updateCompletedSubmissions();
        if (event->commandBuffer.status == MTLCommandBufferStatusError && !event->failureReported)
        {
            event->failureReported = true;
            m_Context.error("[nvrhi] Metal event marker failed.");
        }
        return event->commandBuffer.status == MTLCommandBufferStatusCompleted
            && submissionsSucceeded(event->submissions);
    }

    void Device::waitEventQuery(IEventQuery* query)
    {
        id<MTLCommandBuffer> commandBuffer;
        std::array<uint64_t, uint32_t(CommandQueue::Count)> submissions;
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            EventQuery* event = getEventQuery(query);
            if (!event || !event->commandBuffer)
                return;
            commandBuffer = event->commandBuffer;
            submissions = event->submissions;
        }
        [commandBuffer waitUntilCompleted];
        std::lock_guard<std::mutex> lock(m_Mutex);
        updateCompletedSubmissions();
        if (commandBuffer.status != MTLCommandBufferStatusCompleted || !submissionsSucceeded(submissions))
            m_Context.error("[nvrhi] Metal event wait detected failed GPU work.");
    }

    void Device::resetEventQuery(IEventQuery* query)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        EventQuery* event = getEventQuery(query);
        if (!event)
            return;
        event->commandBuffer = nil;
        event->submissions.fill(0);
        event->failureReported = false;
    }

    bool Device::queryFeatureSupport(Feature feature, void* pInfo, size_t infoSize)
    {
        (void)pInfo; (void)infoSize;
        switch (feature)
        {
        case Feature::ComputeQueue:
        case Feature::CopyQueue:
        case Feature::ConstantBufferRanges:
        case Feature::DeferredCommandLists:
        case Feature::Meshlets:
            return true;
        case Feature::RayTracingAccelStruct:
        case Feature::RayQuery:
        case Feature::RayTracingCompaction:
            return [m_Context.device supportsRaytracing];
        case Feature::BufferInt64Atomics:
            return [m_Context.device supportsFamily:MTLGPUFamilyApple9]
                || [m_Context.device supportsFamily:MTLGPUFamilyMac2]
                || [m_Context.device supportsFamily:MTLGPUFamilyApple8];
        default:
            return false;
        }
    }
    MTLPixelFormat convertFormat(nvrhi::Format format)
    {
        switch (format)
        {
        case Format::RGBA8_UNORM:
            return MTLPixelFormatRGBA8Unorm;
        case Format::BGRA8_UNORM:
            return MTLPixelFormatBGRA8Unorm;
        case Format::SRGBA8_UNORM:
            return MTLPixelFormatRGBA8Unorm_sRGB;
        case Format::SBGRA8_UNORM:
            return MTLPixelFormatBGRA8Unorm_sRGB;
        case Format::R8_UNORM:
            return MTLPixelFormatR8Unorm;
        case Format::RG8_UNORM:
            return MTLPixelFormatRG8Unorm;
        case Format::R32_UINT:
            return MTLPixelFormatR32Uint;
        case Format::R16_FLOAT:
            return MTLPixelFormatR16Float;
        case Format::RG16_FLOAT:
            return MTLPixelFormatRG16Float;
        case Format::RGBA16_FLOAT:
            return MTLPixelFormatRGBA16Float;
        case Format::RGBA16_UNORM:
            return MTLPixelFormatRGBA16Unorm;
        case Format::R32_FLOAT:
            return MTLPixelFormatR32Float;
        case Format::RG32_FLOAT:
            return MTLPixelFormatRG32Float;
        case Format::RGBA32_FLOAT:
            return MTLPixelFormatRGBA32Float;
        case Format::BC1_UNORM:
            return MTLPixelFormatBC1_RGBA;
        case Format::BC1_UNORM_SRGB:
            return MTLPixelFormatBC1_RGBA_sRGB;
        case Format::BC2_UNORM:
            return MTLPixelFormatBC2_RGBA;
        case Format::BC2_UNORM_SRGB:
            return MTLPixelFormatBC2_RGBA_sRGB;
        case Format::BC3_UNORM:
            return MTLPixelFormatBC3_RGBA;
        case Format::BC3_UNORM_SRGB:
            return MTLPixelFormatBC3_RGBA_sRGB;
        case Format::BC4_UNORM:
            return MTLPixelFormatBC4_RUnorm;
        case Format::BC4_SNORM:
            return MTLPixelFormatBC4_RSnorm;
        case Format::BC5_UNORM:
            return MTLPixelFormatBC5_RGUnorm;
        case Format::BC5_SNORM:
            return MTLPixelFormatBC5_RGSnorm;
        case Format::BC6H_UFLOAT:
            return MTLPixelFormatBC6H_RGBUfloat;
        case Format::BC6H_SFLOAT:
            return MTLPixelFormatBC6H_RGBFloat;
        case Format::BC7_UNORM:
            return MTLPixelFormatBC7_RGBAUnorm;
        case Format::BC7_UNORM_SRGB:
            return MTLPixelFormatBC7_RGBAUnorm_sRGB;
        case Format::D32:
            return MTLPixelFormatDepth32Float;
        case Format::D16:
            return MTLPixelFormatDepth16Unorm;
        case Format::D24S8:
            return MTLPixelFormatDepth24Unorm_Stencil8;
        case Format::D32S8:
            return MTLPixelFormatDepth32Float_Stencil8;
        default:
            return MTLPixelFormatInvalid;
        }
    }

    rt::OpacityMicromapHandle Device::createOpacityMicromap(const rt::OpacityMicromapDesc& desc)
    {
        m_Context.unsupported(__func__);
        return nullptr;
    }


    nvrhi::CommandListHandle Device::createCommandList(const CommandListParameters& params)
    {
        if (uint32_t(params.queueType) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal command-list queue type.");
            return nullptr;
        }
        if (params.lifetimeTracker)
        {
            auto* tracker = static_cast<CommandListLifetimeTracker*>(
                params.lifetimeTracker->getNativeObject(ObjectTypes::Nvrhi_Metal3_LifetimeTracker).pointer);
            if (!tracker || tracker->m_Device != this || tracker->m_Queue != params.queueType)
            {
                m_Context.error("[nvrhi] Metal lifetime tracker belongs to a different device, queue, or backend.");
                return nullptr;
            }
        }
        return nvrhi::CommandListHandle::Create(new CommandList(this, m_Context, params));
    }

    uint64_t Device::executeCommandLists(nvrhi::ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue)
    {
        if (numCommandLists == 0)
            return 0;
        if (!pCommandLists || uint32_t(executionQueue) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal command-list submission array or queue.");
            return 0;
        }

        std::lock_guard<std::mutex> lock(m_Mutex);
        size_t prepared = 0;
        for (; prepared < numCommandLists; ++prepared)
        {
            nvrhi::ICommandList* list = pCommandLists[prepared];
            auto* commandList = list ? static_cast<CommandList*>(
                list->getNativeObject(ObjectTypes::Nvrhi_Metal3_CommandList).pointer) : nullptr;
            if (!commandList || commandList->m_Device != this
                || commandList->m_Desc.queueType != executionQueue
                || commandList->m_RecordingState != CommandList::RecordingState::Closed
                || commandList->m_RecordingFailed
                || commandList->trackedCmdBuffer.status != MTLCommandBufferStatusNotEnqueued)
                break;
            commandList->m_RecordingState = CommandList::RecordingState::PendingSubmission;
        }

        if (prepared != numCommandLists)
        {
            for (size_t index = 0; index < prepared; ++index)
            {
                auto* commandList = static_cast<CommandList*>(
                    pCommandLists[index]->getNativeObject(ObjectTypes::Nvrhi_Metal3_CommandList).pointer);
                commandList->m_RecordingState = CommandList::RecordingState::Closed;
            }
            m_Context.error("[nvrhi] Metal submission requires unique, closed, unsubmitted command lists from this device and queue.");
            return 0;
        }

        QueueState& queue = m_Queues[uint32_t(executionQueue)];
        const uint64_t instance = ++queue.submitted;
        if (!queue.pendingWaits.empty())
        {
            id<MTLCommandBuffer> waits = [m_Context.queue(executionQueue) commandBufferWithUnretainedReferences];
            if (!waits)
            {
                m_Context.error("[nvrhi] Failed to allocate a Metal queue wait command buffer.");
                return 0;
            }
            for (const auto& wait : queue.pendingWaits)
                [waits encodeWaitForEvent:wait.first value:wait.second];
            [waits commit];
            queue.pendingWaits.clear();
        }
        m_Context.residency->commit();
        for (size_t index = 0; index < numCommandLists; ++index)
        {
            auto* commandList = static_cast<CommandList*>(
                pCommandLists[index]->getNativeObject(ObjectTypes::Nvrhi_Metal3_CommandList).pointer);
            queue.pending.push_back({commandList->trackedCmdBuffer, instance, index + 1 == numCommandLists});
            for (const BufferHandle& buffer : commandList->m_ReferencedMappableBuffers)
                static_cast<Buffer*>(buffer.Get())->lastUseSubmissions[uint32_t(executionQueue)] = instance;
            if (index + 1 == numCommandLists)
                [commandList->trackedCmdBuffer encodeSignalEvent:queue.event value:instance];
            commandList->submit();
        }
        return instance;
    }

    void Device::queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (uint32_t(waitQueue) >= uint32_t(CommandQueue::Count)
            || uint32_t(executionQueue) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal dependency queue.");
            return;
        }
        updateCompletedSubmissions();
        const QueueState& producer = m_Queues[uint32_t(executionQueue)];
        if (instance > producer.submitted)
        {
            m_Context.error("[nvrhi] Metal queue dependency references an unsubmitted execution identifier.");
            return;
        }
        if (producer.firstFailure != 0 && producer.firstFailure <= instance)
            m_Context.error("[nvrhi] Metal queue dependency references failed GPU work.");
        if (m_Context.queue(waitQueue) == m_Context.queue(executionQueue) || instance <= producer.completed)
            return;
        m_Queues[uint32_t(waitQueue)].pendingWaits.emplace_back(producer.event, instance);
    }

    CommandListLifetimeTrackerHandle Device::createCommandListLifetimeTracker(CommandQueue executionQueue)
    {
        if (uint32_t(executionQueue) >= uint32_t(CommandQueue::Count))
        {
            m_Context.error("[nvrhi] Invalid Metal lifetime-tracker queue.");
            return nullptr;
        }
        return CommandListLifetimeTrackerHandle::Create(new CommandListLifetimeTracker(this, executionQueue));
    }

    FormatSupport Device::queryFormatSupport(Format format)
    {
        if (uint32_t(format) >= uint32_t(Format::COUNT) || format == Format::UNKNOWN)
            return FormatSupport::None;
        FormatSupport support = FormatSupport::None;
        if (convertVertexFormat(format) != MTLVertexFormatInvalid)
            support = support | FormatSupport::Buffer | FormatSupport::VertexBuffer;
        if (format == Format::R16_UINT || format == Format::R32_UINT)
            support = support | FormatSupport::Buffer | FormatSupport::IndexBuffer;
        if (convertFormat(format) == MTLPixelFormatInvalid)
            return support;
        const FormatInfo& info = getFormatInfo(format);
        if (info.blockSize > 1)
            return m_Context.device.supportsBCTextureCompression
                ? support | FormatSupport::Texture | FormatSupport::ShaderLoad | FormatSupport::ShaderSample
                : support;
        if (format == Format::D24S8 && !m_Context.device.depth24Stencil8PixelFormatSupported)
            return support;
        support = support | FormatSupport::Texture | FormatSupport::ShaderLoad;
        const bool float32 = format == Format::R32_FLOAT || format == Format::RG32_FLOAT
            || format == Format::RGBA32_FLOAT || format == Format::D32 || format == Format::D32S8;
        if (format != Format::R32_UINT && (!float32 || m_Context.device.supports32BitFloatFiltering))
            support = support | FormatSupport::ShaderSample;
        if (info.hasDepth || info.hasStencil)
            return support | FormatSupport::DepthStencil;
        support = support | FormatSupport::RenderTarget;
        if (format != Format::R32_UINT)
            support = support | FormatSupport::Blendable;
        if (!info.isSRGB)
            support = support | FormatSupport::Buffer;
        if ((!info.isSRGB && format != Format::BGRA8_UNORM) || [m_Context.device supportsFamily:MTLGPUFamilyApple2])
            support = support | FormatSupport::ShaderUavStore;
        const MTLReadWriteTextureTier tier = m_Context.device.readWriteTextureSupport;
        if ((tier >= MTLReadWriteTextureTier1 && (format == Format::R32_UINT || format == Format::R32_FLOAT))
            || (tier >= MTLReadWriteTextureTier2 && (format == Format::RGBA8_UNORM
                || format == Format::RGBA16_FLOAT || format == Format::RGBA32_FLOAT
                || format == Format::R16_FLOAT || format == Format::R8_UNORM)))
            support = support | FormatSupport::ShaderUavLoad;
        return support;
    }

    coopvec::DeviceFeatures Device::queryCoopVecFeatures()
    {
        return coopvec::DeviceFeatures{};
    }

    coopvec::MatMulFormatSupport Device::queryCoopVecMatMulFormatSupport(const coopvec::MatMulFormatCombo& combination)
    {
        (void)combination;
        return coopvec::MatMulFormatSupport{};
    }

    coopvec::TrainingFormatSupport Device::queryCoopVecTrainingFormatSupport(coopvec::DataType componentType)
    {
        (void)componentType;
        return coopvec::TrainingFormatSupport{};
    }

    size_t Device::getCoopVecMatrixSize(coopvec::DataType type, coopvec::MatrixLayout layout, int rows, int columns)
    {
        (void)type;
        (void)layout;
        (void)rows;
        (void)columns;
        m_Context.unsupported(__func__);
        return 0;
    }
}
