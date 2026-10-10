#include "metal3-backend.h"
#include <Metal/Metal.h>
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>
#include <metal_irconverter_runtime/ir_raytracing.h>
#include <algorithm>
#include <cstring>

namespace nvrhi::metal3
{
    static MTLAccelerationStructureUsage convertBuildFlags(rt::AccelStructBuildFlags flags)
    {
        MTLAccelerationStructureUsage usage = MTLAccelerationStructureUsageNone;
        if ((flags & rt::AccelStructBuildFlags::AllowUpdate) != rt::AccelStructBuildFlags::None)
            usage |= MTLAccelerationStructureUsageRefit;
        return usage;
    }

    static MTLAttributeFormat convertAttributeFormat(Format format)
    {
        switch (format)
        {
        case Format::RGB32_FLOAT: return MTLAttributeFormatFloat3;
        case Format::RG32_FLOAT: return MTLAttributeFormatFloat2;
        case Format::RGBA32_FLOAT: return MTLAttributeFormatFloat4;
        case Format::RGBA16_FLOAT: return MTLAttributeFormatHalf4;
        case Format::RG16_FLOAT: return MTLAttributeFormatHalf2;
        case Format::R32_FLOAT: return MTLAttributeFormatFloat;
        default: return MTLAttributeFormatFloat3;
        }
    }

    AccelStruct::~AccelStruct()
    {
        if (residency)
        {
            if (accelStruct) residency->remove(accelStruct);
            if (gpuHeaderBuffer) residency->remove(gpuHeaderBuffer);
            if (instanceContributionsBuffer) residency->remove(instanceContributionsBuffer);
        }
    }

    Object AccelStruct::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::MTL3_AccelerationStructure)
            return Object((__bridge void*)accelStruct);
        return nullptr;
    }

    rt::AccelStructHandle Device::createAccelStruct(const rt::AccelStructDesc& d)
    {
        if (![m_Context.device supportsRaytracing])
        {
            m_Context.error("[nvrhi] Metal device does not support ray tracing.");
            return nullptr;
        }

        MTLAccelerationStructureDescriptor* descriptor = nil;
        if (d.isTopLevel)
        {
            MTLInstanceAccelerationStructureDescriptor* tlas = [MTLInstanceAccelerationStructureDescriptor descriptor];
            tlas.instanceCount = d.topLevelMaxInstances;
            tlas.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
            tlas.usage = convertBuildFlags(d.buildFlags);
            descriptor = tlas;
        }
        else
        {
            MTLPrimitiveAccelerationStructureDescriptor* blas = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
            NSMutableArray* geometries = [NSMutableArray arrayWithCapacity:d.bottomLevelGeometries.size()];
            for (const rt::GeometryDesc& geom : d.bottomLevelGeometries)
            {
                if (geom.geometryType != rt::GeometryType::Triangles)
                    continue;
                const rt::GeometryTriangles& tri = geom.geometryData.triangles;
                MTLAccelerationStructureTriangleGeometryDescriptor* gd =
                    [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
                auto* vb = static_cast<Buffer*>(tri.vertexBuffer);
                gd.vertexBuffer = vb ? vb->buffer : nil;
                gd.vertexBufferOffset = tri.vertexOffset;
                gd.vertexStride = tri.vertexStride;
                gd.vertexFormat = convertAttributeFormat(tri.vertexFormat);
                if (tri.indexBuffer)
                {
                    auto* ib = static_cast<Buffer*>(tri.indexBuffer);
                    gd.indexBuffer = ib ? ib->buffer : nil;
                    gd.indexBufferOffset = tri.indexOffset;
                    gd.indexType = tri.indexFormat == Format::R16_UINT
                        ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
                    gd.triangleCount = tri.indexCount / 3;
                }
                else
                {
                    gd.triangleCount = tri.vertexCount / 3;
                }
                gd.opaque = (geom.flags & rt::GeometryFlags::Opaque) != rt::GeometryFlags::None;
                if (geom.useTransform)
                {
                    id<MTLBuffer> transformBuffer = [m_Context.device newBufferWithBytes:geom.transform
                        length:sizeof(rt::AffineTransform)
                        options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
                    m_Context.residency->add(transformBuffer);
                    gd.transformationMatrixBuffer = transformBuffer;
                    gd.transformationMatrixBufferOffset = 0;
                }
                [geometries addObject:gd];
            }
            blas.geometryDescriptors = geometries;
            blas.usage = convertBuildFlags(d.buildFlags);
            descriptor = blas;
        }

        MTLAccelerationStructureSizes sizes = [m_Context.device accelerationStructureSizesWithDescriptor:descriptor];
        id<MTLAccelerationStructure> native = [m_Context.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        if (!native)
        {
            m_Context.error("[nvrhi] Failed to create Metal acceleration structure.");
            return nullptr;
        }

        if (!d.debugName.empty())
            native.label = [NSString stringWithUTF8String:d.debugName.c_str()];

        m_Context.residency->add(native);

        AccelStruct* as = new AccelStruct();
        as->desc = d;
        as->accelStruct = native;
        as->residency = m_Context.residency;
        as->allowCompaction = (d.buildFlags & rt::AccelStructBuildFlags::AllowCompaction) != rt::AccelStructBuildFlags::None;
        as->allowUpdate = (d.buildFlags & rt::AccelStructBuildFlags::AllowUpdate) != rt::AccelStructBuildFlags::None;
        return rt::AccelStructHandle::Create(as);
    }

    MemoryRequirements Device::getAccelStructMemoryRequirements(rt::IAccelStruct* abstractAs)
    {
        auto* as = static_cast<AccelStruct*>(abstractAs);
        MemoryRequirements requirements{};
        if (as && as->accelStruct)
        {
            requirements.size = as->accelStruct.size;
            requirements.alignment = 256;
        }
        return requirements;
    }

    void Device::queueCompaction(AccelStruct* as, id<MTLBuffer> sizeBuffer, id<MTLCommandBuffer> buildCommandBuffer)
    {
        std::lock_guard<std::mutex> lock(m_CompactionMutex);
        PendingCompaction pending;
        pending.accelStruct = as;
        pending.sizeBuffer = sizeBuffer;
        pending.buildCommandBuffer = buildCommandBuffer;
        pending.buildSerial = as->buildSerial;
        m_PendingCompactions.push_back(std::move(pending));
    }

    void Device::takeCompletedCompactions(std::vector<PendingCompaction>& completed)
    {
        std::lock_guard<std::mutex> lock(m_CompactionMutex);
        for (size_t i = 0; i < m_PendingCompactions.size();)
        {
            PendingCompaction& pending = m_PendingCompactions[i];
            const MTLCommandBufferStatus status = pending.buildCommandBuffer ? pending.buildCommandBuffer.status
                : MTLCommandBufferStatusError;
            if (status == MTLCommandBufferStatusCompleted)
            {
                completed.push_back(std::move(pending));
                m_PendingCompactions.erase(m_PendingCompactions.begin() + i);
            }
            else if (status == MTLCommandBufferStatusError)
                m_PendingCompactions.erase(m_PendingCompactions.begin() + i);
            else
                ++i;
        }
    }

    bool Device::bindAccelStructMemory(rt::IAccelStruct* as, IHeap* heap, uint64_t offset)
    {
        (void)as;
        (void)heap;
        (void)offset;
        return false;
    }

    rt::cluster::OperationSizeInfo Device::getClusterOperationSizeInfo(const rt::cluster::OperationParams& params)
    {
        (void)params;
        return rt::cluster::OperationSizeInfo{};
    }

    static MTLAccelerationStructureDescriptor* buildDescriptorForBLAS(
        const rt::GeometryDesc* pGeometries, size_t numGeometries,
        rt::AccelStructBuildFlags buildFlags)
    {
        MTLPrimitiveAccelerationStructureDescriptor* blas = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        NSMutableArray* geometries = [NSMutableArray arrayWithCapacity:numGeometries];
        for (size_t i = 0; i < numGeometries; ++i)
        {
            const rt::GeometryDesc& geom = pGeometries[i];
            if (geom.geometryType != rt::GeometryType::Triangles)
                continue;
            const rt::GeometryTriangles& tri = geom.geometryData.triangles;
            MTLAccelerationStructureTriangleGeometryDescriptor* gd =
                [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            auto* vb = static_cast<Buffer*>(tri.vertexBuffer);
            gd.vertexBuffer = vb ? vb->buffer : nil;
            gd.vertexBufferOffset = tri.vertexOffset;
            gd.vertexStride = tri.vertexStride;
            gd.vertexFormat = convertAttributeFormat(tri.vertexFormat);
            if (tri.indexBuffer)
            {
                auto* ib = static_cast<Buffer*>(tri.indexBuffer);
                gd.indexBuffer = ib ? ib->buffer : nil;
                gd.indexBufferOffset = tri.indexOffset;
                gd.indexType = tri.indexFormat == Format::R16_UINT
                    ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
                gd.triangleCount = tri.indexCount / 3;
            }
            else
            {
                gd.triangleCount = tri.vertexCount / 3;
            }
            gd.opaque = (geom.flags & rt::GeometryFlags::Opaque) != rt::GeometryFlags::None;
            [geometries addObject:gd];
        }
        blas.geometryDescriptors = geometries;
        blas.usage = convertBuildFlags(buildFlags);
        return blas;
    }

    static void captureGeometryState(const rt::GeometryDesc* pGeometries, size_t numGeometries,
        std::vector<AccelStruct::GeometryState>& states, std::vector<id<MTLBuffer>>& buffers)
    {
        states.clear();
        buffers.clear();
        for (size_t i = 0; i < numGeometries; ++i)
        {
            const rt::GeometryDesc& geom = pGeometries[i];
            if (geom.geometryType != rt::GeometryType::Triangles)
                continue;
            const rt::GeometryTriangles& tri = geom.geometryData.triangles;
            auto* vb = static_cast<Buffer*>(tri.vertexBuffer);
            auto* ib = static_cast<Buffer*>(tri.indexBuffer);
            AccelStruct::GeometryState state;
            state.vertexBuffer = vb ? (__bridge const void*)vb->buffer : nullptr;
            state.indexBuffer = ib ? (__bridge const void*)ib->buffer : nullptr;
            state.vertexOffset = tri.vertexOffset;
            state.vertexCount = tri.vertexCount;
            state.vertexStride = tri.vertexStride;
            state.vertexFormat = uint32_t(convertAttributeFormat(tri.vertexFormat));
            state.indexed = tri.indexBuffer != nullptr;
            if (state.indexed)
            {
                state.indexOffset = tri.indexOffset;
                state.indexCount = tri.indexCount;
                state.indexFormat = uint32_t(tri.indexFormat == Format::R16_UINT
                    ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32);
            }
            state.geometryFlags = uint32_t(geom.flags);
            state.useTransform = geom.useTransform;
            if (geom.useTransform)
                std::memcpy(state.transform, geom.transform, sizeof(rt::AffineTransform));
            states.push_back(state);
            buffers.push_back(vb ? vb->buffer : nil);
            buffers.push_back(ib ? ib->buffer : nil);
        }
    }

    static bool geometryLayoutMatches(const std::vector<AccelStruct::GeometryState>& cached,
        const std::vector<AccelStruct::GeometryState>& requested)
    {
        if (cached.size() != requested.size())
            return false;
        for (size_t i = 0; i < cached.size(); ++i)
        {
            const AccelStruct::GeometryState& a = cached[i];
            const AccelStruct::GeometryState& b = requested[i];
            if (a.vertexCount != b.vertexCount || a.indexCount != b.indexCount ||
                a.vertexStride != b.vertexStride || a.vertexFormat != b.vertexFormat ||
                a.indexFormat != b.indexFormat || a.geometryFlags != b.geometryFlags ||
                a.indexed != b.indexed || a.useTransform != b.useTransform)
                return false;
            if (a.useTransform && std::memcmp(a.transform, b.transform, sizeof(rt::AffineTransform)) != 0)
                return false;
        }
        return true;
    }

    static bool blasArrayMatches(const std::vector<id<MTLAccelerationStructure>>& cached,
        const std::vector<id<MTLAccelerationStructure>>& requested)
    {
        if (cached.size() != requested.size())
            return false;
        for (size_t i = 0; i < cached.size(); ++i)
        {
            if (cached[i] != requested[i])
                return false;
        }
        return true;
    }

    static bool instanceLayoutMatches(const std::vector<MTLAccelerationStructureUserIDInstanceDescriptor>& cached,
        const MTLAccelerationStructureUserIDInstanceDescriptor* requested, size_t count)
    {
        if (cached.size() != count)
            return false;
        for (size_t i = 0; i < count; ++i)
        {
            const auto& a = cached[i];
            const auto& b = requested[i];
            if (a.options != b.options || a.mask != b.mask ||
                a.intersectionFunctionTableOffset != b.intersectionFunctionTableOffset ||
                a.accelerationStructureIndex != b.accelerationStructureIndex || a.userID != b.userID)
                return false;
        }
        return true;
    }

    static void resetUpdateState(AccelStruct* as)
    {
        as->built = false;
        as->builtInstanceCount = 0;
        as->builtUsage = MTLAccelerationStructureUsageNone;
        as->builtGeometry.clear();
        as->builtGeometryBuffers.clear();
        as->builtBottomLevelAS.clear();
        as->builtInstances.clear();
    }

    void CommandList::buildBottomLevelAccelStruct(rt::IAccelStruct* abstractAs,
        const rt::GeometryDesc* pGeometries, size_t numGeometries,
        rt::AccelStructBuildFlags buildFlags)
    {
        auto* as = static_cast<AccelStruct*>(abstractAs);
        if (!as || !as->accelStruct || !pGeometries || numGeometries == 0)
            return;

        const bool performUpdate = (buildFlags & rt::AccelStructBuildFlags::PerformUpdate) != rt::AccelStructBuildFlags::None;
        rt::AccelStructBuildFlags effectiveFlags = buildFlags;
        if (as->allowUpdate)
            effectiveFlags = effectiveFlags | rt::AccelStructBuildFlags::AllowUpdate;

        MTLAccelerationStructureDescriptor* descriptor = buildDescriptorForBLAS(pGeometries, numGeometries, effectiveFlags);

        std::vector<AccelStruct::GeometryState> geometry;
        std::vector<id<MTLBuffer>> geometryBuffers;
        captureGeometryState(pGeometries, numGeometries, geometry, geometryBuffers);

        const bool layoutMatches = as->built && as->builtUsage == descriptor.usage &&
            geometryLayoutMatches(as->builtGeometry, geometry);
        const MTLAccelerationStructureSizes sizes = layoutMatches ? as->builtSizes :
            [m_Context.device accelerationStructureSizesWithDescriptor:descriptor];

        if (performUpdate && as->allowUpdate && layoutMatches && !as->allowCompaction &&
            sizes.accelerationStructureSize <= as->accelStruct.size)
        {
            if (@available(macOS 13.0, *))
            {
                TransientBufferAllocation refitScratch = m_TransientIndirectResources.allocatePrivate(
                    std::max<NSUInteger>(sizes.refitScratchBufferSize, 256), 256);
                if (!refitScratch.buffer)
                {
                    m_RecordingFailed = true;
                    m_Context.error("[nvrhi] Failed to allocate Metal BLAS refit scratch buffer.");
                    return;
                }

                id<MTLAccelerationStructureCommandEncoder> encoder = getOrCreateAccelEncoder("accelerationStructures");
                if (!encoder)
                    return;
                [encoder refitAccelerationStructure:as->accelStruct
                                         descriptor:descriptor
                                        destination:as->accelStruct
                                      scratchBuffer:refitScratch.buffer
                                scratchBufferOffset:refitScratch.offset
                                            options:MTLAccelerationStructureRefitOptionVertexData];
                as->builtGeometry = std::move(geometry);
                as->builtGeometryBuffers = std::move(geometryBuffers);
                m_ReferencedNativeResources.push_back(as->accelStruct);
                return;
            }
        }

        if (sizes.accelerationStructureSize > as->accelStruct.size)
        {
            id<MTLAccelerationStructure> replacement =
                [m_Context.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            if (!replacement)
            {
                m_RecordingFailed = true;
                m_Context.error("[nvrhi] Failed to resize Metal BLAS.");
                return;
            }
            if (!as->desc.debugName.empty())
                replacement.label = [NSString stringWithUTF8String:as->desc.debugName.c_str()];
            as->residency->remove(as->accelStruct);
            as->accelStruct = replacement;
            as->residency->add(replacement);
            resetUpdateState(as);
            as->compacted = false;
        }

        TransientBufferAllocation scratch = m_TransientIndirectResources.allocatePrivate(
            NSUInteger(sizes.buildScratchBufferSize), 256);
        if (!scratch.buffer)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] Failed to allocate Metal BLAS scratch buffer.");
            return;
        }

        id<MTLAccelerationStructureCommandEncoder> encoder = getOrCreateAccelEncoder("accelerationStructures");
        if (!encoder)
            return;
        [encoder buildAccelerationStructure:as->accelStruct
                                 descriptor:descriptor
                              scratchBuffer:scratch.buffer
                        scratchBufferOffset:scratch.offset];
        if (as->allowCompaction)
        {
            id<MTLBuffer> compactedSizeBuffer = [m_Context.device newBufferWithLength:sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
            if (compactedSizeBuffer)
            {
                [encoder writeCompactedAccelerationStructureSize:as->accelStruct
                                                       toBuffer:compactedSizeBuffer
                                                         offset:0];
                m_ReferencedNativeBuffers.push_back(compactedSizeBuffer);
                ++as->buildSerial;
                m_Device->queueCompaction(as, compactedSizeBuffer, trackedCmdBuffer);
            }
        }
        else
            ++as->buildSerial;

        as->builtGeometry = std::move(geometry);
        as->builtGeometryBuffers = std::move(geometryBuffers);
        as->builtUsage = descriptor.usage;
        as->builtSizes = sizes;
        as->built = true;
        m_ReferencedNativeResources.push_back(as->accelStruct);
    }

    void CommandList::compactBottomLevelAccelStructs()
    {
        std::vector<Device::PendingCompaction> completed;
        m_Device->takeCompletedCompactions(completed);
        if (completed.empty())
            return;

        for (Device::PendingCompaction& pending : completed)
        {
            AccelStruct* as = pending.accelStruct;
            if (!as || !as->accelStruct || as->compacted || as->buildSerial != pending.buildSerial || !pending.sizeBuffer)
                continue;
            const uint32_t compactedSize = *static_cast<const uint32_t*>(pending.sizeBuffer.contents);
            if (compactedSize == 0 || NSUInteger(compactedSize) >= as->accelStruct.size)
                continue;
            id<MTLAccelerationStructure> compact = [m_Context.device newAccelerationStructureWithSize:compactedSize];
            if (!compact)
                continue;
            if (!as->desc.debugName.empty())
                compact.label = [NSString stringWithUTF8String:as->desc.debugName.c_str()];
            id<MTLAccelerationStructureCommandEncoder> encoder = getOrCreateAccelEncoder("accelerationStructures");
            if (!encoder)
                return;
            [encoder copyAndCompactAccelerationStructure:as->accelStruct toAccelerationStructure:compact];
            m_ReferencedNativeResources.push_back(as->accelStruct);
            m_ReferencedNativeResources.push_back(compact);
            as->residency->remove(as->accelStruct);
            as->accelStruct = compact;
            as->residency->add(compact);
            as->compacted = true;
        }
    }

    void CommandList::buildTopLevelAccelStruct(rt::IAccelStruct* abstractAs,
        const rt::InstanceDesc* pInstances, size_t numInstances,
        rt::AccelStructBuildFlags buildFlags)
    {
        auto* as = static_cast<AccelStruct*>(abstractAs);
        if (!as || !as->accelStruct || !pInstances || numInstances == 0)
            return;

        const bool performUpdate = (buildFlags & rt::AccelStructBuildFlags::PerformUpdate) != rt::AccelStructBuildFlags::None;
        rt::AccelStructBuildFlags effectiveFlags = buildFlags;
        if (as->allowUpdate)
            effectiveFlags = effectiveFlags | rt::AccelStructBuildFlags::AllowUpdate;

        NSMutableArray<id<MTLAccelerationStructure>>* blasArray = [NSMutableArray arrayWithCapacity:numInstances];
        std::vector<id<MTLAccelerationStructure>> blasReferences;
        std::unordered_map<const void*, uint32_t> blasIndices;

        for (size_t i = 0; i < numInstances; ++i)
        {
            auto* blas = static_cast<AccelStruct*>(pInstances[i].bottomLevelAS);
            if (!blas || !blas->accelStruct)
                continue;
            const void* key = (__bridge const void*)blas->accelStruct;
            if (blasIndices.find(key) == blasIndices.end())
            {
                blasIndices[key] = uint32_t([blasArray count]);
                [blasArray addObject:blas->accelStruct];
                blasReferences.push_back(blas->accelStruct);
            }
        }

        size_t instanceBufferSize = numInstances * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
        TransientBufferAllocation instanceAlloc = m_TransientIndirectResources.allocateShared(
            NSUInteger(instanceBufferSize), 64);
        if (!instanceAlloc.buffer || !instanceAlloc.cpuAddress)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] Failed to allocate Metal TLAS instance buffer.");
            return;
        }

        auto* mtlInstances = reinterpret_cast<MTLAccelerationStructureUserIDInstanceDescriptor*>(instanceAlloc.cpuAddress);
        std::memset(mtlInstances, 0, instanceBufferSize);
        for (size_t i = 0; i < numInstances; ++i)
        {
            const rt::InstanceDesc& src = pInstances[i];
            MTLAccelerationStructureUserIDInstanceDescriptor& dst = mtlInstances[i];
            for (int col = 0; col < 4; ++col)
                for (int row = 0; row < 3; ++row)
                    dst.transformationMatrix.columns[col][row] = src.transform[row * 4 + col];
            dst.mask = src.instanceMask;
            dst.options = MTLAccelerationStructureInstanceOptionNone;
            if ((src.flags & rt::InstanceFlags::TriangleCullDisable) != rt::InstanceFlags::None)
                dst.options |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
            if ((src.flags & rt::InstanceFlags::TriangleFrontCounterclockwise) != rt::InstanceFlags::None)
                dst.options |= MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
            if ((src.flags & rt::InstanceFlags::ForceOpaque) != rt::InstanceFlags::None)
                dst.options |= MTLAccelerationStructureInstanceOptionOpaque;
            if ((src.flags & rt::InstanceFlags::ForceNonOpaque) != rt::InstanceFlags::None)
                dst.options |= MTLAccelerationStructureInstanceOptionNonOpaque;
            auto* blas = static_cast<AccelStruct*>(src.bottomLevelAS);
            const void* key = blas ? (__bridge const void*)blas->accelStruct : nullptr;
            auto it = key ? blasIndices.find(key) : blasIndices.end();
            dst.accelerationStructureIndex = it != blasIndices.end() ? it->second : 0;
            dst.intersectionFunctionTableOffset = 0;
            dst.userID = src.instanceID;
        }

        MTLInstanceAccelerationStructureDescriptor* tlasDesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
        tlasDesc.instanceCount = numInstances;
        tlasDesc.instanceDescriptorBuffer = instanceAlloc.buffer;
        tlasDesc.instanceDescriptorBufferOffset = instanceAlloc.offset;
        tlasDesc.instancedAccelerationStructures = blasArray;
        tlasDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
        tlasDesc.usage = convertBuildFlags(effectiveFlags);

        MTLAccelerationStructureSizes sizes = [m_Context.device accelerationStructureSizesWithDescriptor:tlasDesc];

        bool storageReplaced = false;
        if (sizes.accelerationStructureSize > as->accelStruct.size)
        {
            id<MTLAccelerationStructure> replacement =
                [m_Context.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            if (!replacement)
            {
                m_RecordingFailed = true;
                m_Context.error("[nvrhi] Failed to resize Metal TLAS.");
                return;
            }
            if (!as->desc.debugName.empty())
                replacement.label = [NSString stringWithUTF8String:as->desc.debugName.c_str()];
            as->residency->remove(as->accelStruct);
            as->accelStruct = replacement;
            as->residency->add(replacement);
            resetUpdateState(as);
            storageReplaced = true;
        }

        bool refitted = false;
        if (performUpdate && as->allowUpdate && !storageReplaced && as->built &&
            as->builtUsage == tlasDesc.usage && as->builtInstanceCount == numInstances &&
            blasArrayMatches(as->builtBottomLevelAS, blasReferences) &&
            instanceLayoutMatches(as->builtInstances, mtlInstances, numInstances))
        {
            if (@available(macOS 13.0, *))
            {
                TransientBufferAllocation refitScratch = m_TransientIndirectResources.allocatePrivate(
                    std::max<NSUInteger>(sizes.refitScratchBufferSize, 256), 256);
                if (!refitScratch.buffer)
                {
                    m_RecordingFailed = true;
                    m_Context.error("[nvrhi] Failed to allocate Metal TLAS refit scratch buffer.");
                    return;
                }

                id<MTLAccelerationStructureCommandEncoder> encoder = getOrCreateAccelEncoder("accelerationStructures");
                if (!encoder)
                    return;
                [encoder refitAccelerationStructure:as->accelStruct
                                         descriptor:tlasDesc
                                        destination:as->accelStruct
                                      scratchBuffer:refitScratch.buffer
                                scratchBufferOffset:refitScratch.offset
                                            options:static_cast<MTLAccelerationStructureRefitOptions>(0)];
                refitted = true;
            }
        }

        if (!refitted)
        {
            TransientBufferAllocation scratch = m_TransientIndirectResources.allocatePrivate(
                NSUInteger(sizes.buildScratchBufferSize), 256);
            if (!scratch.buffer)
            {
                m_RecordingFailed = true;
                m_Context.error("[nvrhi] Failed to allocate Metal TLAS scratch buffer.");
                return;
            }

            id<MTLAccelerationStructureCommandEncoder> encoder = getOrCreateAccelEncoder("accelerationStructures");
            if (!encoder)
                return;
            [encoder buildAccelerationStructure:as->accelStruct
                                     descriptor:tlasDesc
                                  scratchBuffer:scratch.buffer
                            scratchBufferOffset:scratch.offset];

            as->builtBottomLevelAS = blasReferences;
            as->builtInstanceCount = numInstances;
            as->builtUsage = tlasDesc.usage;
            as->builtInstances.assign(mtlInstances, mtlInstances + numInstances);
            as->built = true;
        }

        uint32_t instanceCount32 = uint32_t(numInstances);
        std::vector<uint32_t> instanceContributions(numInstances, 0);
        for (size_t i = 0; i < numInstances; ++i)
            instanceContributions[i] = pInstances[i].instanceContributionToHitGroupIndex;

        NSUInteger icBufferSize = std::max<NSUInteger>(numInstances * sizeof(uint32_t), 4);
        if (!as->instanceContributionsBuffer || as->instanceContributionsBuffer.length < icBufferSize)
        {
            if (as->instanceContributionsBuffer)
                as->residency->remove(as->instanceContributionsBuffer);
            as->instanceContributionsBuffer = [m_Context.device newBufferWithLength:icBufferSize
                options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            as->residency->add(as->instanceContributionsBuffer);
        }

        NSUInteger headerSize = sizeof(IRRaytracingAccelerationStructureGPUHeader);
        if (!as->gpuHeaderBuffer)
        {
            as->gpuHeaderBuffer = [m_Context.device newBufferWithLength:headerSize
                options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            as->residency->add(as->gpuHeaderBuffer);
        }

        IRRaytracingSetAccelerationStructure(
            static_cast<uint8_t*>(as->gpuHeaderBuffer.contents),
            as->accelStruct.gpuResourceID,
            static_cast<uint8_t*>(as->instanceContributionsBuffer.contents),
            as->instanceContributionsBuffer.gpuAddress,
            instanceContributions.data(),
            instanceCount32);

        m_ReferencedNativeResources.push_back(as->accelStruct);
        m_ReferencedNativeBuffers.push_back(instanceAlloc.buffer);
        m_ReferencedNativeBuffers.push_back(as->gpuHeaderBuffer);
        m_ReferencedNativeBuffers.push_back(as->instanceContributionsBuffer);
        for (id<MTLAccelerationStructure> blas in blasArray)
            m_ReferencedNativeResources.push_back(blas);
    }

    void CommandList::buildTopLevelAccelStructFromBuffer(rt::IAccelStruct* as,
        nvrhi::IBuffer* instanceBuffer, uint64_t instanceBufferOffset,
        size_t numInstances, rt::AccelStructBuildFlags buildFlags)
    {
        (void)as;
        (void)instanceBuffer;
        (void)instanceBufferOffset;
        (void)numInstances;
        (void)buildFlags;
        unsupported(__func__);
    }

    void CommandList::copyRaytracingAccelerationStructure(rt::IAccelStruct* destination, rt::IAccelStruct* source)
    {
        auto* dst = static_cast<AccelStruct*>(destination);
        auto* src = static_cast<AccelStruct*>(source);
        if (!dst || !src || !dst->accelStruct || !src->accelStruct)
            return;
        id<MTLAccelerationStructureCommandEncoder> encoder = getOrCreateAccelEncoder("accelerationStructures");
        if (!encoder)
            return;
        [encoder copyAccelerationStructure:src->accelStruct toAccelerationStructure:dst->accelStruct];
        resetUpdateState(dst);
        dst->builtGeometryBuffers = src->builtGeometryBuffers;
        dst->builtBottomLevelAS = src->builtBottomLevelAS;
        m_ReferencedNativeResources.push_back(src->accelStruct);
        m_ReferencedNativeResources.push_back(dst->accelStruct);
    }

    void CommandList::setAccelStructState(rt::IAccelStruct* as, ResourceStates stateBits)
    {
        (void)as;
        (void)stateBits;
    }

    rt::PipelineHandle Device::createRayTracingPipeline(const rt::PipelineDesc& desc)
    {
        (void)desc;
        m_Context.unsupported(__func__);
        return nullptr;
    }
}
