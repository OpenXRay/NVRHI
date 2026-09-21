#include "metal3-backend.h"
#include "nvrhi/nvrhi.h"
#include <Metal/Metal.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#define IR_PRIVATE_IMPLEMENTATION
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

namespace nvrhi::metal3 
{
    // taken from <metal_irconverter_runtime/metal_irconverter_runtime.h>
    // predefined bind point values for shaders converted with metal shader converter
    static constexpr uint32_t c_MscVertexBufferBindPoint = 6;
    static constexpr uint32_t c_IrDrawArgumentsBindPoint = 4;
    static constexpr uint32_t c_IrRuntimeVertexBufferCount = 31;
    static constexpr size_t c_ArgumentTablePageSize = 1024 * 1024;
    static constexpr size_t c_ArgumentTableInitialPageCount = 3;

    struct IcbIndexedIndirectParams
    {
        uint32_t maxDrawCount = 0;
        uint32_t paramOffsetBytes = 0;
        uint32_t countOffsetBytes = 0;
        uint32_t indexType = 0;
        uint32_t indexBufferOffset = 0;
        uint32_t primitiveType = 0;
        uint32_t drawArgumentsBindPoint = c_IrDrawArgumentsBindPoint;
        uint32_t bufferBindingCount = 0;
    };

    struct IcbGraphicsBufferBinding
    {
        uint64_t address = 0;
        uint32_t index = 0;
        uint32_t fragmentStage = 0;
    };

    struct GeometryIndirectParams
    {
        uint32_t drawCount = 0;
        uint32_t paramOffsetBytes = 0;
        uint32_t primitiveType = 0;
        uint32_t gsVertexSizeInBytes = 0;
        uint32_t gsMaxInputPrimitivesPerMeshThreadgroup = 0;
        uint32_t indexed = 0;
        uint32_t indexType = 0;
        uint32_t indexBufferOffsetInElements = 0;
        uint64_t indexBufferAddress = 0;
    };

    struct GeometryIndirectFillState
    {
        id<MTLComputePipelineState> pipeline = nil;
    };

    struct GeometryIndexedIndirectCountParams
    {
        uint32_t maxDrawCount = 0;
        uint32_t paramOffsetBytes = 0;
        uint32_t countOffsetBytes = 0;
        uint32_t primitiveType = 0;
        uint32_t gsVertexSizeInBytes = 0;
        uint32_t gsMaxInputPrimitivesPerMeshThreadgroup = 0;
        uint32_t indexType = 0;
        uint32_t indexBufferOffsetInElements = 0;
        uint64_t indexBufferAddress = 0;
        uint32_t hasObjectArgumentTable = 0;
        uint32_t hasMeshArgumentTable = 0;
        uint32_t hasFragmentArgumentTable = 0;
        uint32_t pad = 0;
    };

    struct GeometryIndexedIndirectCountIcbFillState
    {
        id<MTLComputePipelineState> pipeline = nil;
        id<MTLArgumentEncoder> icbEncoder = nil;
    };

    static const GeometryIndirectFillState& getGeometryIndirectFillState(const MTL3Context& context)
    {
        static std::mutex mutex;
        static GeometryIndirectFillState state;
        static bool attempted = false;

        std::lock_guard<std::mutex> lock(mutex);
        if (attempted)
            return state;
        attempted = true;

        static constexpr const char* source = R"(
            #include <metal_stdlib>
            using namespace metal;

            struct DrawIndirectArguments
            {
                uint vertexCount;
                uint instanceCount;
                uint startVertexLocation;
                uint startInstanceLocation;
            };

            struct DrawIndexedIndirectArguments
            {
                uint indexCount;
                uint instanceCount;
                uint startIndexLocation;
                int baseVertexLocation;
                uint startInstanceLocation;
            };

            // CPU-written constants shared by the non-indexed and indexed paths.
            // They describe how to read NVRHI's D3D-style indirect args and how
            // to configure IRRuntime geometry shader emulation for each draw.
            struct GeometryIndirectParams
            {
                uint drawCount;
                uint paramOffsetBytes;
                uint primitiveType;
                uint gsVertexSizeInBytes;
                uint gsMaxInputPrimitivesPerMeshThreadgroup;
                uint indexed;
                uint indexType;
                uint indexBufferOffsetInElements;
                ulong indexBufferAddress;
            };

            struct IRRuntimeDrawArgument
            {
                uint vertexCountPerInstance;
                uint instanceCount;
                uint startVertexLocation;
                uint startInstanceLocation;
            };

            struct IRRuntimeDrawIndexedArgument
            {
                uint indexCountPerInstance;
                uint instanceCount;
                uint startIndexLocation;
                int baseVertexLocation;
                uint startInstanceLocation;
            };

            union IRRuntimeDrawParams
            {
                IRRuntimeDrawArgument draw;
                IRRuntimeDrawIndexedArgument drawIndexed;
            };

            // Per-draw state consumed by the generated object/mesh shader pair.
            // This is normally prepared by IRRuntime direct-draw wrappers, but
            // indirect draws need a GPU pass because the draw args live on GPU.
            struct IRRuntimeDrawInfo
            {
                ushort indexType;
                uchar primitiveTopology;
                uchar threadsPerPatch;
                ushort maxInputPrimitivesPerMeshThreadgroup;
                ushort objectThreadgroupVertexStride;
                ushort meshThreadgroupPrimitiveStride;
                ushort gsInstanceCount;
                ushort patchesPerObjectThreadgroup;
                ushort inputControlPointsPerPatch;
                ulong indexBuffer;
            };

            // Metal mesh indirect arguments. The helper writes one of these per
            // NVRHI indirect draw so the render pass can call
            // drawMeshThreadgroupsWithIndirectBuffer for each generated entry.
            struct MeshThreadgroupsIndirectArguments
            {
                uint threadgroupsPerGrid[3];
            };

            static uint primitive_vertex_count(uint primitiveType)
            {
                switch (primitiveType)
                {
                case 0: return 1;
                case 1: return 2;
                case 2: return 2;
                case 3: return 3;
                case 4: return 3;
                case 5: return 4;
                case 6: return 6;
                case 7: return 4;
                default: return 0;
                }
            }

            static uint primitive_vertex_overlap(uint primitiveType)
            {
                switch (primitiveType)
                {
                case 2: return 1;
                case 4: return 2;
                case 7: return 3;
                default: return 0;
                }
            }

            static uint min_nonzero(uint a, uint b)
            {
                return a < b ? a : b;
            }

            static IRRuntimeDrawInfo calculate_draw_info(
                uint primitiveType,
                uint gsVertexSizeInBytes,
                uint maxInputPrimitivesPerMeshThreadgroup,
                uint instanceCount)
            {
                const uint primitiveVertexCount = primitive_vertex_count(primitiveType);
                const uint alignment = primitiveVertexCount;
                const uint totalPayloadBytes = 16384;
                const uint payloadBytesForMetadata = 32;
                const uint payloadBytesForVertexData = totalPayloadBytes - payloadBytesForMetadata;
                const uint maxVertexCountLimitedByPayloadMemory =
                    (((payloadBytesForVertexData / gsVertexSizeInBytes)) / alignment) * alignment;
                const uint maxMeshThreadgroupsPerObjectThreadgroup = 1024;
                const uint maxPrimCountLimitedByAmplificationRate =
                    maxMeshThreadgroupsPerObjectThreadgroup * maxInputPrimitivesPerMeshThreadgroup;
                uint maxPrimsPerObjectThreadgroup =
                    min_nonzero(maxVertexCountLimitedByPayloadMemory / primitiveVertexCount,
                        maxPrimCountLimitedByAmplificationRate);
                const uint maxThreadsPerThreadgroup = 256;
                maxPrimsPerObjectThreadgroup =
                    min_nonzero(maxPrimsPerObjectThreadgroup, maxThreadsPerThreadgroup / primitiveVertexCount);

                IRRuntimeDrawInfo info;
                info.indexType = 0;
                info.primitiveTopology = uchar(primitiveType);
                info.threadsPerPatch = uchar(primitiveVertexCount);
                info.maxInputPrimitivesPerMeshThreadgroup = ushort(maxInputPrimitivesPerMeshThreadgroup);
                info.objectThreadgroupVertexStride = ushort(maxPrimsPerObjectThreadgroup * primitiveVertexCount);
                info.meshThreadgroupPrimitiveStride = ushort(maxInputPrimitivesPerMeshThreadgroup);
                info.gsInstanceCount = ushort(instanceCount);
                info.patchesPerObjectThreadgroup = ushort(maxPrimsPerObjectThreadgroup);
                info.inputControlPointsPerPatch = ushort(primitiveVertexCount);
                info.indexBuffer = 0;
                return info;
            }

            static uint3 calculate_object_threadgroup_count(
                uint vertexCount,
                ushort objectThreadgroupVertexStride,
                uint primitiveType,
                uint instanceCount)
            {
                const uint overlap = primitive_vertex_overlap(primitiveType);
                if (vertexCount <= overlap || instanceCount == 0 || objectThreadgroupVertexStride == 0)
                    return uint3(0, 0, 0);

                const uint width =
                    (vertexCount - overlap + uint(objectThreadgroupVertexStride) - 1) /
                    uint(objectThreadgroupVertexStride);
                return uint3(width, instanceCount, 1);
            }

            kernel void nvrhi_metal3_fill_geometry_indirect(
                device const uchar* indirectParams [[buffer(0)]],
                device IRRuntimeDrawInfo* drawInfos [[buffer(1)]],
                device IRRuntimeDrawParams* drawParams [[buffer(2)]],
                device MeshThreadgroupsIndirectArguments* meshIndirectArgs [[buffer(3)]],
                constant GeometryIndirectParams& params [[buffer(4)]],
                uint tid [[thread_position_in_grid]])
            {
                if (tid >= params.drawCount)
                    return;

                // Pick the right NVRHI argument layout, copy its draw fields into
                // the IRRuntime draw-param layout, and remember the vertex/index
                // count used to size the object-stage work.
                const device uchar* drawArgs = indirectParams + params.paramOffsetBytes;
                uint vertexOrIndexCount = 0;
                uint instanceCount = 0;

                IRRuntimeDrawParams runtimeParams;
                if (params.indexed != 0)
                {
                    device const DrawIndexedIndirectArguments* args =
                        reinterpret_cast<device const DrawIndexedIndirectArguments*>(drawArgs);
                    const DrawIndexedIndirectArguments a = args[tid];

                    vertexOrIndexCount = a.indexCount;
                    instanceCount = a.instanceCount;
                    runtimeParams.drawIndexed.indexCountPerInstance = a.indexCount;
                    runtimeParams.drawIndexed.instanceCount = a.instanceCount;
                    runtimeParams.drawIndexed.startIndexLocation =
                        params.indexBufferOffsetInElements + a.startIndexLocation;
                    runtimeParams.drawIndexed.baseVertexLocation = a.baseVertexLocation;
                    runtimeParams.drawIndexed.startInstanceLocation = a.startInstanceLocation;
                }
                else
                {
                    device const DrawIndirectArguments* args =
                        reinterpret_cast<device const DrawIndirectArguments*>(drawArgs);
                    const DrawIndirectArguments a = args[tid];

                    vertexOrIndexCount = a.vertexCount;
                    instanceCount = a.instanceCount;
                    runtimeParams.draw.vertexCountPerInstance = a.vertexCount;
                    runtimeParams.draw.instanceCount = a.instanceCount;
                    runtimeParams.draw.startVertexLocation = a.startVertexLocation;
                    runtimeParams.draw.startInstanceLocation = a.startInstanceLocation;
                }

                // Build the per-draw geometry-emulation metadata. Indexed draws
                // also carry the index-buffer address and Metal index type so the
                // generated mesh stage can fetch indices.
                IRRuntimeDrawInfo info = calculate_draw_info(params.primitiveType,
                    params.gsVertexSizeInBytes,
                    params.gsMaxInputPrimitivesPerMeshThreadgroup,
                    instanceCount);
                if (params.indexed != 0)
                {
                    info.indexType = ushort(params.indexType + 1);
                    info.indexBuffer = params.indexBufferAddress;
                }

                // Convert the source vertex/index count into the mesh pipeline's
                // indirect threadgroup dimensions.
                uint3 threadgroups = calculate_object_threadgroup_count(vertexOrIndexCount,
                    info.objectThreadgroupVertexStride,
                    params.primitiveType,
                    instanceCount);

                drawInfos[tid] = info;
                drawParams[tid] = runtimeParams;
                meshIndirectArgs[tid].threadgroupsPerGrid[0] = threadgroups.x;
                meshIndirectArgs[tid].threadgroupsPerGrid[1] = threadgroups.y;
                meshIndirectArgs[tid].threadgroupsPerGrid[2] = threadgroups.z;
            }
        )";

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 13.0, *))
            options.languageVersion = MTLLanguageVersion3_0;

        id<MTLLibrary> library = [context.device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                               options:options
                                                                 error:&error];
        if (!library)
        {
            std::string message = "[metal3] failed to compile geometry-indirect helper";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            context.error(message);
            return state;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"nvrhi_metal3_fill_geometry_indirect"];
        if (!function)
        {
            context.error("[metal3] geometry-indirect helper function missing");
            return state;
        }

        state.pipeline = [context.device newComputePipelineStateWithFunction:function error:&error];
        if (!state.pipeline)
        {
            std::string message = "[metal3] failed to create geometry-indirect helper pipeline";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            context.error(message);
        }

        return state;
    }

    // builds a compute shader used by geometry-emulation
    // drawIndexedIndirectCount. The helper, reads NVRHI's indexed
    // indirect args plus the GPU count buffer, and writes mesh draw commands into
    // an MTLIndirectCommandBuffer. That avoids a CPU readback of the count and
    // lets converted geometry shaders replay through Metal's object/mesh path.
    static const GeometryIndexedIndirectCountIcbFillState& getGeometryIndexedIndirectCountIcbFillState(const MTL3Context& context)
    {
        static std::mutex mutex;
        static GeometryIndexedIndirectCountIcbFillState state;
        static bool attempted = false;

        std::lock_guard<std::mutex> lock(mutex);
        if (attempted)
            return state;
        attempted = true;

        static constexpr const char* source = R"(
            #include <metal_stdlib>
            using namespace metal;

            struct DrawIndexedIndirectArguments
            {
                uint indexCount;
                uint instanceCount;
                uint startIndexLocation;
                int baseVertexLocation;
                uint startInstanceLocation;
            };

            struct GeometryIndexedIndirectCountParams
            {
                uint maxDrawCount;
                uint paramOffsetBytes;
                uint countOffsetBytes;
                uint primitiveType;
                uint gsVertexSizeInBytes;
                uint gsMaxInputPrimitivesPerMeshThreadgroup;
                uint indexType;
                uint indexBufferOffsetInElements;
                ulong indexBufferAddress;
                uint hasObjectArgumentTable;
                uint hasMeshArgumentTable;
                uint hasFragmentArgumentTable;
                uint pad;
            };

            struct IcbArgumentBuffer
            {
                command_buffer icb [[id(0)]];
            };

            struct IRRuntimeDrawIndexedArgument
            {
                uint indexCountPerInstance;
                uint instanceCount;
                uint startIndexLocation;
                int baseVertexLocation;
                uint startInstanceLocation;
            };

            union IRRuntimeDrawParams
            {
                IRRuntimeDrawIndexedArgument drawIndexed;
            };

            struct IRRuntimeDrawInfo
            {
                ushort indexType;
                uchar primitiveTopology;
                uchar threadsPerPatch;
                ushort maxInputPrimitivesPerMeshThreadgroup;
                ushort objectThreadgroupVertexStride;
                ushort meshThreadgroupPrimitiveStride;
                ushort gsInstanceCount;
                ushort patchesPerObjectThreadgroup;
                ushort inputControlPointsPerPatch;
                ulong indexBuffer;
            };

            constant uint kIRArgumentBufferBindPoint = 2;
            constant uint kIRArgumentBufferDrawArgumentsBindPoint = 4;
            constant uint kIRArgumentBufferUniformsBindPoint = 5;

            static uint primitive_vertex_count(uint primitiveType)
            {
                switch (primitiveType)
                {
                case 0: return 1;
                case 1: return 2;
                case 2: return 2;
                case 3: return 3;
                case 4: return 3;
                case 5: return 4;
                case 6: return 6;
                case 7: return 4;
                default: return 0;
                }
            }

            static uint primitive_vertex_overlap(uint primitiveType)
            {
                switch (primitiveType)
                {
                case 2: return 1;
                case 4: return 2;
                case 7: return 3;
                default: return 0;
                }
            }

            static IRRuntimeDrawInfo calculate_draw_info(
                uint primitiveType,
                uint gsVertexSizeInBytes,
                uint maxInputPrimitivesPerMeshThreadgroup,
                uint instanceCount,
                uint indexType,
                ulong indexBufferAddress)
            {
                const uint primitiveVertexCount = primitive_vertex_count(primitiveType);

                IRRuntimeDrawInfo info;
                info.indexType = ushort(indexType + 1);
                info.primitiveTopology = uchar(primitiveType);
                info.threadsPerPatch = uchar(primitiveVertexCount);
                info.maxInputPrimitivesPerMeshThreadgroup = ushort(maxInputPrimitivesPerMeshThreadgroup);
                info.objectThreadgroupVertexStride = 0;
                info.meshThreadgroupPrimitiveStride = ushort(maxInputPrimitivesPerMeshThreadgroup);
                info.gsInstanceCount = ushort(instanceCount);
                info.patchesPerObjectThreadgroup = 0;
                info.inputControlPointsPerPatch = ushort(primitiveVertexCount);
                info.indexBuffer = indexBufferAddress;

                if (primitiveVertexCount == 0 ||
                    gsVertexSizeInBytes == 0 ||
                    maxInputPrimitivesPerMeshThreadgroup == 0)
                {
                    return info;
                }

                const uint alignment = primitiveVertexCount;
                const uint totalPayloadBytes = 16384;
                const uint payloadBytesForMetadata = 32;
                const uint payloadBytesForVertexData = totalPayloadBytes - payloadBytesForMetadata;
                const uint maxVertexCountLimitedByPayloadMemory =
                    (((payloadBytesForVertexData / gsVertexSizeInBytes)) / alignment) * alignment;
                const uint maxMeshThreadgroupsPerObjectThreadgroup = 1024;
                const uint maxPrimCountLimitedByAmplificationRate =
                    maxMeshThreadgroupsPerObjectThreadgroup * maxInputPrimitivesPerMeshThreadgroup;
                uint maxPrimsPerObjectThreadgroup = min(
                    maxVertexCountLimitedByPayloadMemory / primitiveVertexCount,
                    maxPrimCountLimitedByAmplificationRate);
                const uint maxThreadsPerThreadgroup = 256;
                maxPrimsPerObjectThreadgroup = min(
                    maxPrimsPerObjectThreadgroup,
                    maxThreadsPerThreadgroup / primitiveVertexCount);

                info.objectThreadgroupVertexStride = ushort(maxPrimsPerObjectThreadgroup * primitiveVertexCount);
                info.patchesPerObjectThreadgroup = ushort(maxPrimsPerObjectThreadgroup);
                return info;
            }

            static uint3 calculate_object_threadgroup_count(
                uint indexCount,
                ushort objectThreadgroupVertexStride,
                uint primitiveType,
                uint instanceCount)
            {
                const uint overlap = primitive_vertex_overlap(primitiveType);
                if (indexCount <= overlap || instanceCount == 0 || objectThreadgroupVertexStride == 0)
                    return uint3(0, 0, 0);

                const uint width =
                    (indexCount - overlap + uint(objectThreadgroupVertexStride) - 1) /
                    uint(objectThreadgroupVertexStride);
                return uint3(width, instanceCount, 1);
            }

            kernel void nvrhi_metal3_fill_geometry_indexed_indirect_count_icb(
                device const uchar* indirectParams [[buffer(0)]],
                device const uchar* rawCount [[buffer(1)]],
                device uint2* executionRange [[buffer(2)]],
                constant IcbArgumentBuffer& icbArgumentBuffer [[buffer(3)]],
                device IRRuntimeDrawInfo* drawInfos [[buffer(4)]],
                device IRRuntimeDrawParams* drawParams [[buffer(5)]],
                constant GeometryIndexedIndirectCountParams& params [[buffer(6)]],
                device const uchar* vertexBufferTable [[buffer(7)]],
                device const uchar* objectArgumentTable [[buffer(8)]],
                device const uchar* meshArgumentTable [[buffer(9)]],
                device const uchar* fragmentArgumentTable [[buffer(10)]],
                uint tid [[thread_position_in_grid]])
            {
                device const uint* countPtr = reinterpret_cast<device const uint*>(rawCount + params.countOffsetBytes);
                const uint gpuCount = min(*countPtr, params.maxDrawCount);

                // Metal executes ICB commands through a GPU-readable execution
                // range. Thread 0 converts NVRHI's count buffer into that range,
                // keeping the whole counted draw on GPU without a readback.
                if (tid == 0)
                    executionRange[0] = uint2(0, gpuCount);

                if (tid >= gpuCount)
                    return;

                // Each live thread owns one D3D/NVRHI indexed indirect record.
                // The helper expands that compact draw record into the two
                // IRRuntime records consumed by the generated object/mesh stages.
                device const DrawIndexedIndirectArguments* args =
                    reinterpret_cast<device const DrawIndexedIndirectArguments*>(indirectParams + params.paramOffsetBytes);
                const DrawIndexedIndirectArguments a = args[tid];

                IRRuntimeDrawInfo info = calculate_draw_info(params.primitiveType,
                    params.gsVertexSizeInBytes,
                    params.gsMaxInputPrimitivesPerMeshThreadgroup,
                    a.instanceCount,
                    params.indexType,
                    params.indexBufferAddress);

                IRRuntimeDrawParams runtimeParams;
                runtimeParams.drawIndexed.indexCountPerInstance = a.indexCount;
                runtimeParams.drawIndexed.instanceCount = a.instanceCount;
                runtimeParams.drawIndexed.startIndexLocation =
                    params.indexBufferOffsetInElements + a.startIndexLocation;
                runtimeParams.drawIndexed.baseVertexLocation = a.baseVertexLocation;
                runtimeParams.drawIndexed.startInstanceLocation = a.startInstanceLocation;

                drawInfos[tid] = info;
                drawParams[tid] = runtimeParams;

                // Zero-sized draws still have valid runtime records, but do not
                // need an ICB command because there is no mesh work to execute.
                if (a.indexCount == 0 || a.instanceCount == 0)
                    return;

                const uint3 threadgroups = calculate_object_threadgroup_count(a.indexCount,
                    info.objectThreadgroupVertexStride,
                    params.primitiveType,
                    a.instanceCount);
                if (threadgroups.x == 0 || threadgroups.y == 0)
                    return;

                device const uchar* drawInfoBytes =
                    reinterpret_cast<device const uchar*>(&drawInfos[tid]);
                device const uchar* drawParamsBytes =
                    reinterpret_cast<device const uchar*>(&drawParams[tid]);

                // Geometry-emulation mesh shaders read the IR runtime draw
                // metadata from object and mesh stages, matching the direct
                // IRRuntimeDrawIndexedPrimitivesGeometryEmulation path.
                // Because ICB commands do not inherit buffers in this path, the
                // compute shader writes every table binding the replayed mesh draw
                // will need: original vertex buffers, reflected argument tables,
                // and the per-draw IRRuntime records generated above.
                render_command command(icbArgumentBuffer.icb, tid);
                command.set_object_buffer(vertexBufferTable, 0);
                if (params.hasObjectArgumentTable != 0)
                    command.set_object_buffer(objectArgumentTable, kIRArgumentBufferBindPoint);
                if (params.hasMeshArgumentTable != 0)
                    command.set_mesh_buffer(meshArgumentTable, kIRArgumentBufferBindPoint);
                if (params.hasFragmentArgumentTable != 0)
                    command.set_fragment_buffer(fragmentArgumentTable, kIRArgumentBufferBindPoint);
                command.set_object_buffer(drawInfoBytes, kIRArgumentBufferUniformsBindPoint);
                command.set_mesh_buffer(drawInfoBytes, kIRArgumentBufferUniformsBindPoint);
                command.set_object_buffer(drawParamsBytes, kIRArgumentBufferDrawArgumentsBindPoint);
                command.set_mesh_buffer(drawParamsBytes, kIRArgumentBufferDrawArgumentsBindPoint);
                command.draw_mesh_threadgroups(threadgroups,
                    uint3(info.objectThreadgroupVertexStride + primitive_vertex_overlap(params.primitiveType), 1, 1),
                    uint3(params.gsMaxInputPrimitivesPerMeshThreadgroup, 1, 1));
            }
        )";

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 14.0, *))
            options.languageVersion = MTLLanguageVersion3_1;

        id<MTLLibrary> library = [context.device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                               options:options
                                                                 error:&error];
        if (!library)
        {
            std::string message = "[metal3] failed to compile geometry counted indexed-indirect ICB helper";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            context.error(message);
            return state;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"nvrhi_metal3_fill_geometry_indexed_indirect_count_icb"];
        if (!function)
        {
            context.error("[metal3] geometry counted indexed-indirect ICB helper function missing");
            return state;
        }

        // The ICB is passed to the compute shader through an argument buffer;
        // cache the encoder so each draw call only has to encode the ICB object.
        state.icbEncoder = [function newArgumentEncoderWithBufferIndex:3];
        if (!state.icbEncoder)
        {
            context.error("[metal3] failed to create geometry counted indexed-indirect ICB argument encoder");
            return state;
        }

        if (@available(macOS 11.0, *))
        {
            MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function;
            // Required because this compute pipeline writes render mesh commands
            // into the ICB instead of only producing ordinary buffer data.
            descriptor.supportIndirectCommandBuffers = YES;
            state.pipeline = [context.device newComputePipelineStateWithDescriptor:descriptor
                                                                           options:MTLPipelineOptionNone
                                                                        reflection:nil
                                                                             error:&error];
        }
        else
        {
            state.pipeline = [context.device newComputePipelineStateWithFunction:function error:&error];
        }

        if (!state.pipeline)
        {
            std::string message = "[metal3] failed to create geometry counted indexed-indirect ICB helper pipeline";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            context.error(message);
        }

        return state;
    }

    /*
    // helper used by drawIndexedIndirectCount. Metal cannot
    // execute a counted indexed-indirect draw directly, so a compute shader
    // converts the NVRHI args/count buffers into an MTLIndirectCommandBuffer.
    // The pipeline runs that compute shader; icbEncoder encodes the ICB object
    // into the helper shader's argument buffer.
    */
    struct IndexedIndirectIcbFillState
    {
        id<MTLComputePipelineState> pipeline = nil;
        id<MTLArgumentEncoder> icbEncoder = nil;
    };

    // Compile and cache the ICB-fill compute pipeline once. The returned state
    // stays empty if compilation fails, and callers simply skip counted indirect
    // draws rather than trying to fall back to a CPU readback of the count.
    static const IndexedIndirectIcbFillState& getIndexedIndirectIcbFillState(const MTL3Context& context)
    {
        static std::mutex mutex;
        static IndexedIndirectIcbFillState state;
        static bool attempted = false;

        std::lock_guard<std::mutex> lock(mutex);
        if (attempted)
            return state;
        attempted = true;

        static constexpr const char* source = R"(
            #include <metal_stdlib>
            using namespace metal;

            struct DrawIndexedIndirectArguments
            {
                uint indexCount;
                uint instanceCount;
                uint startIndexLocation;
                int baseVertexLocation;
                uint startInstanceLocation;
            };

            struct IcbIndexedIndirectParams
            {
                uint maxDrawCount;
                uint paramOffsetBytes;
                uint countOffsetBytes;
                uint indexType;
                uint indexBufferOffset;
                uint primitiveType;
                uint drawArgumentsBindPoint;
                uint bufferBindingCount;
            };

            struct IcbGraphicsBufferBinding
            {
                device const uchar* address;
                uint index;
                uint fragmentStage;
            };

            struct IcbArgumentBuffer
            {
                command_buffer icb [[id(0)]];
            };

            static primitive_type to_primitive_type(uint value)
            {
                switch (value)
                {
                case 0: return primitive_type::point;
                case 1: return primitive_type::line;
                case 2: return primitive_type::line_strip;
                case 4: return primitive_type::triangle_strip;
                default: return primitive_type::triangle;
                }
            }

            kernel void nvrhi_metal3_fill_indexed_indirect_icb(
                device const uchar* indirectParams [[buffer(0)]],
                device const uchar* rawCount [[buffer(1)]],
                device uint2* executionRange [[buffer(2)]],
                constant IcbArgumentBuffer& icbArgumentBuffer [[buffer(3)]],
                device const uchar* indexBuffer [[buffer(4)]],
                constant IcbIndexedIndirectParams& params [[buffer(5)]],
                device const IcbGraphicsBufferBinding* bufferBindings [[buffer(6)]],
                    uint tid [[thread_position_in_grid]])
            {
                device const uint* countPtr = reinterpret_cast<device const uint*>(rawCount + params.countOffsetBytes);
                const uint gpuCount = min(*countPtr, params.maxDrawCount);

                // Metal executes an ICB over an explicit range. write that range from
                // the GPU count buffer so the render pass never needs a CPU readback.
                if (tid == 0)
                    executionRange[0] = uint2(0, gpuCount);

                if (tid >= gpuCount)
                    return;

                device const DrawIndexedIndirectArguments* args =
                    reinterpret_cast<device const DrawIndexedIndirectArguments*>(indirectParams + params.paramOffsetBytes);
                const DrawIndexedIndirectArguments a = args[tid];

                if (a.indexCount == 0 || a.instanceCount == 0)
                    return;

                // One compute thread writes one render command into the same slot as
                // its source indirect argument.
                render_command command(icbArgumentBuffer.icb, tid);
                for (uint i = 0; i < params.bufferBindingCount; ++i)
                {
                    const IcbGraphicsBufferBinding binding = bufferBindings[i];
                    if (binding.fragmentStage)
                        command.set_fragment_buffer(binding.address, binding.index);
                    else
                        command.set_vertex_buffer(binding.address, binding.index);
                }

                // Metal Shader Converter reads D3D draw parameters from this runtime
                // bind point. Point every ICB command at its own source indirect args.
                device const uchar* drawArgs =
                    indirectParams + params.paramOffsetBytes + tid * uint(sizeof(DrawIndexedIndirectArguments));
                command.set_vertex_buffer(drawArgs, params.drawArgumentsBindPoint);

                if (params.indexType == 1)
                {
                    device const ushort* indices =
                        reinterpret_cast<device const ushort*>(indexBuffer + params.indexBufferOffset) + a.startIndexLocation;
                    command.draw_indexed_primitives(to_primitive_type(params.primitiveType),
                        a.indexCount,
                        indices,
                        a.instanceCount,
                        uint(a.baseVertexLocation),
                        a.startInstanceLocation);
                }
                else
                {
                    device const uint* indices =
                        reinterpret_cast<device const uint*>(indexBuffer + params.indexBufferOffset) + a.startIndexLocation;
                    command.draw_indexed_primitives(to_primitive_type(params.primitiveType),
                        a.indexCount,
                        indices,
                        a.instanceCount,
                        uint(a.baseVertexLocation),
                        a.startInstanceLocation);
                }
            }
        )";

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 13.0, *))
            options.languageVersion = MTLLanguageVersion3_0;

        id<MTLLibrary> library = [context.device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                               options:options
                                                                 error:&error];
        if (!library)
        {
            std::string message = "[metal3] failed to compile indexed-indirect ICB helper";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            context.error(message);
            return state;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"nvrhi_metal3_fill_indexed_indirect_icb"];
        if (!function)
        {
            context.error("[metal3] indexed-indirect ICB helper function missing");
            return state;
        }

        state.icbEncoder = [function newArgumentEncoderWithBufferIndex:3];
        if (!state.icbEncoder)
        {
            context.error("[metal3] failed to create indexed-indirect ICB argument encoder");
            return state;
        }

        if (@available(macOS 11.0, *))
        {
            MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function;
            // Required because this compute pipeline writes render commands into
            // an MTLIndirectCommandBuffer.
            descriptor.supportIndirectCommandBuffers = YES;
            state.pipeline = [context.device newComputePipelineStateWithDescriptor:descriptor
                                                                           options:MTLPipelineOptionNone
                                                                        reflection:nil
                                                                             error:&error];
        }
        else
        {
            state.pipeline = [context.device newComputePipelineStateWithFunction:function error:&error];
        }

        if (!state.pipeline)
        {
            std::string message = "[metal3] failed to create indexed-indirect ICB helper pipeline";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            context.error(message); 
        }
        
        return state;
    }

    bool traceMetalRuntime()
    {
        static bool enabled = [] {
            const char* value = std::getenv("LDV_METAL3_TRACE");
#if defined(NDEBUG)
            return value && std::string(value) != "0";
#else
            return !value || std::string(value) != "0";
#endif
        }();
        return enabled;
    }

    static bool encoderExecutionStatusEnabled()
    {
        static const bool enabled = [] {
            const char* value = std::getenv("LDV_METAL3_ENCODER_STATUS");
            return value ? std::strcmp(value, "0") != 0 : traceMetalRuntime();
        }();
        return enabled;
    }

    static bool argumentTableStatsEnabled()
    {
        static bool enabled = [] {
            const char* value = std::getenv("LDV_METAL3_ARGUMENT_TABLE_STATS");
            return value && std::string(value) != "0";
        }();
        return enabled;
    }

    static bool indirectResourceStatsEnabled()
    {
        static bool enabled = [] {
            const char* value = std::getenv("LDV_METAL3_INDIRECT_RESOURCE_STATS");
            return value && std::string(value) != "0";
        }();
        return enabled;
    }

    static size_t alignUp(size_t value, size_t alignment)
    {
        if (alignment <= 1)
            return value;
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static const char* resourceTypeName(ResourceType type)
    {
        switch (type)
        {
        case ResourceType::Texture_SRV: return "Texture_SRV";
        case ResourceType::Texture_UAV: return "Texture_UAV";
        case ResourceType::TypedBuffer_SRV: return "TypedBuffer_SRV";
        case ResourceType::TypedBuffer_UAV: return "TypedBuffer_UAV";
        case ResourceType::StructuredBuffer_SRV: return "StructuredBuffer_SRV";
        case ResourceType::StructuredBuffer_UAV: return "StructuredBuffer_UAV";
        case ResourceType::RawBuffer_SRV: return "RawBuffer_SRV";
        case ResourceType::RawBuffer_UAV: return "RawBuffer_UAV";
        case ResourceType::ConstantBuffer: return "ConstantBuffer";
        case ResourceType::VolatileConstantBuffer: return "VolatileConstantBuffer";
        case ResourceType::Sampler: return "Sampler";
        case ResourceType::None: return "None";
        default: return "Other";
        }
    }

    static MetalArgumentTableCacheKey makeArgumentTableCacheKey(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets)
    {
        MetalArgumentTableCacheKey key;
        key.plan = &plan;
        key.bindingSetCount = bindingSets.size();

        for (size_t index = 0; index < bindingSets.size(); ++index)
        {
            IBindingSet* bindingSet = bindingSets[index];
            key.bindingSets[index] = bindingSet;
            if (bindingSet && !bindingSet->getDesc())
            {
                auto* table = static_cast<DescriptorTable*>(bindingSet);
                std::lock_guard<std::mutex> lock(table->mutex);
                key.bindingSetVersions[index] = table->version;
            }
            else
                key.bindingSetVersions[index] = bindingSet ? static_cast<BindingSet*>(bindingSet)->version : 0;
        }

        return key;
    }


    static bool isSrvType(ResourceType type)
    {
        return type == ResourceType::Texture_SRV ||
            type == ResourceType::TypedBuffer_SRV ||
            type == ResourceType::StructuredBuffer_SRV ||
            type == ResourceType::RawBuffer_SRV ||
            type == ResourceType::RayTracingAccelStruct;
    }

    static bool isUavType(ResourceType type)
    {
        return type == ResourceType::Texture_UAV ||
            type == ResourceType::TypedBuffer_UAV ||
            type == ResourceType::StructuredBuffer_UAV ||
            type == ResourceType::RawBuffer_UAV;
    }

    static bool isBufferType(ResourceType type)
    {
        return type == ResourceType::ConstantBuffer ||
            type == ResourceType::VolatileConstantBuffer ||
            type == ResourceType::TypedBuffer_SRV ||
            type == ResourceType::TypedBuffer_UAV ||
            type == ResourceType::StructuredBuffer_SRV ||
            type == ResourceType::StructuredBuffer_UAV ||
            type == ResourceType::RawBuffer_SRV ||
            type == ResourceType::RawBuffer_UAV;
    }

    static bool matchesMscArgumentType(ResourceType resourceType, MscArgumentType argumentType)
    {
        switch (argumentType)
        {
        case MscArgumentType::SRV:
            return isSrvType(resourceType);
        case MscArgumentType::UAV:
            return isUavType(resourceType);
        case MscArgumentType::CBV:
            return resourceType == ResourceType::ConstantBuffer || resourceType == ResourceType::VolatileConstantBuffer;
        case MscArgumentType::Sampler:
            return resourceType == ResourceType::Sampler;
        }

        return false;
    }

    static IRRuntimePrimitiveType convertRuntimePrimitiveType(PrimitiveType primitiveType)
    {
        switch (primitiveType)
        {
        case PrimitiveType::PointList: return IRRuntimePrimitiveTypePoint;
        case PrimitiveType::LineList: return IRRuntimePrimitiveTypeLine;
        case PrimitiveType::LineStrip: return IRRuntimePrimitiveTypeLineStrip;
        case PrimitiveType::TriangleStrip: return IRRuntimePrimitiveTypeTriangleStrip;
        case PrimitiveType::TriangleListWithAdjacency: return IRRuntimePrimitiveTypeTriangleWithAdj;
        case PrimitiveType::TriangleStripWithAdjacency: return IRRuntimePrimitiveTypeTriangleWithAdj;
        case PrimitiveType::TriangleList:
        case PrimitiveType::TriangleFan:
        default:
            return IRRuntimePrimitiveTypeTriangle;
        }
    }

    // Metal binding model:
    // - regular CB/SRV/UAV/sampler bindings are encoded into Metal Shader
    //   Converter descriptor tables using the reflected per-stage plan.
    // - volatile constant buffers are also encoded as descriptor-table CBVs,
    //   but their IRBufferView points at the current command-list upload
    //   allocation produced by writeBuffer().
    // - vertex, index, indirect, helper, and argument-table buffers remain
    //   direct Metal bindings because they are not ordinary shader resources.
    static const MetalBindingResource* findArgumentTableResource(
        const BindingSetVector& bindingSets,
        const MetalBindingPlanEntry& planEntry)
    {
        if (!planEntry.layoutMatched || planEntry.layoutIndex >= bindingSets.size())
            return nullptr;
        if (!bindingSets[planEntry.layoutIndex] || !bindingSets[planEntry.layoutIndex]->getDesc())
            return nullptr;

        auto* set = static_cast<BindingSet*>(bindingSets[planEntry.layoutIndex]);
        if (!set)
            return nullptr;

        for (const MetalBindingResource& entry : set->entries)
        {
            if (entry.type == ResourceType::None ||
                entry.type == ResourceType::SamplerFeedbackTexture_UAV ||
                !matchesMscArgumentType(entry.type, planEntry.argumentType))
                continue;

            if (entry.slot + entry.arrayElement == planEntry.slot)
                return &entry;
        }

        return nullptr;
    }

    bool CommandList::encodeArgumentTableEntry(IRDescriptorTableEntry* entry, const MetalBindingResource& resource)
    {
        if (resource.type != ResourceType::VolatileConstantBuffer)
        {
            const bool encoded = encodeMetalBindingResource(entry, resource);
            if (encoded && isBufferType(resource.type))
                referenceBuffer(static_cast<IBuffer*>(resource.resource.Get()));
            return encoded;
        }
        auto* buffer = static_cast<Buffer*>(resource.resource.Get());
        auto allocationIt = buffer ? m_VolatileBufferAllocations.find(buffer) : m_VolatileBufferAllocations.end();
        if (allocationIt == m_VolatileBufferAllocations.end() || !allocationIt->second.allocation.buffer)
            return false;
        const VolatileBufferAllocation& allocation = allocationIt->second;
        if (resource.bufferOffset >= allocation.writtenSize)
            return false;
        IRBufferView view{};
        view.buffer = allocation.allocation.buffer;
        view.bufferOffset = allocation.allocation.offset + resource.bufferOffset;
        view.bufferSize = std::min(resource.bufferSize, NSUInteger(allocation.writtenSize - resource.bufferOffset));
        IRDescriptorTableSetBufferView(entry, &view);
        m_ReferencedNativeBuffers.push_back(allocation.allocation.buffer);
        return true;
    }

    // Upload buffers retain their historical 4 MiB minimum by default. Callers
    // may request a smaller dedicated page size for a different data class.
    // m_CompletedSerial tracks which submitted command buffers have finished on the GPU
    UploadManager::UploadManager(const MTL3Context& context, size_t uploadChunkSize, size_t scratchMaxMem,
        bool isScratchBuffer, size_t minimumChunkSize, size_t initialChunkCount)
        : m_Context(context)
        , m_DefaultChunkSize(std::max(uploadChunkSize, minimumChunkSize))
        , m_CompletedSerial(std::make_shared<std::atomic<uint64_t>>(0))
    {
        (void)scratchMaxMem;
        (void)isScratchBuffer;

        m_Chunks.reserve(initialChunkCount);
        for (size_t index = 0; index < initialChunkCount; ++index)
        {
            id<MTLBuffer> buffer = [m_Context.device newBufferWithLength:NSUInteger(m_DefaultChunkSize)
                options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
            if (!buffer)
            {
                m_Context.error("[nvrhi] Failed to allocate initial Metal upload chunk.");
                break;
            }
            m_Context.residency->add(buffer);

            Chunk chunk;
            chunk.buffer = buffer;
            chunk.cpuAddress = static_cast<uint8_t*>([buffer contents]);
            chunk.size = m_DefaultChunkSize;
            m_Chunks.push_back(chunk);
        }
    }

    UploadManager::~UploadManager()
    {
        for (const Chunk& chunk : m_Chunks)
            m_Context.residency->remove(chunk.buffer);
    }
    // called, from cmdList.open, and every upload allocation made during this command (when cmdList open) list gets tagged with m_ActiveSerial
    void UploadManager::beginCommandBuffer()
    {
        m_ActiveSerial = ++m_SubmittedSerial;
        m_CurrentChunk = size_t(-1);
    }
    void UploadManager::submitCommandBuffer(id<MTLCommandBuffer> commandBuffer)
    {
        if (!commandBuffer || m_ActiveSerial == 0)
            return;

        const uint64_t submittedSerial = m_ActiveSerial;
        std::shared_ptr<std::atomic<uint64_t>> completedSerial = m_CompletedSerial;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            uint64_t previousValue = completedSerial->load(std::memory_order_relaxed);
            while (previousValue < submittedSerial &&
                !completedSerial->compare_exchange_weak(previousValue, submittedSerial,
                    std::memory_order_release, std::memory_order_relaxed))
            {
            }
        }];
        m_ActiveSerial = 0;
        m_CurrentChunk = size_t(-1);
    }

    void UploadManager::discardCommandBuffer()
    {
        if (m_ActiveSerial == 0)
            return;
        for (Chunk& chunk : m_Chunks)
        {
            if (chunk.lastUsedSerial == m_ActiveSerial)
            {
                chunk.lastUsedSerial = 0;
                chunk.offset = 0;
            }
        }
        m_ActiveSerial = 0;
        m_CurrentChunk = size_t(-1);
    }

    /*
     * this finds a chunk with enough free space, or creates a new one.
    */
    UploadManager::Chunk* UploadManager::findOrCreateChunk(size_t size, size_t alignment)
    {
        if (m_CurrentChunk != size_t(-1))
        {
            Chunk& chunk = m_Chunks[m_CurrentChunk];
            // try the current chunk it it has room for data, if yes return chunk*
            if (alignUp(chunk.offset, alignment) + size <= chunk.size)
                return &chunk;
        }

        const uint64_t completedSerial = m_CompletedSerial->load(std::memory_order_acquire);
        for (size_t index = 0; index < m_Chunks.size(); ++index)
        {
            Chunk& chunk = m_Chunks[index];
            // try if the old chunks are free now, checked with the GPU sync; if yes return chunk
            if (chunk.lastUsedSerial <= completedSerial && size <= chunk.size)
            {
                chunk.offset = 0;
                m_CurrentChunk = index;
                return &chunk;
            }
        }

        // if no free chunk found, create a new MTL buffer with the m_DefaultChunkSize, and return chunk from that memory
        const size_t chunkSize = std::max(m_DefaultChunkSize, alignUp(size, alignment));
        id<MTLBuffer> buffer = [m_Context.device newBufferWithLength:NSUInteger(chunkSize)
            options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (!buffer)
        {
            m_Context.error("[nvrhi] Failed to allocate Metal upload chunk.");
            return nullptr;
        }
        m_Context.residency->add(buffer);

        Chunk chunk;
        chunk.buffer = buffer;
        chunk.cpuAddress = static_cast<uint8_t*>([buffer contents]);
        chunk.size = chunkSize;
        m_Chunks.push_back(chunk);
        m_CurrentChunk = m_Chunks.size() - 1;
        return &m_Chunks.back();
    }

    // public allocater, used for asking chunks
    UploadAllocation UploadManager::suballocate(size_t size, size_t alignment)
    {
        UploadAllocation allocation;
        if (size == 0)
            return allocation;

        if (m_ActiveSerial == 0)
            beginCommandBuffer();

        Chunk* chunk = findOrCreateChunk(size, alignment);
        if (!chunk)
            return allocation;

        const size_t offset = alignUp(chunk->offset, alignment);
        allocation.buffer = chunk->buffer;
        allocation.offset = NSUInteger(offset);
        allocation.cpuAddress = chunk->cpuAddress + offset;

        chunk->offset = offset + size;
        chunk->lastUsedSerial = m_ActiveSerial;
        return allocation;
    }

    bool TransientIndirectResourcePool::IndirectCommandBufferKey::operator==(const IndirectCommandBufferKey& other) const
    {
        return commandTypes == other.commandTypes &&
            inheritPipelineState == other.inheritPipelineState &&
            inheritBuffers == other.inheritBuffers &&
            maxVertexBufferBindCount == other.maxVertexBufferBindCount &&
            maxFragmentBufferBindCount == other.maxFragmentBufferBindCount &&
            maxObjectBufferBindCount == other.maxObjectBufferBindCount &&
            maxMeshBufferBindCount == other.maxMeshBufferBindCount;
    }

    TransientIndirectResourcePool::TransientIndirectResourcePool(const MTL3Context& context)
        : m_Context(context)
        , m_State(std::make_shared<State>())
    {
        // The renderer permits three frames in flight. Pages are created lazily
        // in these slots only when an indirect path actually needs them.
        // no one in their sane mind goes beyond 3; less than 3 is fine here
        m_State->slots.resize(3);
    }

    TransientIndirectResourcePool::~TransientIndirectResourcePool()
    {
        std::lock_guard<std::mutex> lock(m_State->mutex);
        for (const FrameSlot& slot : m_State->slots)
        {
            for (const BufferPage& page : slot.sharedPages)
                m_Context.residency->remove(page.buffer);
            for (const BufferPage& page : slot.privatePages)
                m_Context.residency->remove(page.buffer);
            for (const PooledIndirectCommandBuffer& pooled : slot.indirectCommandBuffers)
                m_Context.residency->remove(pooled.commandBuffer);
        }
    }

    void TransientIndirectResourcePool::beginCommandBuffer()
    {
        std::lock_guard<std::mutex> lock(m_State->mutex);
        m_ActiveSlot = size_t(-1);
        for (size_t index = 0; index < m_State->slots.size(); ++index)
        {
            if (!m_State->slots[index].inFlight)
            {
                m_ActiveSlot = index;
                break;
            }
        }

        if (m_ActiveSlot == size_t(-1))
        {
            // Never overwrite resources that an earlier command buffer may
            // still use. An overflow slot is returned to the same free pool
            // once its completion handler fires.
            m_State->slots.emplace_back();
            m_ActiveSlot = m_State->slots.size() - 1;
            m_State->slots[m_ActiveSlot].isOverflow = true;
        }

        FrameSlot& slot = m_State->slots[m_ActiveSlot];
        slot.inFlight = true;
        slot.stats = {};
        slot.stats.usedOverflowSlot = slot.isOverflow;
        for (BufferPage& page : slot.sharedPages)
            page.writeOffset = 0;
        for (BufferPage& page : slot.privatePages)
            page.writeOffset = 0;
    }

    void TransientIndirectResourcePool::submitCommandBuffer(id<MTLCommandBuffer> commandBuffer)
    {
        if (!commandBuffer || m_ActiveSlot == size_t(-1))
            return;

        const std::shared_ptr<State> state = m_State;
        const size_t slotIndex = m_ActiveSlot;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (slotIndex < state->slots.size())
                state->slots[slotIndex].inFlight = false;
        }];
        m_ActiveSlot = size_t(-1);
    }

    void TransientIndirectResourcePool::discardCommandBuffer()
    {
        std::lock_guard<std::mutex> lock(m_State->mutex);
        if (m_ActiveSlot != size_t(-1))
        {
            m_State->slots[m_ActiveSlot].inFlight = false;
            m_ActiveSlot = size_t(-1);
        }
    }

    TransientIndirectResourcePool::Stats TransientIndirectResourcePool::getActiveStats() const
    {
        if (m_ActiveSlot == size_t(-1))
            return {};
        Stats stats = m_State->slots[m_ActiveSlot].stats;
        stats.sharedPageCount = m_State->slots[m_ActiveSlot].sharedPages.size();
        stats.privatePageCount = m_State->slots[m_ActiveSlot].privatePages.size();
        return stats;
    }

    TransientBufferAllocation TransientIndirectResourcePool::allocate(
        NSUInteger size, NSUInteger alignment, MTLResourceOptions options,
        std::vector<BufferPage> FrameSlot::*pages)
    {
        TransientBufferAllocation allocation;
        if (size == 0 || m_ActiveSlot == size_t(-1))
            return allocation;

        FrameSlot& slot = m_State->slots[m_ActiveSlot];
        std::vector<BufferPage>& pageList = slot.*pages;
        const NSUInteger alignedSize = NSUInteger(alignUp(size, alignment));

        for (BufferPage& page : pageList)
        {
            const NSUInteger offset = NSUInteger(alignUp(page.writeOffset, alignment));
            if (offset <= page.capacity && alignedSize <= page.capacity - offset)
            {
                // This is only a slice reservation. No native Metal resource is
                // created for normal indirect calls while an existing page fits.
                allocation.buffer = page.buffer;
                allocation.offset = offset;
                allocation.size = size;
                allocation.cpuAddress = page.cpuAddress ? page.cpuAddress + offset : nullptr;
                page.writeOffset = offset + alignedSize;
                slot.stats.bytesReserved += size;
                return allocation;
            }
        }

        // A page is created only after all existing pages in this completed
        // frame slot are full. Oversized requests receive one reusable page.
        constexpr NSUInteger kPageSize = 1024 * 1024;
        const NSUInteger pageSize = std::max(kPageSize, alignedSize);
        id<MTLBuffer> buffer = [m_Context.device newBufferWithLength:pageSize
            options:options | MTLResourceHazardTrackingModeUntracked];
        if (!buffer)
        {
            m_Context.error("[metal3] failed to allocate transient indirect resource page");
            return allocation;
        }
        m_Context.residency->add(buffer);

        BufferPage page;
        page.buffer = buffer;
        page.capacity = pageSize;
        page.cpuAddress = options == MTLResourceStorageModeShared
            ? static_cast<uint8_t*>([buffer contents]) : nullptr;
        page.writeOffset = alignedSize;
        pageList.push_back(page);

        if (options == MTLResourceStorageModeShared)
            ++slot.stats.sharedPagesCreated;
        else
            ++slot.stats.privatePagesCreated;
        slot.stats.bytesReserved += size;

        allocation.buffer = buffer;
        allocation.size = size;
        allocation.cpuAddress = page.cpuAddress;
        return allocation;
    }

    // Shared pages use MTLResourceStorageModeShared and expose CPU memory. They hold data written by the CPU:
    // -> "paramsBuffer" contents
    // -> "icbArgumentBuffer" contents populated through MTLArgumentEncoder

    TransientBufferAllocation TransientIndirectResourcePool::allocateShared(NSUInteger size, NSUInteger alignment)
    {
        return allocate(size, alignment, MTLResourceStorageModeShared, &FrameSlot::sharedPages);
    }

    /*
    // Private pages use MTLResourceStorageModePrivate; CPU address is null. They hold GPU-only data:
    // executionRange
    // drawInfoBuffer
    // drawParamsBuffer
    // meshIndirectArgsBuffer
    */
    TransientBufferAllocation TransientIndirectResourcePool::allocatePrivate(NSUInteger size, NSUInteger alignment)
    {
        return allocate(size, alignment, MTLResourceStorageModePrivate, &FrameSlot::privatePages);
    }

    id<MTLIndirectCommandBuffer> TransientIndirectResourcePool::acquireIndirectCommandBuffer(
        MTLIndirectCommandBufferDescriptor* descriptor, NSUInteger requiredCapacity, bool* wasCreated,
        NSUInteger* capacityOut)
    {
        if (wasCreated)
            *wasCreated = false;
        if (capacityOut)
            *capacityOut = 0;
        if (!descriptor || requiredCapacity == 0 || m_ActiveSlot == size_t(-1))
            return nil;

        IndirectCommandBufferKey key;
        key.commandTypes = descriptor.commandTypes;
        key.inheritPipelineState = descriptor.inheritPipelineState;
        key.inheritBuffers = descriptor.inheritBuffers;
        key.maxVertexBufferBindCount = descriptor.maxVertexBufferBindCount;
        key.maxFragmentBufferBindCount = descriptor.maxFragmentBufferBindCount;
        key.maxObjectBufferBindCount = descriptor.maxObjectBufferBindCount;
        key.maxMeshBufferBindCount = descriptor.maxMeshBufferBindCount;

        FrameSlot& slot = m_State->slots[m_ActiveSlot];
        for (const PooledIndirectCommandBuffer& pooled : slot.indirectCommandBuffers)
        {
            if (pooled.key == key && pooled.capacity >= requiredCapacity)
            {
                // Reusing this ICB is safe because the containing slot belongs
                // exclusively to the currently-recording command buffer.
                ++slot.stats.indirectCommandBuffersReused;
                if (capacityOut)
                    *capacityOut = pooled.capacity;
                return pooled.commandBuffer;
            }
        }

        NSUInteger capacity = 1;
        while (capacity < requiredCapacity && capacity <= std::numeric_limits<NSUInteger>::max() / 2)
            capacity <<= 1;
        if (capacity < requiredCapacity)
            capacity = requiredCapacity;

        // ICBs cannot be suballocated by offset. Grow capacity geometrically
        // and retain the complete object in this frame slot for later reuse.
        id<MTLIndirectCommandBuffer> commandBuffer =
            [m_Context.device newIndirectCommandBufferWithDescriptor:descriptor
                                                     maxCommandCount:capacity
                                                            options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
        if (!commandBuffer)
        {
            m_Context.error("[metal3] failed to allocate transient indirect command buffer");
            return nil;
        }
        m_Context.residency->add(commandBuffer);

        PooledIndirectCommandBuffer pooled;
        pooled.key = key;
        pooled.capacity = capacity;
        pooled.commandBuffer = commandBuffer;
        slot.indirectCommandBuffers.push_back(pooled);
        ++slot.stats.indirectCommandBuffersCreated;
        if (wasCreated)
            *wasCreated = true;
        if (capacityOut)
            *capacityOut = capacity;
        return commandBuffer;
    }

    CommandList::CommandList(class Device* device, const MTL3Context& context, const CommandListParameters& params)
        : m_Context(context)
            , m_Device(device)
            , m_LifetimeTracker(params.lifetimeTracker
                ? static_cast<CommandListLifetimeTracker*>(params.lifetimeTracker->getNativeObject(ObjectTypes::Nvrhi_Metal3_LifetimeTracker).pointer)
                : device->m_DefaultLifetimeTrackers[uint32_t(params.queueType)].Get())
            , m_UploadManager(context, params.uploadChunkSize, 0, false)
            , m_ArgumentTableManager(context, c_ArgumentTablePageSize, 0, false,
                c_ArgumentTablePageSize, c_ArgumentTableInitialPageCount)
            , m_TransientIndirectResources(context)
            , m_StateTracker(context.messageCallback)
            , m_Desc(params) {}
        
    CommandList::~CommandList()
    {
        std::lock_guard<std::mutex> lock(m_Device->m_Mutex);
        endEncoding();
        if (m_Device->m_OpenImmediateCommandList == this)
            m_Device->m_OpenImmediateCommandList = nullptr;
        trackedCmdBuffer = nil;
        m_UploadManager.discardCommandBuffer();
        m_ArgumentTableManager.discardCommandBuffer();
        m_TransientIndirectResources.discardCommandBuffer();
    }

    Object CommandList::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::Nvrhi_Metal3_CommandList:
            return Object(this);
        default:
            return nullptr;
        }
    }

    id<MTLCommandBuffer> CommandList::getNativeCommandBuffer()
    {
        if (m_BlitEncoder)
            endEncoding();
        return trackedCmdBuffer;
    }

    void CommandList::setTracyGpuScope(const char* name, const char* file, const char* function, uint32_t line, void* context)
    {
        m_TracyGpuScope.name = name;
        m_TracyGpuScope.file = file;
        m_TracyGpuScope.function = function;
        m_TracyGpuScope.line = line;
        m_TracyGpuScope.context = context;
        m_TracyGpuScope.active = name && file && function && context;
    }

    void CommandList::clearTracyGpuScope()
    {
        m_TracyGpuScope = TracyGpuScopeDesc{};
    }

#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
    tracy::SourceLocationData* CommandList::getOrCreateTracySourceLocation()
    {
        if (!m_TracyGpuScope.active)
            return nullptr;

        for (const auto& sourceLocation : m_TracySourceLocations)
        {
            if (sourceLocation->line == m_TracyGpuScope.line &&
                std::strcmp(sourceLocation->name ? sourceLocation->name : "", m_TracyGpuScope.name ? m_TracyGpuScope.name : "") == 0 &&
                std::strcmp(sourceLocation->file ? sourceLocation->file : "", m_TracyGpuScope.file ? m_TracyGpuScope.file : "") == 0 &&
                std::strcmp(sourceLocation->function ? sourceLocation->function : "", m_TracyGpuScope.function ? m_TracyGpuScope.function : "") == 0)
                return sourceLocation.get();
        }

        auto sourceLocation = std::make_unique<tracy::SourceLocationData>();
        sourceLocation->name = m_TracyGpuScope.name;
        sourceLocation->function = m_TracyGpuScope.function;
        sourceLocation->file = m_TracyGpuScope.file;
        sourceLocation->line = m_TracyGpuScope.line;
        sourceLocation->color = 0;

        tracy::SourceLocationData* ptr = sourceLocation.get();
        m_TracySourceLocations.push_back(std::move(sourceLocation));
        return ptr;
    }

    void CommandList::beginTracyRenderEncoderZone(MTLRenderPassDescriptor* desc)
    {
        if (!desc || !m_TracyGpuScope.active || !m_TracyGpuScope.context)
            return;

        tracy::SourceLocationData* sourceLocation = getOrCreateTracySourceLocation();
        if (!sourceLocation)
            return;

        m_TracyEncoderScope.emplace(static_cast<TracyMetalCtx*>(m_TracyGpuScope.context), desc, sourceLocation, true);
    }

    void CommandList::beginTracyComputeEncoderZone(MTLComputePassDescriptor* desc)
    {
        if (!desc || !m_TracyGpuScope.active || !m_TracyGpuScope.context)
            return;

        tracy::SourceLocationData* sourceLocation = getOrCreateTracySourceLocation();
        if (!sourceLocation)
            return;

        m_TracyEncoderScope.emplace(static_cast<TracyMetalCtx*>(m_TracyGpuScope.context), desc, sourceLocation, true);
    }
#endif

    static constexpr MTLRenderStages c_AllRenderStages =
        MTLRenderStageVertex | MTLRenderStageFragment | MTLRenderStageObject | MTLRenderStageMesh;

    void CommandList::beginEncoding(id<MTLRenderCommandEncoder> encoder, const char* operation)
    {
        annotateEncoder(encoder, operation);
        if (m_EncoderFenceArmed)
            [encoder waitForFence:m_EncoderFence beforeStages:c_AllRenderStages];
    }

    void CommandList::beginEncoding(id<MTLComputeCommandEncoder> encoder, const char* operation)
    {
        annotateEncoder(encoder, operation);
        if (m_EncoderFenceArmed)
            [encoder waitForFence:m_EncoderFence];
    }

    void CommandList::beginEncoding(id<MTLBlitCommandEncoder> encoder, const char* operation)
    {
        annotateEncoder(encoder, operation);
        if (m_EncoderFenceArmed)
            [encoder waitForFence:m_EncoderFence];
    }

    void CommandList::endEncoding(id<MTLRenderCommandEncoder> encoder)
    {
        if (!encoder)
            return;
        if (m_EncoderFence)
        {
            [encoder updateFence:m_EncoderFence afterStages:c_AllRenderStages];
            m_EncoderFenceArmed = true;
        }
        [encoder endEncoding];
    }

    void CommandList::endEncoding(id<MTLComputeCommandEncoder> encoder)
    {
        if (!encoder)
            return;
        if (m_EncoderFence)
        {
            [encoder updateFence:m_EncoderFence];
            m_EncoderFenceArmed = true;
        }
        [encoder endEncoding];
    }

    void CommandList::endEncoding(id<MTLBlitCommandEncoder> encoder)
    {
        if (!encoder)
            return;
        if (m_EncoderFence)
        {
            [encoder updateFence:m_EncoderFence];
            m_EncoderFenceArmed = true;
        }
        [encoder endEncoding];
    }

    void CommandList::beginEncoding(id<MTLAccelerationStructureCommandEncoder> encoder, const char* operation)
    {
        annotateEncoder(encoder, operation);
        if (m_EncoderFenceArmed)
            [encoder waitForFence:m_EncoderFence];
    }

    void CommandList::endEncoding(id<MTLAccelerationStructureCommandEncoder> encoder)
    {
        if (!encoder)
            return;
        if (m_EncoderFence)
        {
            [encoder updateFence:m_EncoderFence];
            m_EncoderFenceArmed = true;
        }
        [encoder endEncoding];
    }

    void CommandList::endEncoding()
    {
        if (m_RenderEncoder)
        {
            endEncoding(m_RenderEncoder);
            m_RenderEncoder = nil;
        }
        if (m_ComputeEncoder)
        {
            endEncoding(m_ComputeEncoder);
            m_ComputeEncoder = nil;
        }
        if (m_BlitEncoder)
        {
            endEncoding(m_BlitEncoder);
            m_BlitEncoder = nil;
        }
        m_BlitEncoderWrites.clear();
        m_BindingStatesDirty = true;
        m_ArgumentTablesDirty = false;
        m_RenderEncoderWrites.clear();
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        m_TracyEncoderScope.reset();
#endif
    }

    void CommandList::unsupported(const char* operation)
    {
        m_RecordingFailed = true;
        m_Context.unsupported(operation);
    }

    void CommandList::open()
    {
        std::lock_guard<std::mutex> lock(m_Device->m_Mutex);
        if (m_RecordingState == RecordingState::Open)
        {
            m_Context.error("[nvrhi] Cannot open an already open Metal command list.");
            return;
        }
        if (m_Desc.enableImmediateExecution &&
            ((m_RecordingState == RecordingState::Closed && !m_RecordingFailed) || m_Device->m_OpenImmediateCommandList))
        {
            m_Context.error("[nvrhi] Immediate Metal command lists must be executed before reopening and cannot be recorded concurrently.");
            return;
        }

        const bool wasSubmitted = m_RecordingState == RecordingState::Submitted;
        trackedCmdBuffer = nil;
        m_EncoderFence = m_Device->m_Queues[uint32_t(m_Desc.queueType)].fence;
        m_EncoderFenceArmed = m_Device->m_Queues[uint32_t(m_Desc.queueType)].fenceArmed;
        m_PendingClears.clear();
        m_RenderEncoderWrites.clear();
        m_UploadManager.discardCommandBuffer();
        m_ArgumentTableManager.discardCommandBuffer();
        m_TransientIndirectResources.discardCommandBuffer();
        m_RecordingState = RecordingState::Initial;
        if (encoderExecutionStatusEnabled())
        {
            MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];
            descriptor.retainedReferences = YES;
            descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
            trackedCmdBuffer = [m_Context.queue(m_Desc.queueType) commandBufferWithDescriptor:descriptor];
        }
        else
            trackedCmdBuffer = [m_Context.queue(m_Desc.queueType) commandBuffer];
        if (!trackedCmdBuffer)
        {
            m_Context.error("[nvrhi] Failed to allocate a Metal command buffer.");
            return;
        }
        if (!m_Device->m_CommandBufferConfigurationReported)
        {
            m_Device->m_CommandBufferConfigurationReported = true;
            m_Context.info("[metal3] command-buffer diagnostics: trace=" + std::to_string(traceMetalRuntime()) +
                " encoderStatusRequested=" + std::to_string(encoderExecutionStatusEnabled()) +
                " encoderStatusEffective=" + std::to_string(bool(trackedCmdBuffer.errorOptions &
                    MTLCommandBufferErrorOptionEncoderExecutionStatus)) +
                " errorOptions=" + std::to_string(uint64_t(trackedCmdBuffer.errorOptions)) +
                " retainedReferences=" + std::to_string(bool(trackedCmdBuffer.retainedReferences)));
        }
        m_RecordingState = RecordingState::Open;
        m_RecordingFailed = false;
        if (m_Desc.enableImmediateExecution)
            m_Device->m_OpenImmediateCommandList = this;

        m_ReferencedBindingSets.clear();
        m_ReferencedNativeBuffers.clear();
        m_ReferencedNativeResources.clear();
        m_ReferencedDescriptorSnapshots.clear();
        m_ReferencedStateResources.clear();
        m_ReferencedMappableBuffers.clear();
        if (wasSubmitted)
        {
            TrackedCommandBuffer storage = m_LifetimeTracker->acquireResourceStorage();
            m_ReferencedBindingSets.swap(storage.bindingSets);
            m_ReferencedNativeBuffers.swap(storage.buffers);
            m_ReferencedNativeResources.swap(storage.resources);
            m_ReferencedDescriptorSnapshots.swap(storage.descriptorSnapshots);
            m_ReferencedStateResources.swap(storage.stateResources);
        }
        else
            m_StateTracker = CommandListResourceStateTracker(m_Context.messageCallback);
        auto resetAllocations = [](auto& cache) {
            const size_t live = std::count_if(cache.begin(), cache.end(), [](const auto& entry) {
                return entry.second.allocation.buffer != nil;
            });
            const bool prune = cache.size() > 2 * live + 1024;
            for (auto it = cache.begin(); it != cache.end();)
            {
                if (prune && !it->second.allocation.buffer)
                    it = cache.erase(it);
                else
                {
                    it->second = {};
                    ++it;
                }
            }
        };
        m_PushConstantSize = 0;
        m_BindingStatesDirty = true;
        m_ArgumentTablesDirty = false;
        resetAllocations(m_ArgumentTableCache);
        m_DebugGroups.clear();
        m_UploadManager.beginCommandBuffer();
        m_ArgumentTableManager.beginCommandBuffer();
        m_TransientIndirectResources.beginCommandBuffer();
        m_ArgumentTableAllocationCount = 0;
        m_ArgumentTablePageCountAtOpen = m_ArgumentTableManager.getChunkCount();
        resetAllocations(m_VolatileBufferAllocations);
        m_VolatileBufferWriteVersion = 0;
        m_CurrentGraphicsStateValid = false;
        m_CurrentComputeStateValid = false;
        m_CurrentMeshletStateValid = false;
        m_CurrentFramebuffer = nullptr;
        m_GeometryEmulationDrawStateValid = false;
        m_GeometryEmulationVertexBuffers = nil;
        m_GeometryEmulationVertexBuffersOffset = 0;
        m_RenderEncoder = nil;
        m_ComputeEncoder = nil;
        m_BlitEncoder = nil;
    }
    void CommandList::close()
    {
        std::lock_guard<std::mutex> lock(m_Device->m_Mutex);
        if (m_RecordingState != RecordingState::Open)
        {
            m_Context.error("[nvrhi] Cannot close a Metal command list that is not open.");
            return;
        }
        endEncoding();
        flushPendingClears();
        m_StateTracker.keepBufferInitialStates();
        m_StateTracker.keepTextureInitialStates();
        commitBarriers();
        m_RecordingState = RecordingState::Closed;
        if (m_Device->m_OpenImmediateCommandList == this)
            m_Device->m_OpenImmediateCommandList = nullptr;
        if (indirectResourceStatsEnabled())
        {
            const TransientIndirectResourcePool::Stats stats = m_TransientIndirectResources.getActiveStats();
            m_Context.info("[metal3] indirect resources: shared_pages=" +
                std::to_string(stats.sharedPageCount) +
                " private_pages=" + std::to_string(stats.privatePageCount) +
                " shared_pages_created=" + std::to_string(stats.sharedPagesCreated) +
                " private_pages_created=" + std::to_string(stats.privatePagesCreated) +
                " bytes_reserved=" + std::to_string(stats.bytesReserved) +
                " icbs_reused=" + std::to_string(stats.indirectCommandBuffersReused) +
                " icbs_created=" + std::to_string(stats.indirectCommandBuffersCreated) +
                " overflow_slot=" + std::to_string(stats.usedOverflowSlot ? 1 : 0));
        }
        if (argumentTableStatsEnabled())
        {
            const size_t pageCount = m_ArgumentTableManager.getChunkCount();
            m_Context.info("[metal3] argument-table frame: tables=" +
                std::to_string(m_ArgumentTableAllocationCount) +
                " pages=" + std::to_string(pageCount) +
                " new_pages=" + std::to_string(pageCount - m_ArgumentTablePageCountAtOpen));
        }
    }

    void CommandList::submit()
    {
        m_UploadManager.submitCommandBuffer(trackedCmdBuffer);
        m_ArgumentTableManager.submitCommandBuffer(trackedCmdBuffer);
        m_TransientIndirectResources.submitCommandBuffer(trackedCmdBuffer);
        m_StateTracker.commandListSubmitted();
        if (m_EncoderFenceArmed)
            m_Device->m_Queues[uint32_t(m_Desc.queueType)].fenceArmed = true;
        m_LifetimeTracker->m_CommandBuffers.push_back({
            trackedCmdBuffer,
            std::move(m_ReferencedBindingSets),
            std::move(m_ReferencedNativeBuffers),
            std::move(m_ReferencedNativeResources),
            std::move(m_ReferencedDescriptorSnapshots),
            std::move(m_ReferencedStateResources)
        });
        [trackedCmdBuffer commit];
        m_RecordingState = RecordingState::Submitted;
    }

    void CommandList::clearState()
    {
        m_CurrentGraphicsState = GraphicsState();
        m_CurrentComputeState = ComputeState();
        m_CurrentMeshletState = MeshletState();
        m_CurrentGraphicsStateValid = false;
        m_CurrentComputeStateValid = false;
        m_CurrentMeshletStateValid = false;
        m_CurrentFramebuffer = nullptr;
        m_GeometryEmulationDrawStateValid = false;
        m_GeometryEmulationVertexBuffers = nil;
        m_GeometryEmulationVertexBuffersOffset = 0;
        endEncoding();
    }

    void CommandList::clearTextureFloat(ITexture* t, TextureSubresourceSet subresources, const Color& clearColor)
    {
        auto* texture = static_cast<Texture*>(t);
        if (!texture || !texture->texture)
            return;
        subresources = subresources.resolve(texture->desc, false);
        endEncoding();
        for (uint32_t mip = subresources.baseMipLevel; mip < subresources.baseMipLevel + subresources.numMipLevels; ++mip)
        {
            for (uint32_t slice = subresources.baseArraySlice; slice < subresources.baseArraySlice + subresources.numArraySlices; ++slice)
            {
                PendingClear* pending = nullptr;
                for (PendingClear& existing : m_PendingClears)
                {
                    if (existing.texture == texture && existing.mipLevel == mip && existing.arraySlice == slice && !existing.isDepthStencil)
                    {
                        pending = &existing;
                        break;
                    }
                }
                if (!pending)
                {
                    m_PendingClears.emplace_back();
                    pending = &m_PendingClears.back();
                    pending->texture = texture;
                    pending->mipLevel = mip;
                    pending->arraySlice = slice;
                }
                pending->color = clearColor;
            }
        }
    }

    void CommandList::clearDepthStencilTexture(ITexture* t, TextureSubresourceSet subresources, bool clearDepth, float depth, bool clearStencil, uint8_t stencil)
    {
        auto* texture = static_cast<Texture*>(t);
        if (!texture || !texture->texture || (!clearDepth && !clearStencil))
            return;
        subresources = subresources.resolve(texture->desc, false);
        const FormatInfo& format = getFormatInfo(texture->desc.format);
        clearDepth = clearDepth && format.hasDepth;
        clearStencil = clearStencil && format.hasStencil;
        if (!clearDepth && !clearStencil)
            return;
        endEncoding();
        for (uint32_t mip = subresources.baseMipLevel; mip < subresources.baseMipLevel + subresources.numMipLevels; ++mip)
        {
            for (uint32_t slice = subresources.baseArraySlice; slice < subresources.baseArraySlice + subresources.numArraySlices; ++slice)
            {
                PendingClear* pending = nullptr;
                for (PendingClear& existing : m_PendingClears)
                {
                    if (existing.texture == texture && existing.mipLevel == mip && existing.arraySlice == slice && existing.isDepthStencil)
                    {
                        pending = &existing;
                        break;
                    }
                }
                if (!pending)
                {
                    m_PendingClears.emplace_back();
                    pending = &m_PendingClears.back();
                    pending->texture = texture;
                    pending->mipLevel = mip;
                    pending->arraySlice = slice;
                    pending->isDepthStencil = true;
                }
                if (clearDepth)
                {
                    pending->clearDepth = true;
                    pending->depth = depth;
                }
                if (clearStencil)
                {
                    pending->clearStencil = true;
                    pending->stencil = stencil;
                }
            }
        }
    }

    void CommandList::encodeStandaloneClear(const PendingClear& clear)
    {
        auto* texture = static_cast<Texture*>(clear.texture.Get());
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        if (clear.isDepthStencil)
        {
            const FormatInfo& format = getFormatInfo(texture->desc.format);
            if (format.hasDepth)
            {
                rp.depthAttachment.texture = texture->texture;
                rp.depthAttachment.level = clear.mipLevel;
                rp.depthAttachment.slice = clear.arraySlice;
                rp.depthAttachment.loadAction = clear.clearDepth ? MTLLoadActionClear : MTLLoadActionLoad;
                rp.depthAttachment.storeAction = MTLStoreActionStore;
                rp.depthAttachment.clearDepth = clear.depth;
            }
            if (format.hasStencil)
            {
                rp.stencilAttachment.texture = texture->texture;
                rp.stencilAttachment.level = clear.mipLevel;
                rp.stencilAttachment.slice = clear.arraySlice;
                rp.stencilAttachment.loadAction = clear.clearStencil ? MTLLoadActionClear : MTLLoadActionLoad;
                rp.stencilAttachment.storeAction = MTLStoreActionStore;
                rp.stencilAttachment.clearStencil = clear.stencil;
            }
        }
        else
        {
            rp.colorAttachments[0].texture = texture->texture;
            rp.colorAttachments[0].level = clear.mipLevel;
            rp.colorAttachments[0].slice = clear.arraySlice;
            rp.colorAttachments[0].loadAction = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor = MTLClearColorMake(clear.color.r, clear.color.g, clear.color.b, clear.color.a);
        }
        id<MTLRenderCommandEncoder> encoder = [trackedCmdBuffer renderCommandEncoderWithDescriptor:rp];
        beginEncoding(encoder, "clear");
        endEncoding(encoder);
    }

    void CommandList::flushPendingClears()
    {
        for (const PendingClear& clear : m_PendingClears)
            encodeStandaloneClear(clear);
        m_PendingClears.clear();
    }

    void CommandList::clearTextureUInt(ITexture* t, TextureSubresourceSet subresources, uint32_t clearColor)
    {
        Color color(float(clearColor), 0.f, 0.f, 0.f);
        clearTextureFloat(t, subresources, color);
    }

    void CommandList::copyTexture(ITexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice)
    {
        auto* d = static_cast<Texture*>(dest);
        auto* s = static_cast<Texture*>(src);
        if (!d || !s || !d->texture || !s->texture)
            return;

        const TextureSlice ds = destSlice.resolve(d->desc);
        const TextureSlice ss = srcSlice.resolve(s->desc);
        const uint32_t srcMipWidth = std::max(1u, s->desc.width >> ss.mipLevel);
        const uint32_t srcMipHeight = std::max(1u, s->desc.height >> ss.mipLevel);
        const uint32_t srcMipDepth = s->desc.dimension == TextureDimension::Texture3D
            ? std::max(1u, s->desc.depth >> ss.mipLevel)
            : 1u;
        const uint32_t dstMipWidth = std::max(1u, d->desc.width >> ds.mipLevel);
        const uint32_t dstMipHeight = std::max(1u, d->desc.height >> ds.mipLevel);
        const uint32_t dstMipDepth = d->desc.dimension == TextureDimension::Texture3D
            ? std::max(1u, d->desc.depth >> ds.mipLevel)
            : 1u;

        if (ss.x >= srcMipWidth || ss.y >= srcMipHeight || ss.z >= srcMipDepth ||
            ds.x >= dstMipWidth || ds.y >= dstMipHeight || ds.z >= dstMipDepth)
            return;

        const MTLSize copySize = MTLSizeMake(
            std::min({ ss.width, ds.width, srcMipWidth - ss.x, dstMipWidth - ds.x }),
            std::min({ ss.height, ds.height, srcMipHeight - ss.y, dstMipHeight - ds.y }),
            std::min({ ss.depth, ds.depth, srcMipDepth - ss.z, dstMipDepth - ds.z }));

        if (copySize.width == 0 || copySize.height == 0 || copySize.depth == 0)
            return;

        endEncoding();
        id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
        beginEncoding(blit, __func__);
        [blit copyFromTexture:s->texture
                  sourceSlice:ss.arraySlice
                  sourceLevel:ss.mipLevel
                 sourceOrigin:MTLOriginMake(ss.x, ss.y, ss.z)
                   sourceSize:copySize
                    toTexture:d->texture
             destinationSlice:ds.arraySlice
             destinationLevel:ds.mipLevel
            destinationOrigin:MTLOriginMake(ds.x, ds.y, ds.z)];
        endEncoding(blit);
    }

    /*
     * writeTexture computes packed row layout, asks for upload memory, writes rows into it, then does:
        copyFromBuffer:upload.buffer
        sourceOffset:upload.offset
        sourceBytesPerRow:naturalRowPitch
        sourceBytesPerImage:naturalBytesPerImage
        toTexture:texture->texture
     * So textures also avoid per-upload Metal buffer allocation.
    */
    void CommandList::writeTexture(ITexture* dest, uint32_t arraySlice, uint32_t mipLevel, const void* data, size_t rowPitch, size_t depthPitch)
    {
        auto* texture = static_cast<Texture*>(dest);
        if (!texture || !texture->texture || !data)
            return;

        const TextureDesc& desc = texture->desc;
        if (mipLevel >= desc.mipLevels || arraySlice >= desc.arraySize)
            return;

        const FormatInfo& formatInfo = getFormatInfo(desc.format);
        const uint32_t mipWidth = std::max(1u, desc.width >> mipLevel);
        const uint32_t mipHeight = std::max(1u, desc.height >> mipLevel);
        const uint32_t mipDepth = desc.dimension == TextureDimension::Texture3D
            ? std::max(1u, desc.depth >> mipLevel)
            : 1u;

        // Metal's buffer-to-texture copy describes rows in bytes, but NVRHI rowPitch
        // describes the caller's source layout. For block-compressed formats, one
        // logical row is a row of compression blocks, not pixels, so compute the
        // tightly packed GPU row size from block columns and bytes per block.
        const uint32_t blockCols = (mipWidth + formatInfo.blockSize - 1u) / formatInfo.blockSize;
        const uint32_t blockRows = (mipHeight + formatInfo.blockSize - 1u) / formatInfo.blockSize;
        const size_t naturalRowPitch = size_t(blockCols) * formatInfo.bytesPerBlock;

        if (rowPitch == 0)
            rowPitch = naturalRowPitch;
        if (rowPitch < naturalRowPitch)
            return;

        const size_t naturalBytesPerImage = naturalRowPitch * blockRows;
        if (depthPitch == 0)
            depthPitch = rowPitch * blockRows;
        if (depthPitch < rowPitch * blockRows)
            return;

        const size_t copySize = naturalBytesPerImage * mipDepth;
        UploadAllocation upload = m_UploadManager.suballocate(copySize, 256);
        if (!upload.buffer || !upload.cpuAddress)
            return;

        // pack only the meaningful row bytes into the upload buffer. The caller's
        // rows/depth slices may include padding, while the Metal copy below uses a
        // tightly packed sourceBytesPerRow/sourceBytesPerImage layout.
        auto* destBytes = static_cast<uint8_t*>(upload.cpuAddress);
        const auto* srcBytes = static_cast<const uint8_t*>(data);
        for (uint32_t z = 0; z < mipDepth; ++z)
        {
            uint8_t* destImage = destBytes + naturalBytesPerImage * z;
            const uint8_t* srcImage = srcBytes + depthPitch * z;
            for (uint32_t row = 0; row < blockRows; ++row)
            {
                memcpy(destImage + naturalRowPitch * row, srcImage + rowPitch * row, naturalRowPitch);
            }
        }

        endEncoding();
        id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
        beginEncoding(blit, __func__);
        [blit copyFromBuffer:upload.buffer
                 sourceOffset:upload.offset
            sourceBytesPerRow:naturalRowPitch
          sourceBytesPerImage:naturalBytesPerImage
                   sourceSize:MTLSizeMake(mipWidth, mipHeight, mipDepth)
                    toTexture:texture->texture
             destinationSlice:arraySlice
             destinationLevel:mipLevel
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        endEncoding(blit);
    }

    void CommandList::resolveTexture(ITexture* dest, const TextureSubresourceSet& dstSubresources, ITexture* src, const TextureSubresourceSet& srcSubresources)
    {
        auto* d = static_cast<Texture*>(dest);
        auto* s = static_cast<Texture*>(src);
        if (!d || !s || !d->texture || !s->texture)
            return;

        if (s->desc.sampleCount <= 1 || d->desc.sampleCount != 1 || s->desc.format != d->desc.format)
            return;

        const FormatInfo& formatInfo = getFormatInfo(d->desc.format);
        if (formatInfo.kind == FormatKind::DepthStencil)
            return;

        // NVRHI resolveTexture can target a range of mip levels and array slices.
        // Metal resolves one attachment subresource per render pass, so resolve
        // both ranges first and then issue one tiny render pass for each pair.
        const TextureSubresourceSet dstSR = dstSubresources.resolve(d->desc, false);
        const TextureSubresourceSet srcSR = srcSubresources.resolve(s->desc, false);
        if (dstSR.numArraySlices != srcSR.numArraySlices || dstSR.numMipLevels != srcSR.numMipLevels)
            return;

        endEncoding();

        for (ArraySlice arrayIndex = 0; arrayIndex < dstSR.numArraySlices; ++arrayIndex)
        {
            for (MipLevel mipLevel = 0; mipLevel < dstSR.numMipLevels; ++mipLevel)
            {
                // Metal does not expose MSAA resolve as a blit operation. The
                // source MSAA texture is attached for loading, and the resolved
                // single-sample texture is written by the resolve store action.
                MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
                MTLRenderPassColorAttachmentDescriptor* colorAttachment = rp.colorAttachments[0];
                colorAttachment.texture = s->texture;
                colorAttachment.level = mipLevel + srcSR.baseMipLevel;
                colorAttachment.slice = arrayIndex + srcSR.baseArraySlice;
                colorAttachment.loadAction = MTLLoadActionLoad;
                colorAttachment.storeAction = MTLStoreActionMultisampleResolve;
                colorAttachment.resolveTexture = d->texture;
                colorAttachment.resolveLevel = mipLevel + dstSR.baseMipLevel;
                colorAttachment.resolveSlice = arrayIndex + dstSR.baseArraySlice;

                id<MTLRenderCommandEncoder> encoder = [trackedCmdBuffer renderCommandEncoderWithDescriptor:rp];
                beginEncoding(encoder, __func__);
                endEncoding(encoder);
            }
        }
    }
    void CommandList::writeBuffer(IBuffer* b, const void* data, size_t dataSize, uint64_t destOffsetBytes)
    {
        auto* buffer = static_cast<Buffer*>(b);
        if (!buffer || !data || dataSize == 0)
            return;

        if (destOffsetBytes > buffer->desc.byteSize || dataSize > buffer->desc.byteSize - destOffsetBytes)
            return;

        // if isVolatile, write into the CPU-GPU shared mem directly and return
        if (buffer->desc.isVolatile)
        {
            UploadAllocation allocation = m_UploadManager.suballocate(buffer->desc.byteSize, 256);
            if (!allocation.buffer || !allocation.cpuAddress)
                return;

            if (destOffsetBytes != 0 || dataSize != buffer->desc.byteSize)
            {
                auto previous = m_VolatileBufferAllocations.find(buffer);
                if (previous != m_VolatileBufferAllocations.end() && previous->second.allocation.buffer)
                    std::memcpy(allocation.cpuAddress, previous->second.allocation.cpuAddress, buffer->desc.byteSize);
                else
                    std::memset(allocation.cpuAddress, 0, buffer->desc.byteSize);
            }
            std::memcpy(static_cast<uint8_t*>(allocation.cpuAddress) + destOffsetBytes, data, dataSize);
            VolatileBufferAllocation record;
            record.allocation = allocation;
            record.writtenSize = buffer->desc.byteSize;
            record.version = ++m_VolatileBufferWriteVersion;
            m_VolatileBufferAllocations[buffer] = record;
            m_ArgumentTablesDirty = true;
            return;
        }

        if (!buffer->buffer)
            return;

        // if not volatile, for normal buffer, CPU writes into upload shared memory, and then perform blit op
        UploadAllocation upload = m_UploadManager.suballocate(dataSize, 256);
        if (!upload.buffer || !upload.cpuAddress)
            return;

        memcpy(upload.cpuAddress, data, dataSize);
        referenceBuffer(buffer);

        const void* destination = (__bridge const void*)buffer->buffer;
        if (m_BlitEncoder && std::find(m_BlitEncoderWrites.begin(), m_BlitEncoderWrites.end(), destination) !=
            m_BlitEncoderWrites.end())
            endEncoding();
        if (!m_BlitEncoder)
        {
            m_BlitEncoder = beginBlitEncoder();
            beginEncoding(m_BlitEncoder, __func__);
        }
        [m_BlitEncoder copyFromBuffer:upload.buffer
                sourceOffset:upload.offset
                    toBuffer:buffer->buffer
           destinationOffset:NSUInteger(destOffsetBytes)
                        size:NSUInteger(dataSize)];
        m_BlitEncoderWrites.push_back(destination);
    }

    void CommandList::clearBufferUInt(IBuffer* b, uint32_t clearValue)
    {
        auto* buffer = static_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return;

        const size_t clearSize = size_t(buffer->desc.byteSize);
        if (clearSize == 0)
            return;
        referenceBuffer(buffer);

        if (clearValue == 0)
        {
            // This clear must be ordered with later GPU work. CPU writes into a
            // shared buffer can race already-encoded dispatches, which lets GPU
            // append counters accumulate across culling passes. A blit fill is
            // recorded in the command buffer, so Metal executes it before the
            // following compute encoder.
            endEncoding();
            id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
            beginEncoding(blit, __func__);
            [blit fillBuffer:buffer->buffer
                       range:NSMakeRange(0, NSUInteger(buffer->desc.byteSize))
                       value:0];
            endEncoding(blit);
            return;
        }
        // repeatable pattern does the same style blit
        const uint8_t bytePattern = uint8_t(clearValue & 0xffu);
        const bool isRepeatedBytePattern = clearValue == uint32_t(bytePattern) * 0x01010101u;
        if (isRepeatedBytePattern)
        {
            endEncoding();
            id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
            beginEncoding(blit, __func__);
            [blit fillBuffer:buffer->buffer
                       range:NSMakeRange(0, NSUInteger(clearSize))
                       value:bytePattern];
            endEncoding(blit);
            return;
        }

        // if no repeatable pattern, perform the copy from buffer with UploadManager
        const size_t wordCount = clearSize / sizeof(uint32_t);
        const size_t uploadSize = wordCount * sizeof(uint32_t);
        if (uploadSize == 0)
            return;

        UploadAllocation upload = m_UploadManager.suballocate(uploadSize, 256);
        if (!upload.buffer || !upload.cpuAddress)
            return;

        uint32_t* words = static_cast<uint32_t*>(upload.cpuAddress);
        for (size_t i = 0; i < wordCount; ++i)
            words[i] = clearValue;

        endEncoding();
        id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
        beginEncoding(blit, __func__);
        [blit copyFromBuffer:upload.buffer
                sourceOffset:upload.offset
                    toBuffer:buffer->buffer
           destinationOffset:0
                        size:NSUInteger(uploadSize)];
        endEncoding(blit);
    }

    void CommandList::copyBuffer(IBuffer* dest, uint64_t destOffsetBytes, IBuffer* src, uint64_t srcOffsetBytes, uint64_t dataSizeBytes)
    {
        auto* d = static_cast<Buffer*>(dest);
        auto* s = static_cast<Buffer*>(src);
        if (!d || !s || !d->buffer || !s->buffer || dataSizeBytes == 0)
            return;
        // volatile buffers should always be handled with cmdList.writeBuffer, not copyBuffer
        if (d->desc.isVolatile || s->desc.isVolatile)
            return;

        if (srcOffsetBytes > s->desc.byteSize || dataSizeBytes > s->desc.byteSize - srcOffsetBytes)
            return;

        if (destOffsetBytes > d->desc.byteSize || dataSizeBytes > d->desc.byteSize - destOffsetBytes)
            return;
        referenceBuffer(d);
        referenceBuffer(s);

        endEncoding();
        id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
        beginEncoding(blit, __func__);
        [blit copyFromBuffer:s->buffer
                sourceOffset:NSUInteger(srcOffsetBytes)
                    toBuffer:d->buffer
           destinationOffset:NSUInteger(destOffsetBytes)
                        size:NSUInteger(dataSizeBytes)];
        endEncoding(blit);
    }

    void CommandList::setGraphicsState(const GraphicsState& state)
    {
        if (m_CurrentFramebuffer != state.framebuffer)
            endEncoding();
        m_CurrentGraphicsState = state;
        m_CurrentFramebuffer = state.framebuffer;
        m_CurrentGraphicsStateValid = true;
        m_CurrentMeshletStateValid = false;
        if (state.pipeline)
            m_ReferencedStateResources.emplace_back(state.pipeline);
        if (state.framebuffer)
            m_ReferencedStateResources.emplace_back(state.framebuffer);
        for (const VertexBufferBinding& binding : state.vertexBuffers)
            referenceBuffer(binding.buffer);
        referenceBuffer(state.indexBuffer.buffer);
        referenceBuffer(state.indirectParams);
        referenceBuffer(state.indirectCountBuffer);
        if (m_EnableAutomaticBarriers)
        {
            for (IBindingSet* bindingSet : state.bindings)
                setResourceStatesForBindingSet(bindingSet);
            for (const VertexBufferBinding& binding : state.vertexBuffers)
                setBufferState(binding.buffer, ResourceStates::VertexBuffer);
            setBufferState(state.indexBuffer.buffer, ResourceStates::IndexBuffer);
            setBufferState(state.indirectParams, ResourceStates::IndirectArgument);
            setBufferState(state.indirectCountBuffer, ResourceStates::IndirectArgument);
            if (state.framebuffer)
            {
                for (const FramebufferAttachment& attachment : state.framebuffer->getDesc().colorAttachments)
                    setTextureState(attachment.texture, attachment.subresources, ResourceStates::RenderTarget);
                const FramebufferAttachment& depth = state.framebuffer->getDesc().depthAttachment;
                setTextureState(depth.texture, depth.subresources, depth.isReadOnly ? ResourceStates::DepthRead : ResourceStates::DepthWrite);
            }
            commitBarriers();
        }
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(state.pipeline);
        if (!encoder || !pipeline)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] setGraphicsState skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no"));
            return;
        }

        applyGraphicsStateToEncoder(encoder, state);
    }

    id<MTLRenderCommandEncoder> CommandList::getOrCreateRenderEncoder()
    {
        if (m_RenderEncoder)
            return m_RenderEncoder;
        endEncoding();

        auto* framebuffer = static_cast<Framebuffer*>(m_CurrentFramebuffer);
        if (!framebuffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] render encoder requested without framebuffer");
            return nil;
        }

        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        for (NSUInteger index = 0; index < framebuffer->desc.colorAttachments.size(); ++index)
        {
            const auto& attachment = framebuffer->desc.colorAttachments[index];
            auto* texture = static_cast<Texture*>(framebuffer->desc.colorAttachments[index].texture);
            rp.colorAttachments[index].texture = texture ? texture->texture : nil;
            rp.colorAttachments[index].level = attachment.subresources.baseMipLevel;
            rp.colorAttachments[index].slice = attachment.subresources.baseArraySlice;
            rp.colorAttachments[index].loadAction = MTLLoadActionLoad;
            rp.colorAttachments[index].storeAction = MTLStoreActionStore;
            for (auto it = m_PendingClears.begin(); it != m_PendingClears.end(); ++it)
            {
                if (it->texture != texture || it->isDepthStencil || it->mipLevel != attachment.subresources.baseMipLevel ||
                    it->arraySlice != attachment.subresources.baseArraySlice)
                    continue;
                rp.colorAttachments[index].loadAction = MTLLoadActionClear;
                rp.colorAttachments[index].clearColor = MTLClearColorMake(it->color.r, it->color.g, it->color.b, it->color.a);
                m_PendingClears.erase(it);
                break;
            }
        }
        if (framebuffer->desc.depthAttachment.texture)
        {
            const auto& attachment = framebuffer->desc.depthAttachment;
            auto* depth = static_cast<Texture*>(framebuffer->desc.depthAttachment.texture);
            const bool hasStencil = depth && getFormatInfo(depth->desc.format).hasStencil;
            rp.depthAttachment.texture = depth ? depth->texture : nil;
            rp.depthAttachment.level = attachment.subresources.baseMipLevel;
            rp.depthAttachment.slice = attachment.subresources.baseArraySlice;
            rp.depthAttachment.loadAction = MTLLoadActionLoad;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
            if (hasStencil)
            {
                rp.stencilAttachment.texture = depth->texture;
                rp.stencilAttachment.level = attachment.subresources.baseMipLevel;
                rp.stencilAttachment.slice = attachment.subresources.baseArraySlice;
                rp.stencilAttachment.loadAction = MTLLoadActionLoad;
                rp.stencilAttachment.storeAction = MTLStoreActionStore;
            }
            for (auto it = m_PendingClears.begin(); it != m_PendingClears.end(); ++it)
            {
                if (it->texture != depth || !it->isDepthStencil || it->mipLevel != attachment.subresources.baseMipLevel ||
                    it->arraySlice != attachment.subresources.baseArraySlice)
                    continue;
                if (it->clearDepth)
                {
                    rp.depthAttachment.loadAction = MTLLoadActionClear;
                    rp.depthAttachment.clearDepth = it->depth;
                }
                if (it->clearStencil && hasStencil)
                {
                    rp.stencilAttachment.loadAction = MTLLoadActionClear;
                    rp.stencilAttachment.clearStencil = it->stencil;
                }
                m_PendingClears.erase(it);
                break;
            }
        }
        flushPendingClears();
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        beginTracyRenderEncoderZone(rp);
#endif

        m_RenderEncoderWrites.clear();
        m_RenderEncoder = [trackedCmdBuffer renderCommandEncoderWithDescriptor:rp];
        beginEncoding(m_RenderEncoder, "render");
        if (!m_RenderEncoder)
            m_Context.error("[metal3-trace] failed to create render command encoder");
        return m_RenderEncoder;
    }

    id<MTLComputeCommandEncoder> CommandList::getOrCreateComputeEncoder()
    {
        if (m_ComputeEncoder)
            return m_ComputeEncoder;
        endEncoding();
        flushPendingClears();
        MTLComputePassDescriptor* cp = [MTLComputePassDescriptor computePassDescriptor];
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        beginTracyComputeEncoderZone(cp);
#endif
        m_ComputeEncoder = [trackedCmdBuffer computeCommandEncoderWithDescriptor:cp];
        beginEncoding(m_ComputeEncoder, "compute");
        if (!m_ComputeEncoder)
            m_Context.error("[metal3-trace] failed to create compute command encoder");
        return m_ComputeEncoder;
    }

    id<MTLBlitCommandEncoder> CommandList::beginBlitEncoder()
    {
        endEncoding();
        flushPendingClears();
        return [trackedCmdBuffer blitCommandEncoder];
    }

    id<MTLComputeCommandEncoder> CommandList::beginTransientComputeEncoder()
    {
        endEncoding();
        flushPendingClears();
        return [trackedCmdBuffer computeCommandEncoder];
    }
    
    void CommandList::draw(const DrawArguments& args)
    {
        if (!m_CurrentGraphicsStateValid)
            return;
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        if (!encoder || !pipeline)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] draw skipped: encoder/pipeline missing");
            return;
        }
        if (m_BindingStatesDirty)
            applyGraphicsStateToEncoder(encoder, m_CurrentGraphicsState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();
        if (pipeline->usesGeometryEmulation)
        {
            if (!m_GeometryEmulationDrawStateValid)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] draw skipped: geometry-emulation state was not fully bound");
                return;
            }

            IRRuntimeGeometryPipelineConfig config{};
            config.gsVertexSizeInBytes = pipeline->geometryVertexSizeInBytes;
            config.gsMaxInputPrimitivesPerMeshThreadgroup =
                pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup;

            IRRuntimeDrawPrimitivesGeometryEmulation(encoder,
                convertRuntimePrimitiveType(pipeline->desc.primType),
                config,
                args.instanceCount,
                args.vertexCount,
                args.startVertexLocation,
                args.startInstanceLocation);
            return;
        }

        IRRuntimeDrawPrimitives(encoder, pipeline->primitiveType, args.startVertexLocation, args.vertexCount, args.instanceCount, args.startInstanceLocation);
    }

    void CommandList::drawIndexed(const DrawArguments& args)
    {
        if (!m_CurrentGraphicsStateValid)
            return;
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indexBuffer = static_cast<Buffer*>(m_CurrentGraphicsState.indexBuffer.buffer);
        if (!encoder || !pipeline || !indexBuffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexed skipped: encoder/pipeline/index missing");
            return;
        }
        if (m_BindingStatesDirty)
            applyGraphicsStateToEncoder(encoder, m_CurrentGraphicsState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();
        if (pipeline->usesGeometryEmulation)
        {
            if (!m_GeometryEmulationDrawStateValid)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] drawIndexed skipped: geometry-emulation state was not fully bound");
                return;
            }
            if (!indexBuffer->buffer)
                return;

            IRRuntimeGeometryPipelineConfig config{};
            config.gsVertexSizeInBytes = pipeline->geometryVertexSizeInBytes;
            config.gsMaxInputPrimitivesPerMeshThreadgroup =
                pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup;

            const uint32_t indexSize = m_CurrentGraphicsState.indexBuffer.format == Format::R32_UINT ? 4u : 2u;
            if (m_CurrentGraphicsState.indexBuffer.offset % indexSize != 0 && traceMetalRuntime())
                m_Context.warning("[metal3-trace] geometry-emulation indexed draw has an unaligned index buffer offset");

            const uint32_t startIndex =
                uint32_t(m_CurrentGraphicsState.indexBuffer.offset / indexSize) + args.startIndexLocation;
            IRRuntimeDrawIndexedPrimitivesGeometryEmulation(encoder,
                convertRuntimePrimitiveType(pipeline->desc.primType),
                convertIndexFormat(m_CurrentGraphicsState.indexBuffer.format),
                indexBuffer->buffer,
                config,
                args.instanceCount,
                args.vertexCount,
                startIndex,
                args.startVertexLocation,
                args.startInstanceLocation);
            return;
        }

        IRRuntimeDrawIndexedPrimitives(encoder, pipeline->primitiveType, args.vertexCount, convertIndexFormat(m_CurrentGraphicsState.indexBuffer.format), indexBuffer->buffer, m_CurrentGraphicsState.indexBuffer.offset + args.startIndexLocation * (m_CurrentGraphicsState.indexBuffer.format == Format::R32_UINT ? 4 : 2), args.instanceCount, args.startVertexLocation, args.startInstanceLocation);
    }

    void CommandList::drawIndirect(uint32_t offsetBytes, uint32_t drawCount)
    {
        if (!m_CurrentGraphicsStateValid)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndirect skipped: graphics state invalid");
            return;
        }
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        if (!encoder || !pipeline || !indirectParams || !indirectParams->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndirect skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no") +
                    " indirect='" + (indirectParams ? indirectParams->desc.debugName : std::string("<null>")) + "'");
            return;
        }
        if (m_BindingStatesDirty)
            applyGraphicsStateToEncoder(encoder, m_CurrentGraphicsState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();

        // uncomment for debugging
        // static int drawIndirectLogCount = 0;
        // if (traceMetalRuntime() && drawIndirectLogCount++ < 64)
        //     m_Context.info("[metal3-trace] drawIndirect count=" + std::to_string(drawCount) +
        //         " offset=" + std::to_string(offsetBytes) + " args='" + indirectParams->desc.debugName + "'");

        if (pipeline->usesGeometryEmulation)
        {
            drawIndirectGeometryEmulation(offsetBytes, drawCount);
            return;
        }

        for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex)
        {
            // Metal Shader Converter vertex shaders read draw parameters from
            // the IR runtime bind points. Use the runtime wrapper for indirect
            // draws too; raw Metal draws do not populate those translated
            // D3D-style draw parameter bindings.
            IRRuntimeDrawPrimitives(encoder, pipeline->primitiveType, indirectParams->buffer, offsetBytes);
            offsetBytes += sizeof(DrawIndirectArguments);
        }
    }

    void CommandList::drawIndirectGeometryEmulation(uint32_t offsetBytes, uint32_t drawCount)
    {
        if (drawCount == 0)
            return;
        if (!m_GeometryEmulationDrawStateValid)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndirect skipped: geometry-emulation state was not fully bound");
            return;
        }

        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        if (!pipeline || !indirectParams || !indirectParams->buffer)
            return;

        const uint64_t requiredBytes = uint64_t(offsetBytes) + uint64_t(drawCount) * sizeof(DrawIndirectArguments);
        if (requiredBytes > indirectParams->desc.byteSize)
        {
            m_Context.warning("[metal3-trace] geometry-emulation drawIndirect range exceeds args buffer: required=" +
                std::to_string(requiredBytes) + " bufferSize=" + std::to_string(indirectParams->desc.byteSize) +
                " args='" + indirectParams->desc.debugName + "'");
            return;
        }

        if (@available(macOS 13.0, *))
        {
            const GeometryIndirectFillState& fillState = getGeometryIndirectFillState(m_Context);
            if (!fillState.pipeline)
            {
                m_Context.warning("[metal3] geometry-emulation drawIndirect skipped because the helper pipeline is unavailable");
                return;
            }

            const NSUInteger drawInfoStride = sizeof(IRRuntimeDrawInfo);
            const NSUInteger drawParamsStride = sizeof(IRRuntimeDrawParams);
            const NSUInteger meshArgsStride = sizeof(MTLDispatchThreadgroupsIndirectArguments);
            const NSUInteger drawInfoSize = drawInfoStride * NSUInteger(drawCount);
            const NSUInteger drawParamsSize = drawParamsStride * NSUInteger(drawCount);
            const NSUInteger meshArgsSize = meshArgsStride * NSUInteger(drawCount);

            // The helper compute pass expands each NVRHI DrawIndirectArguments
            // record into the three buffers consumed by IRRuntime GS emulation:
            // draw info, draw params, and Metal mesh indirect dispatch args.
            // These are distinct slices for this call. They share native pages
            // with other calls only at non-overlapping offsets.
            const TransientBufferAllocation drawInfoAllocation = m_TransientIndirectResources.allocatePrivate(drawInfoSize);
            const TransientBufferAllocation drawParams = m_TransientIndirectResources.allocatePrivate(drawParamsSize);
            const TransientBufferAllocation meshIndirectArgs = m_TransientIndirectResources.allocatePrivate(meshArgsSize);
            const TransientBufferAllocation paramsAllocation =
                m_TransientIndirectResources.allocateShared(sizeof(GeometryIndirectParams));
            if (!drawInfoAllocation.buffer || !drawParams.buffer || !meshIndirectArgs.buffer ||
                !paramsAllocation.buffer || !paramsAllocation.cpuAddress)
            {
                m_Context.error("[metal3] failed to allocate geometry-emulation drawIndirect resources");
                return;
            }

            auto* params = static_cast<GeometryIndirectParams*>(paramsAllocation.cpuAddress);
            // Non-indexed draws do not need index-buffer metadata; the helper
            // only reads vertexCount/instance/start values from indirectParams.
            params->drawCount = drawCount;
            params->paramOffsetBytes = offsetBytes;
            params->primitiveType = uint32_t(convertRuntimePrimitiveType(pipeline->desc.primType));
            params->gsVertexSizeInBytes = pipeline->geometryVertexSizeInBytes;
            params->gsMaxInputPrimitivesPerMeshThreadgroup =
                pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup;
            params->indexed = 0;
            params->indexType = 0;
            params->indexBufferOffsetInElements = 0;
            params->indexBufferAddress = 0;

            m_ReferencedNativeBuffers.push_back(drawInfoAllocation.buffer);
            m_ReferencedNativeBuffers.push_back(drawParams.buffer);
            m_ReferencedNativeBuffers.push_back(meshIndirectArgs.buffer);
            m_ReferencedNativeBuffers.push_back(paramsAllocation.buffer);

            // Leave the render pass, run the compute expansion, then restore the
            // geometry-emulation graphics state before issuing mesh indirect draws.
            endEncoding();
            id<MTLComputeCommandEncoder> compute = beginTransientComputeEncoder();
            beginEncoding(compute, __func__);
            [compute setComputePipelineState:fillState.pipeline];
            [compute setBuffer:indirectParams->buffer offset:0 atIndex:0];
            [compute setBuffer:drawInfoAllocation.buffer offset:drawInfoAllocation.offset atIndex:1];
            [compute setBuffer:drawParams.buffer offset:drawParams.offset atIndex:2];
            [compute setBuffer:meshIndirectArgs.buffer offset:meshIndirectArgs.offset atIndex:3];
            [compute setBuffer:paramsAllocation.buffer offset:paramsAllocation.offset atIndex:4];

            const NSUInteger threads = std::max<NSUInteger>(1, fillState.pipeline.threadExecutionWidth);
            const NSUInteger groups = (NSUInteger(drawCount) + threads - 1) / threads;
            [compute dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            endEncoding(compute);

            id<MTLRenderCommandEncoder> renderEncoder = getOrCreateRenderEncoder();
            if (!renderEncoder)
                return;

            applyGraphicsStateToEncoder(renderEncoder, m_CurrentGraphicsState);
            if (!m_GeometryEmulationDrawStateValid)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] drawIndirect skipped: geometry-emulation state restore failed");
                return;
            }

            IRRuntimeGeometryPipelineConfig config{};
            config.gsVertexSizeInBytes = pipeline->geometryVertexSizeInBytes;
            config.gsMaxInputPrimitivesPerMeshThreadgroup =
                pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup;
            IRRuntimeDrawInfo drawInfo = IRRuntimeCalculateDrawInfoForGSEmulation(
                convertRuntimePrimitiveType(pipeline->desc.primType),
                (MTLIndexType)-1,
                config.gsVertexSizeInBytes,
                config.gsMaxInputPrimitivesPerMeshThreadgroup,
                1);
            drawInfo.indexType = kIRNonIndexedDraw;

            uint32_t objectThreadgroupSize = 1;
            uint32_t meshThreadgroupSize = 1;
            IRRuntimeCalculateThreadgroupSizeForGeometry(
                convertRuntimePrimitiveType(pipeline->desc.primType),
                config.gsMaxInputPrimitivesPerMeshThreadgroup,
                drawInfo.objectThreadgroupVertexStride,
                &objectThreadgroupSize,
                &meshThreadgroupSize);


            // Each NVRHI indirect draw became one Metal mesh indirect dispatch.
            // Bind that draw's IRRuntime records and execute the GPU-written
            // threadgroup dimensions.
            for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex)
            {
                [renderEncoder setObjectBuffer:drawInfoAllocation.buffer
                                        offset:drawInfoAllocation.offset + drawInfoStride * NSUInteger(drawIndex)
                                       atIndex:kIRArgumentBufferUniformsBindPoint];
                [renderEncoder setMeshBuffer:drawInfoAllocation.buffer
                                      offset:drawInfoAllocation.offset + drawInfoStride * NSUInteger(drawIndex)
                                     atIndex:kIRArgumentBufferUniformsBindPoint];
                [renderEncoder setObjectBuffer:drawParams.buffer
                                        offset:drawParams.offset + drawParamsStride * NSUInteger(drawIndex)
                                       atIndex:kIRArgumentBufferDrawArgumentsBindPoint];
                [renderEncoder setMeshBuffer:drawParams.buffer
                                      offset:drawParams.offset + drawParamsStride * NSUInteger(drawIndex)
                                     atIndex:kIRArgumentBufferDrawArgumentsBindPoint];

                [renderEncoder drawMeshThreadgroupsWithIndirectBuffer:meshIndirectArgs.buffer
                                                 indirectBufferOffset:meshIndirectArgs.offset + meshArgsStride * NSUInteger(drawIndex)
                                          threadsPerObjectThreadgroup:MTLSizeMake(objectThreadgroupSize, 1, 1)
                                            threadsPerMeshThreadgroup:MTLSizeMake(meshThreadgroupSize, 1, 1)];
            }
        }
        else if (traceMetalRuntime())
        {
            m_Context.warning("[metal3-trace] geometry-emulation drawIndirect requires mesh draw support");
        }
    }
    
    void CommandList::drawIndexedIndirect(uint32_t offsetBytes, uint32_t drawCount)
    {
        if (!m_CurrentGraphicsStateValid)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexedIndirect skipped: graphics state invalid");
            return;
        }
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indexBuffer = static_cast<Buffer*>(m_CurrentGraphicsState.indexBuffer.buffer);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        if (!encoder || !pipeline || !indexBuffer || !indexBuffer->buffer || !indirectParams || !indirectParams->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexedIndirect skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no") +
                    " index='" + (indexBuffer ? indexBuffer->desc.debugName : std::string("<null>")) +
                    "' indirect='" + (indirectParams ? indirectParams->desc.debugName : std::string("<null>")) + "'");
            return;
        }
        if (m_BindingStatesDirty)
            applyGraphicsStateToEncoder(encoder, m_CurrentGraphicsState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();

        const uint64_t requiredBytes = uint64_t(offsetBytes) + uint64_t(drawCount) * sizeof(DrawIndexedIndirectArguments);
        if (requiredBytes > indirectParams->desc.byteSize)
            m_Context.warning("[metal3-trace] drawIndexedIndirect range exceeds args buffer: required=" +
                std::to_string(requiredBytes) + " bufferSize=" + std::to_string(indirectParams->desc.byteSize) +
                " args='" + indirectParams->desc.debugName + "'");

        static int drawIndexedIndirectLogCount = 0;
        if (traceMetalRuntime() && drawIndexedIndirectLogCount++ < 96)
            m_Context.info("[metal3-trace] drawIndexedIndirect count=" + std::to_string(drawCount) +
                " offset=" + std::to_string(offsetBytes) +
                " index='" + indexBuffer->desc.debugName +
                "' args='" + indirectParams->desc.debugName + "'");

        if (pipeline->usesGeometryEmulation)
        {
            drawIndexedIndirectGeometryEmulation(offsetBytes, drawCount);
            return;
        }

        for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex)
        {
            // Keep converted vertex shaders in the same draw-parameter path as
            // direct indexed draws. This is especially important for indirect
            // base-instance/start-instance data produced by GPU culling.
            IRRuntimeDrawIndexedPrimitives(encoder,
                pipeline->primitiveType,
                convertIndexFormat(m_CurrentGraphicsState.indexBuffer.format),
                indexBuffer->buffer,
                m_CurrentGraphicsState.indexBuffer.offset,
                indirectParams->buffer,
                offsetBytes);
            offsetBytes += sizeof(DrawIndexedIndirectArguments);
        }
    }

    void CommandList::drawIndexedIndirectGeometryEmulation(uint32_t offsetBytes, uint32_t drawCount)
    {
        if (drawCount == 0)
            return;
        if (!m_GeometryEmulationDrawStateValid)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexedIndirect skipped: geometry-emulation state was not fully bound");
            return;
        }

        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indexBuffer = static_cast<Buffer*>(m_CurrentGraphicsState.indexBuffer.buffer);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        if (!pipeline || !indexBuffer || !indexBuffer->buffer || !indirectParams || !indirectParams->buffer)
            return;

        const uint64_t requiredBytes = uint64_t(offsetBytes) + uint64_t(drawCount) * sizeof(DrawIndexedIndirectArguments);
        if (requiredBytes > indirectParams->desc.byteSize)
        {
            m_Context.warning("[metal3-trace] geometry-emulation drawIndexedIndirect range exceeds args buffer: required=" +
                std::to_string(requiredBytes) + " bufferSize=" + std::to_string(indirectParams->desc.byteSize) +
                " args='" + indirectParams->desc.debugName + "'");
            return;
        }

        const uint32_t indexSize = m_CurrentGraphicsState.indexBuffer.format == Format::R32_UINT ? 4u : 2u;
        if (m_CurrentGraphicsState.indexBuffer.offset % indexSize != 0 && traceMetalRuntime())
            m_Context.warning("[metal3-trace] geometry-emulation indexed indirect draw has an unaligned index buffer offset");

        if (@available(macOS 13.0, *))
        {
            const GeometryIndirectFillState& fillState = getGeometryIndirectFillState(m_Context);
            if (!fillState.pipeline)
            {
                m_Context.warning("[metal3] geometry-emulation drawIndexedIndirect skipped because the helper pipeline is unavailable");
                return;
            }

            const NSUInteger drawInfoStride = sizeof(IRRuntimeDrawInfo);
            const NSUInteger drawParamsStride = sizeof(IRRuntimeDrawParams);
            const NSUInteger meshArgsStride = sizeof(MTLDispatchThreadgroupsIndirectArguments);
            const NSUInteger drawInfoSize = drawInfoStride * NSUInteger(drawCount);
            const NSUInteger drawParamsSize = drawParamsStride * NSUInteger(drawCount);
            const NSUInteger meshArgsSize = meshArgsStride * NSUInteger(drawCount);

            // Indexed geometry emulation uses the same helper expansion as
            // drawIndirect, plus index-buffer metadata for mesh-stage index fetch.
            // These are distinct slices for this call. They share native pages
            // with other calls only at non-overlapping offsets.
            const TransientBufferAllocation drawInfoAllocation = m_TransientIndirectResources.allocatePrivate(drawInfoSize);
            const TransientBufferAllocation drawParams = m_TransientIndirectResources.allocatePrivate(drawParamsSize);
            const TransientBufferAllocation meshIndirectArgs = m_TransientIndirectResources.allocatePrivate(meshArgsSize);
            const TransientBufferAllocation paramsAllocation =
                m_TransientIndirectResources.allocateShared(sizeof(GeometryIndirectParams));
            if (!drawInfoAllocation.buffer || !drawParams.buffer || !meshIndirectArgs.buffer ||
                !paramsAllocation.buffer || !paramsAllocation.cpuAddress)
            {
                m_Context.error("[metal3] failed to allocate geometry-emulation drawIndexedIndirect resources");
                return;
            }

            auto* params = static_cast<GeometryIndirectParams*>(paramsAllocation.cpuAddress);
            // The helper reads DrawIndexedIndirectArguments from indirectParams
            // and patches startIndex with the bound NVRHI index-buffer offset.
            params->drawCount = drawCount;
            params->paramOffsetBytes = offsetBytes;
            params->primitiveType = uint32_t(convertRuntimePrimitiveType(pipeline->desc.primType));
            params->gsVertexSizeInBytes = pipeline->geometryVertexSizeInBytes;
            params->gsMaxInputPrimitivesPerMeshThreadgroup =
                pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup;
            params->indexed = 1;
            params->indexType = uint32_t(convertIndexFormat(m_CurrentGraphicsState.indexBuffer.format));
            params->indexBufferOffsetInElements =
                uint32_t(m_CurrentGraphicsState.indexBuffer.offset / indexSize);
            params->indexBufferAddress = indexBuffer->getGpuVirtualAddress();

            m_ReferencedNativeBuffers.push_back(drawInfoAllocation.buffer);
            m_ReferencedNativeBuffers.push_back(drawParams.buffer);
            m_ReferencedNativeBuffers.push_back(meshIndirectArgs.buffer);
            m_ReferencedNativeBuffers.push_back(paramsAllocation.buffer);

            // Generate IRRuntime per-draw records and mesh indirect args on the
            // GPU, because the original indexed indirect args are GPU-owned.
            endEncoding();
            id<MTLComputeCommandEncoder> compute = beginTransientComputeEncoder();
            beginEncoding(compute, __func__);
            [compute setComputePipelineState:fillState.pipeline];
            [compute setBuffer:indirectParams->buffer offset:0 atIndex:0];
            [compute setBuffer:drawInfoAllocation.buffer offset:drawInfoAllocation.offset atIndex:1];
            [compute setBuffer:drawParams.buffer offset:drawParams.offset atIndex:2];
            [compute setBuffer:meshIndirectArgs.buffer offset:meshIndirectArgs.offset atIndex:3];
            [compute setBuffer:paramsAllocation.buffer offset:paramsAllocation.offset atIndex:4];

            const NSUInteger threads = std::max<NSUInteger>(1, fillState.pipeline.threadExecutionWidth);
            const NSUInteger groups = (NSUInteger(drawCount) + threads - 1) / threads;
            [compute dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            endEncoding(compute);

            id<MTLRenderCommandEncoder> renderEncoder = getOrCreateRenderEncoder();
            if (!renderEncoder)
                return;

            applyGraphicsStateToEncoder(renderEncoder, m_CurrentGraphicsState);
            if (!m_GeometryEmulationDrawStateValid)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] drawIndexedIndirect skipped: geometry-emulation state restore failed");
                return;
            }

            IRRuntimeGeometryPipelineConfig config{};
            config.gsVertexSizeInBytes = pipeline->geometryVertexSizeInBytes;
            config.gsMaxInputPrimitivesPerMeshThreadgroup =
                pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup;
            IRRuntimeDrawInfo drawInfo = IRRuntimeCalculateDrawInfoForGSEmulation(
                convertRuntimePrimitiveType(pipeline->desc.primType),
                convertIndexFormat(m_CurrentGraphicsState.indexBuffer.format),
                config.gsVertexSizeInBytes,
                config.gsMaxInputPrimitivesPerMeshThreadgroup,
                1);

            uint32_t objectThreadgroupSize = 1;
            uint32_t meshThreadgroupSize = 1;
            IRRuntimeCalculateThreadgroupSizeForGeometry(
                convertRuntimePrimitiveType(pipeline->desc.primType),
                config.gsMaxInputPrimitivesPerMeshThreadgroup,
                drawInfo.objectThreadgroupVertexStride,
                &objectThreadgroupSize,
                &meshThreadgroupSize);


            // Replay the expanded draws through the mesh pipeline. The object and
            // mesh stages receive the generated draw info and draw params for the
            // current indirect draw entry.
            for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex)
            {
                [renderEncoder setObjectBuffer:drawInfoAllocation.buffer
                                        offset:drawInfoAllocation.offset + drawInfoStride * NSUInteger(drawIndex)
                                       atIndex:kIRArgumentBufferUniformsBindPoint];
                [renderEncoder setMeshBuffer:drawInfoAllocation.buffer
                                      offset:drawInfoAllocation.offset + drawInfoStride * NSUInteger(drawIndex)
                                     atIndex:kIRArgumentBufferUniformsBindPoint];
                [renderEncoder setObjectBuffer:drawParams.buffer
                                        offset:drawParams.offset + drawParamsStride * NSUInteger(drawIndex)
                                       atIndex:kIRArgumentBufferDrawArgumentsBindPoint];
                [renderEncoder setMeshBuffer:drawParams.buffer
                                      offset:drawParams.offset + drawParamsStride * NSUInteger(drawIndex)
                                     atIndex:kIRArgumentBufferDrawArgumentsBindPoint];

                [renderEncoder drawMeshThreadgroupsWithIndirectBuffer:meshIndirectArgs.buffer
                                                 indirectBufferOffset:meshIndirectArgs.offset + meshArgsStride * NSUInteger(drawIndex)
                                          threadsPerObjectThreadgroup:MTLSizeMake(objectThreadgroupSize, 1, 1)
                                            threadsPerMeshThreadgroup:MTLSizeMake(meshThreadgroupSize, 1, 1)];
            }
        }
        else if (traceMetalRuntime())
        {
            m_Context.warning("[metal3-trace] geometry-emulation drawIndexedIndirect requires mesh draw support");
        }
    }

    void CommandList::drawIndexedIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount)
    {
        if (!m_CurrentGraphicsStateValid || maxDrawCount == 0)
            return;

        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indexBuffer = static_cast<Buffer*>(m_CurrentGraphicsState.indexBuffer.buffer);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        auto* indirectCount = static_cast<Buffer*>(m_CurrentGraphicsState.indirectCountBuffer);
        if (!pipeline || !indexBuffer || !indexBuffer->buffer || !indirectParams || !indirectParams->buffer ||
            !indirectCount || !indirectCount->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexedIndirectCount skipped: missing pipeline/index/args/count buffer");
            return;
        }

        if (pipeline->usesGeometryEmulation)
        {
            // Converted geometry shaders run as Metal object/mesh pipelines, so
            // the normal indexed-draw ICB path cannot replay them. Use the GS
            // emulation path that writes mesh-threadgroup commands
            drawIndexedIndirectCountGeometryEmulation(paramOffsetBytes, countOffsetBytes, maxDrawCount);
            return;
        }

        const IndexedIndirectIcbFillState& fillState = getIndexedIndirectIcbFillState(m_Context);
        if (!fillState.pipeline || !fillState.icbEncoder)
        {
            m_Context.warning("[metal3] counted indexed indirect draw skipped because the ICB helper pipeline is unavailable");
            return;
        }

        id<MTLRenderCommandEncoder> renderEncoder = getOrCreateRenderEncoder();
        if (!renderEncoder)
            return;
        if (m_BindingStatesDirty)
            applyGraphicsStateToEncoder(renderEncoder, m_CurrentGraphicsState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();
        if (m_RecordingFailed)
            return;
        uint32_t bufferBindingCount = 1;
        NSUInteger vertexBufferBindCount = kIRArgumentBufferUniformsBindPoint + 1;
        NSUInteger fragmentBufferBindCount = 0;
        for (NSUInteger index = 0; index < m_VertexBufferBindings.size(); ++index)
        {
            if (m_VertexBufferBindings[index].buffer)
            {
                ++bufferBindingCount;
                vertexBufferBindCount = index + 1;
            }
            if (m_FragmentBufferBindings[index].buffer)
            {
                ++bufferBindingCount;
                fragmentBufferBindCount = index + 1;
            }
        }
        vertexBufferBindCount = std::max<NSUInteger>(vertexBufferBindCount, kIRArgumentBufferUniformsBindPoint + 1);
        endEncoding();
        // Metal supports indexed indirect draws, but not NVRHI's counted
        // indexed indirect operation directly. Generate a render ICB on the GPU
        // from the same args/count buffers and execute it with a GPU-written
        // MTLIndirectCommandBufferExecutionRange.

        MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeDrawIndexed;
        icbDesc.inheritPipelineState = YES;
        icbDesc.inheritBuffers = NO;
        icbDesc.maxVertexBufferBindCount = vertexBufferBindCount;
        icbDesc.maxFragmentBufferBindCount = fragmentBufferBindCount;

        // The page slices have nonzero offsets; propagate those exact offsets
        // through the encoder and execute calls below.
        NSUInteger icbCapacity = 0;
        id<MTLIndirectCommandBuffer> icb =
            m_TransientIndirectResources.acquireIndirectCommandBuffer(icbDesc, NSUInteger(maxDrawCount), nullptr, &icbCapacity);
        const TransientBufferAllocation executionRange =
            m_TransientIndirectResources.allocatePrivate(sizeof(MTLIndirectCommandBufferExecutionRange));
        const TransientBufferAllocation paramsAllocation =
            m_TransientIndirectResources.allocateShared(sizeof(IcbIndexedIndirectParams) +
                bufferBindingCount * sizeof(IcbGraphicsBufferBinding));
        const TransientBufferAllocation icbArgumentBuffer =
            m_TransientIndirectResources.allocateShared(fillState.icbEncoder.encodedLength);
        if (!icb || !executionRange.buffer || !paramsAllocation.buffer || !paramsAllocation.cpuAddress ||
            !icbArgumentBuffer.buffer || !icbArgumentBuffer.cpuAddress)
        {
            m_Context.error("[metal3] failed to allocate counted indexed indirect ICB resources");
            return;
        }

        // Metal compute shaders receive an indirect command buffer through an
        // argument buffer. This encodes the ICB object into that argument buffer
        // so nvrhi_metal3_fill_indexed_indirect_icb can write render commands.
        [fillState.icbEncoder setArgumentBuffer:icbArgumentBuffer.buffer offset:icbArgumentBuffer.offset];
        [fillState.icbEncoder setIndirectCommandBuffer:icb atIndex:0];

        // Small CPU-written constant block consumed by the helper shader. It
        // describes where to read draw args/counts, how to interpret the index
        // buffer, and which MSC runtime vertex-buffer slot should receive each
        // draw's original D3D-style indirect argument record.
        auto* params = reinterpret_cast<IcbIndexedIndirectParams*>(paramsAllocation.cpuAddress);
        params->maxDrawCount = maxDrawCount;
        params->paramOffsetBytes = paramOffsetBytes;
        params->countOffsetBytes = countOffsetBytes;
        params->indexType = IRMetalIndexToIRIndex(m_CurrentGraphicsState.indexBuffer.format == Format::R16_UINT
            ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32);
        params->indexBufferOffset = uint32_t(m_CurrentGraphicsState.indexBuffer.offset);
        params->primitiveType = uint32_t(pipeline->primitiveType);
        params->drawArgumentsBindPoint = c_IrDrawArgumentsBindPoint;
        params->bufferBindingCount = bufferBindingCount;
        auto* bufferBindings = reinterpret_cast<IcbGraphicsBufferBinding*>(params + 1);
        uint32_t bindingIndex = 0;
        bufferBindings[bindingIndex++] = {
            paramsAllocation.buffer.gpuAddress + paramsAllocation.offset + offsetof(IcbIndexedIndirectParams, indexType),
            uint32_t(kIRArgumentBufferUniformsBindPoint), 0 };
        for (NSUInteger index = 0; index < m_VertexBufferBindings.size(); ++index)
        {
            const GraphicsBufferBinding& vertex = m_VertexBufferBindings[index];
            if (vertex.buffer)
                bufferBindings[bindingIndex++] = { vertex.buffer.gpuAddress + vertex.offset, uint32_t(index), 0 };
            const GraphicsBufferBinding& fragment = m_FragmentBufferBindings[index];
            if (fragment.buffer)
                bufferBindings[bindingIndex++] = { fragment.buffer.gpuAddress + fragment.offset, uint32_t(index), 1 };
        }

        m_ReferencedNativeResources.push_back(icb);
        m_ReferencedNativeBuffers.push_back(executionRange.buffer);
        m_ReferencedNativeBuffers.push_back(paramsAllocation.buffer);
        m_ReferencedNativeBuffers.push_back(icbArgumentBuffer.buffer);

        // ICB memory is reused by Metal internally; reset the command range before
        // the compute pass selectively fills commands 0..min(gpuCount,maxDrawCount).
        id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
        beginEncoding(blit, "reset indexed ICB");
        [blit resetCommandsInBuffer:icb withRange:NSMakeRange(0, icbCapacity)];
        endEncoding(blit);

        // Fill the ICB on the GPU. Thread 0 writes the execution range from the
        // GPU count buffer, and each thread below that count writes one indexed
        // draw command from indirectParams[tid].
        id<MTLComputeCommandEncoder> compute = beginTransientComputeEncoder();
        beginEncoding(compute, "fill indexed ICB");
        [compute setComputePipelineState:fillState.pipeline];
        [compute setBuffer:indirectParams->buffer offset:0 atIndex:0];
        [compute setBuffer:indirectCount->buffer offset:0 atIndex:1];
        [compute setBuffer:executionRange.buffer offset:executionRange.offset atIndex:2];
        [compute setBuffer:icbArgumentBuffer.buffer offset:icbArgumentBuffer.offset atIndex:3];
        [compute setBuffer:indexBuffer->buffer offset:0 atIndex:4];
        [compute setBuffer:paramsAllocation.buffer offset:paramsAllocation.offset atIndex:5];
        [compute setBuffer:paramsAllocation.buffer offset:paramsAllocation.offset + sizeof(IcbIndexedIndirectParams) atIndex:6];
        const NSUInteger threads = std::max<NSUInteger>(1, fillState.pipeline.threadExecutionWidth);
        const NSUInteger groups = (NSUInteger(maxDrawCount) + threads - 1) / threads;
        [compute dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        endEncoding(compute);

        renderEncoder = getOrCreateRenderEncoder();
        if (!renderEncoder)
            return;
        applyGraphicsStateToEncoder(renderEncoder, m_CurrentGraphicsState);
        // Execute only the GPU-written range. This is what turns NVRHI's count
        // buffer into Metal's MTLIndirectCommandBufferExecutionRange.
        [renderEncoder executeCommandsInBuffer:icb indirectBuffer:executionRange.buffer
                          indirectBufferOffset:executionRange.offset];
    }

    void CommandList::drawIndexedIndirectCountGeometryEmulation(
        uint32_t paramOffsetBytes,
        uint32_t countOffsetBytes,
        uint32_t maxDrawCount)
    {
        if (maxDrawCount == 0)
            return;
        if (!m_GeometryEmulationDrawStateValid)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexedIndirectCount skipped: geometry-emulation state was not fully bound");
            return;
        }

        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indexBuffer = static_cast<Buffer*>(m_CurrentGraphicsState.indexBuffer.buffer);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        auto* indirectCount = static_cast<Buffer*>(m_CurrentGraphicsState.indirectCountBuffer);
        if (!pipeline || !indexBuffer || !indexBuffer->buffer || !indirectParams || !indirectParams->buffer ||
            !indirectCount || !indirectCount->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] geometry-emulation drawIndexedIndirectCount skipped: missing pipeline/index/args/count buffer");
            return;
        }

        if (pipeline->geometryVertexSizeInBytes == 0 ||
            pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup == 0)
        {
            m_Context.warning("[metal3] geometry-emulation drawIndexedIndirectCount skipped: invalid GS reflection values");
            return;
        }

        const uint64_t requiredParamBytes =
            uint64_t(paramOffsetBytes) + uint64_t(maxDrawCount) * sizeof(DrawIndexedIndirectArguments);
        if (requiredParamBytes > indirectParams->desc.byteSize)
        {
            m_Context.warning("[metal3-trace] geometry-emulation drawIndexedIndirectCount range exceeds args buffer: required=" +
                std::to_string(requiredParamBytes) + " bufferSize=" + std::to_string(indirectParams->desc.byteSize) +
                " args='" + indirectParams->desc.debugName + "'");
            return;
        }

        const uint64_t requiredCountBytes = uint64_t(countOffsetBytes) + sizeof(uint32_t);
        if (requiredCountBytes > indirectCount->desc.byteSize)
        {
            m_Context.warning("[metal3-trace] geometry-emulation drawIndexedIndirectCount count offset exceeds count buffer: required=" +
                std::to_string(requiredCountBytes) + " bufferSize=" + std::to_string(indirectCount->desc.byteSize) +
                " count='" + indirectCount->desc.debugName + "'");
            return;
        }

        const uint32_t indexSize = m_CurrentGraphicsState.indexBuffer.format == Format::R32_UINT ? 4u : 2u;
        if (m_CurrentGraphicsState.indexBuffer.offset % indexSize != 0)
        {
            m_Context.warning("[metal3-trace] geometry-emulation counted indexed indirect draw has an unaligned index buffer offset");
            return;
        }

        if (@available(macOS 14.0, *))
        {
            // Counted geometry-emulation draws need compute-written mesh commands
            // in an ICB. macOS 14 is the first version with the mesh ICB support
            // this path relies on.
            const GeometryIndexedIndirectCountIcbFillState& fillState =
                getGeometryIndexedIndirectCountIcbFillState(m_Context);
            if (!fillState.pipeline || !fillState.icbEncoder)
            {
                m_Context.warning("[metal3] geometry-emulation counted indexed indirect draw skipped because the ICB helper pipeline is unavailable");
                return;
            }
            // The compute pass records complete ICB commands, so it needs stable
            // buffers for every table the object/mesh/fragment stages will read
            // when those commands are executed later by the render encoder.
            id<MTLBuffer> vertexBufferTable = m_GeometryEmulationVertexBuffers;
            const NSUInteger vertexBufferTableOffset = m_GeometryEmulationVertexBuffersOffset;
            const ArgumentTableAllocation objectArgumentTable = getOrCreateArgumentTable(pipeline->objectBindingPlan, m_CurrentGraphicsState.bindings);
            const ArgumentTableAllocation meshArgumentTable = getOrCreateArgumentTable(pipeline->meshBindingPlan, m_CurrentGraphicsState.bindings);
            const ArgumentTableAllocation fragmentArgumentTable = pipeline->desc.PS
                ? getOrCreateArgumentTable(pipeline->fragmentBindingPlan, m_CurrentGraphicsState.bindings)
                : ArgumentTableAllocation{};
            if (!vertexBufferTable ||
                (pipeline->objectBindingPlan.resourceCount != 0 && !objectArgumentTable.buffer) ||
                (pipeline->meshBindingPlan.resourceCount != 0 && !meshArgumentTable.buffer) ||
                (pipeline->fragmentBindingPlan.resourceCount != 0 && !fragmentArgumentTable.buffer))
            {
                m_Context.warning("[metal3] geometry-emulation counted indexed indirect draw skipped because reflected state tables are missing");
                return;
            }

            const NSUInteger drawInfoStride = sizeof(IRRuntimeDrawInfo);
            const NSUInteger drawParamsStride = sizeof(IRRuntimeDrawParams);
            const NSUInteger drawInfoSize = drawInfoStride * NSUInteger(maxDrawCount);
            const NSUInteger drawParamsSize = drawParamsStride * NSUInteger(maxDrawCount);

            MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
            icbDesc.commandTypes = MTLIndirectCommandTypeDrawMeshThreadgroups;
            icbDesc.inheritPipelineState = YES;
            // These commands need per-draw IR runtime buffers at slots 4/5, so
            // they cannot inherit buffers from the parent encoder. Encode every
            // object/mesh/fragment buffer the converted GS pipeline needs.
            icbDesc.inheritBuffers = NO;
            icbDesc.maxObjectBufferBindCount = std::max<NSUInteger>(
                kIRArgumentBufferUniformsBindPoint + 1,
                std::max<NSUInteger>(kIRArgumentBufferDrawArgumentsBindPoint + 1,
                    kIRArgumentBufferBindPoint + 1));
            icbDesc.maxMeshBufferBindCount = std::max<NSUInteger>(
                kIRArgumentBufferUniformsBindPoint + 1,
                std::max<NSUInteger>(kIRArgumentBufferDrawArgumentsBindPoint + 1,
                    kIRArgumentBufferBindPoint + 1));
            icbDesc.maxFragmentBufferBindCount = kIRArgumentBufferBindPoint + 1;

            // The ICB is whole-object pooled; the supporting buffers are page
            // slices and therefore must keep their returned offsets.
            NSUInteger icbCapacity = 0;
            id<MTLIndirectCommandBuffer> icb =
                m_TransientIndirectResources.acquireIndirectCommandBuffer(icbDesc, NSUInteger(maxDrawCount), nullptr, &icbCapacity);
            const TransientBufferAllocation executionRange =
                m_TransientIndirectResources.allocatePrivate(sizeof(MTLIndirectCommandBufferExecutionRange));
            const TransientBufferAllocation drawInfo = m_TransientIndirectResources.allocatePrivate(drawInfoSize);
            const TransientBufferAllocation drawParams = m_TransientIndirectResources.allocatePrivate(drawParamsSize);
            const TransientBufferAllocation paramsAllocation =
                m_TransientIndirectResources.allocateShared(sizeof(GeometryIndexedIndirectCountParams));
            const TransientBufferAllocation icbArgumentBuffer =
                m_TransientIndirectResources.allocateShared(fillState.icbEncoder.encodedLength);
            if (!icb || !executionRange.buffer || !drawInfo.buffer || !drawParams.buffer ||
                !paramsAllocation.buffer || !paramsAllocation.cpuAddress ||
                !icbArgumentBuffer.buffer || !icbArgumentBuffer.cpuAddress)
            {
                m_Context.error("[metal3] failed to allocate geometry-emulation counted indexed indirect ICB resources");
                return;
            }

            // Metal exposes the ICB to a compute shader through an argument
            // buffer. The helper writes render_command entries into this ICB.
            [fillState.icbEncoder setArgumentBuffer:icbArgumentBuffer.buffer offset:icbArgumentBuffer.offset];
            [fillState.icbEncoder setIndirectCommandBuffer:icb atIndex:0];

            // CPU-written constants tell the helper where to read the GPU count
            // and args, how to interpret the current index buffer, and whether
            // optional reflected argument tables must be encoded into each command.
            auto* params = reinterpret_cast<GeometryIndexedIndirectCountParams*>(paramsAllocation.cpuAddress);
            params->maxDrawCount = maxDrawCount;
            params->paramOffsetBytes = paramOffsetBytes;
            params->countOffsetBytes = countOffsetBytes;
            params->primitiveType = uint32_t(convertRuntimePrimitiveType(pipeline->desc.primType));
            params->gsVertexSizeInBytes = pipeline->geometryVertexSizeInBytes;
            params->gsMaxInputPrimitivesPerMeshThreadgroup =
                pipeline->geometryMaxInputPrimitivesPerMeshThreadgroup;
            params->indexType = uint32_t(convertIndexFormat(m_CurrentGraphicsState.indexBuffer.format));
            params->indexBufferOffsetInElements =
                uint32_t(m_CurrentGraphicsState.indexBuffer.offset / indexSize);
            params->indexBufferAddress = indexBuffer->getGpuVirtualAddress();
            params->hasObjectArgumentTable = objectArgumentTable.buffer ? 1u : 0u;
            params->hasMeshArgumentTable = meshArgumentTable.buffer ? 1u : 0u;
            params->hasFragmentArgumentTable = fragmentArgumentTable.buffer ? 1u : 0u;

            m_ReferencedNativeResources.push_back(icb);
            m_ReferencedNativeBuffers.push_back(executionRange.buffer);
            m_ReferencedNativeBuffers.push_back(drawInfo.buffer);
            m_ReferencedNativeBuffers.push_back(drawParams.buffer);
            m_ReferencedNativeBuffers.push_back(paramsAllocation.buffer);
            m_ReferencedNativeBuffers.push_back(icbArgumentBuffer.buffer);
            m_ReferencedNativeBuffers.push_back(vertexBufferTable);
            if (objectArgumentTable.buffer)
                m_ReferencedNativeBuffers.push_back(objectArgumentTable.buffer);
            if (meshArgumentTable.buffer)
                m_ReferencedNativeBuffers.push_back(meshArgumentTable.buffer);
            if (fragmentArgumentTable.buffer)
                m_ReferencedNativeBuffers.push_back(fragmentArgumentTable.buffer);

            endEncoding();

            // Reset the full possible command range; the compute pass only writes
            // commands below min(gpuCount, maxDrawCount), and executeCommands uses
            // the GPU-written range to skip the rest.
            id<MTLBlitCommandEncoder> blit = beginBlitEncoder();
            beginEncoding(blit, "reset geometry ICB");
            [blit resetCommandsInBuffer:icb withRange:NSMakeRange(0, icbCapacity)];
            endEncoding(blit);

            // Fill the per-draw IRRuntime records and ICB commands on GPU. This is
            // the counted part of the path: the shader reads indirectCount and no
            // CPU-side loop or readback is needed.
            id<MTLComputeCommandEncoder> compute = beginTransientComputeEncoder();
            beginEncoding(compute, "fill geometry ICB");
            if (!compute)
            {
                m_Context.error("[metal3] failed to create compute encoder for geometry counted indexed indirect ICB fill");
                return;
            }
            [compute setComputePipelineState:fillState.pipeline];
            [compute setBuffer:indirectParams->buffer offset:0 atIndex:0];
            [compute setBuffer:indirectCount->buffer offset:0 atIndex:1];
            [compute setBuffer:executionRange.buffer offset:executionRange.offset atIndex:2];
            [compute setBuffer:icbArgumentBuffer.buffer offset:icbArgumentBuffer.offset atIndex:3];
            [compute setBuffer:drawInfo.buffer offset:drawInfo.offset atIndex:4];
            [compute setBuffer:drawParams.buffer offset:drawParams.offset atIndex:5];
            [compute setBuffer:paramsAllocation.buffer offset:paramsAllocation.offset atIndex:6];
            [compute setBuffer:vertexBufferTable offset:vertexBufferTableOffset atIndex:7];
            [compute setBuffer:(objectArgumentTable.buffer ? objectArgumentTable.buffer : paramsAllocation.buffer)
                        offset:(objectArgumentTable.buffer ? objectArgumentTable.offset : paramsAllocation.offset) atIndex:8];
            [compute setBuffer:(meshArgumentTable.buffer ? meshArgumentTable.buffer : paramsAllocation.buffer)
                        offset:(meshArgumentTable.buffer ? meshArgumentTable.offset : paramsAllocation.offset) atIndex:9];
            [compute setBuffer:(fragmentArgumentTable.buffer ? fragmentArgumentTable.buffer : paramsAllocation.buffer)
                        offset:(fragmentArgumentTable.buffer ? fragmentArgumentTable.offset : paramsAllocation.offset) atIndex:10];

            const NSUInteger threads = std::max<NSUInteger>(1, fillState.pipeline.threadExecutionWidth);
            const NSUInteger groups = (NSUInteger(maxDrawCount) + threads - 1) / threads;
            [compute dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            endEncoding(compute);

            id<MTLRenderCommandEncoder> renderEncoder = getOrCreateRenderEncoder();
            if (!renderEncoder)
                return;

            applyGraphicsStateToEncoder(renderEncoder, m_CurrentGraphicsState);
            if (!m_GeometryEmulationDrawStateValid)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] drawIndexedIndirectCount skipped: geometry-emulation state restore failed");
                return;
            }

            // Execute only commands 0..gpuCount-1, where gpuCount was clamped by
            // the compute helper and written into executionRange on the GPU.
            [renderEncoder executeCommandsInBuffer:icb
                                    indirectBuffer:executionRange.buffer
                              indirectBufferOffset:executionRange.offset];
        }
        else if (traceMetalRuntime())
        {
            m_Context.warning("[metal3-trace] geometry-emulation drawIndexedIndirectCount requires macOS 14 mesh ICB support");
        }
    }

    void CommandList::setComputeState(const ComputeState& state)
    {
        m_CurrentComputeState = state;
        m_CurrentComputeStateValid = true;
        if (state.pipeline)
            m_ReferencedStateResources.emplace_back(state.pipeline);
        referenceBuffer(state.indirectParams);
        if (m_EnableAutomaticBarriers)
        {
            for (IBindingSet* bindingSet : state.bindings)
                setResourceStatesForBindingSet(bindingSet);
            if (state.indirectParams)
                setBufferState(state.indirectParams, ResourceStates::IndirectArgument);
            commitBarriers();
        }
        id<MTLComputeCommandEncoder> encoder = getOrCreateComputeEncoder();
        auto* pipeline = static_cast<ComputePipeline*>(state.pipeline);
        if (!encoder || !pipeline)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] setComputeState skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no"));
            return;
        }

        [encoder setComputePipelineState:pipeline->pipeline];
        m_BindingStatesDirty = false;
        m_ArgumentTablesDirty = false;
        applyComputeBindings(encoder, state);

        const ArgumentTableAllocation allocation = getOrCreateArgumentTable(pipeline->computeBindingPlan, state.bindings);
        if (allocation.buffer)
        {
            m_ReferencedNativeBuffers.push_back(allocation.buffer);
            [encoder setBuffer:allocation.buffer offset:allocation.offset atIndex:kIRArgumentBufferBindPoint];
        }
        else if (traceMetalRuntime() && pipeline->computeBindingPlan.valid && pipeline->computeBindingPlan.resourceCount != 0)
        {
            static int missingComputeArgBufferLogs = 0;
            if (missingComputeArgBufferLogs++ < 16)
                m_Context.warning("[metal3-trace] setComputeState created no cached argument table");
        }
    }

    void CommandList::dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
    {
        if (!m_CurrentComputeStateValid)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] dispatch skipped: compute state invalid");
            return;
        }
        if (!m_ComputeEncoder || m_BindingStatesDirty)
            setComputeState(m_CurrentComputeState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();
        id<MTLComputeCommandEncoder> encoder = m_ComputeEncoder;
        auto* pipeline = static_cast<ComputePipeline*>(m_CurrentComputeState.pipeline);
        if (!encoder || !pipeline)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] dispatch skipped: encoder/pipeline missing");
            return;
        }
        MTLSize threadsPerGroup = pipeline->threadsPerGroup;
        [encoder dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, groupsZ) threadsPerThreadgroup:threadsPerGroup];
    }

    void CommandList::dispatchIndirect(uint32_t offsetBytes)
    {
        auto* arguments = static_cast<Buffer*>(m_CurrentComputeState.indirectParams);
        auto* pipeline = static_cast<ComputePipeline*>(m_CurrentComputeState.pipeline);
        if (m_RecordingState != RecordingState::Open || !m_CurrentComputeStateValid || !pipeline ||
            !arguments || !arguments->buffer || !arguments->desc.isDrawIndirectArgs ||
            offsetBytes % 4 != 0 || offsetBytes > arguments->desc.byteSize ||
            arguments->desc.byteSize - offsetBytes < sizeof(DispatchIndirectArguments))
        {
            m_RecordingFailed = true;
            m_Context.error("[metal3] Indirect dispatch requires a compute pipeline and an aligned argument-buffer range.");
            return;
        }
        if (!m_ComputeEncoder || m_BindingStatesDirty)
            setComputeState(m_CurrentComputeState);
        else if (m_ArgumentTablesDirty)
            rebindArgumentTables();
        if (!m_ComputeEncoder)
        {
            m_RecordingFailed = true;
            m_Context.error("[metal3] Cannot create the indirect dispatch compute encoder.");
            return;
        }
        m_ReferencedStateResources.emplace_back(arguments);
        [m_ComputeEncoder dispatchThreadgroupsWithIndirectBuffer:arguments->buffer
            indirectBufferOffset:offsetBytes threadsPerThreadgroup:pipeline->threadsPerGroup];
    }

    void CommandList::applyGraphicsStateToEncoder(id<MTLRenderCommandEncoder> encoder, const GraphicsState& state)
    {
        auto* pipeline = static_cast<GraphicsPipeline*>(state.pipeline);
        if (!encoder || !pipeline)
            return;

        m_GeometryEmulationDrawStateValid = false;
        m_GeometryEmulationVertexBuffers = nil;
        m_GeometryEmulationVertexBuffersOffset = 0;
        m_VertexBufferBindings.fill({});
        m_FragmentBufferBindings.fill({});

        [encoder setRenderPipelineState:pipeline->pipeline];
        [encoder setDepthStencilState:pipeline->depthStencilState];
        [encoder setCullMode:pipeline->cullMode];
        [encoder setFrontFacingWinding:pipeline->frontWinding];
        const DepthStencilState& depthStencil = pipeline->desc.renderState.depthStencilState;
        [encoder setStencilReferenceValue:depthStencil.dynamicStencilRef ? state.dynamicStencilRefValue : depthStencil.stencilRefValue];
        [encoder setBlendColorRed:state.blendConstantColor.r green:state.blendConstantColor.g
            blue:state.blendConstantColor.b alpha:state.blendConstantColor.a];
        m_BindingStatesDirty = false;
        m_ArgumentTablesDirty = false;

        for (const Viewport& vp : state.viewport.viewports)
            [encoder setViewport:MTLViewport{ vp.minX, vp.minY, vp.width(), vp.height(), vp.minZ, vp.maxZ }];
        for (const Rect& rect : state.viewport.scissorRects)
            [encoder setScissorRect:MTLScissorRect{ NSUInteger(rect.minX), NSUInteger(rect.minY), NSUInteger(rect.width()), NSUInteger(rect.height()) }];

        for (const VertexBufferBinding& vb : state.vertexBuffers)
        {
            auto* buffer = static_cast<Buffer*>(vb.buffer);
            if (traceMetalRuntime() && !buffer)
                m_Context.warning("[metal3-trace] null vertex buffer at slot " + std::to_string(vb.slot));
            m_ReferencedNativeBuffers.push_back(buffer ? buffer->buffer : nil);
            bindGraphicsBuffer(encoder, buffer ? buffer->buffer : nil, NSUInteger(vb.offset),
                c_MscVertexBufferBindPoint + vb.slot, MTLRenderStageVertex);
        }

        // Keep binding sets alive and report null resources once from the common
        // graphics path. Shader resources, including volatile CBVs, are encoded
        // through the reflected MSC argument tables below.
        applyBindingSets(state.bindings);

        if (pipeline->usesGeometryEmulation)
        {
            if (@available(macOS 13.0, *))
            {
                m_GeometryEmulationDrawStateValid = bindGeometryEmulationVertexBuffers(encoder, *pipeline, state);
                bindGraphicsArgumentTable(encoder, state.bindings, pipeline->objectBindingPlan, MTLRenderStageObject);
                bindGraphicsArgumentTable(encoder, state.bindings, pipeline->meshBindingPlan, MTLRenderStageMesh);
                if (pipeline->desc.PS)
                {
                    bindGraphicsArgumentTable(encoder, state.bindings, pipeline->fragmentBindingPlan, MTLRenderStageFragment);
                }
            }
            else if (traceMetalRuntime())
            {
                m_Context.warning("[metal3-trace] geometry-emulation graphics state requires object/mesh stage binding support");
            }
            return;
        }

        bindGraphicsArgumentTable(encoder, state.bindings, pipeline->vertexBindingPlan, MTLRenderStageVertex);
        if (pipeline->desc.PS)
        {
            bindGraphicsArgumentTable(encoder, state.bindings, pipeline->fragmentBindingPlan, MTLRenderStageFragment);
        }
    }
    /*
    // Builds the Metal Shader Converter descriptor table for one shader stage.
    // The pipeline's reflected binding plan tells us which argument-table index
    // each HLSL resource occupies, and layoutIndex selects the matching NVRHI
    // binding set from the current graphics/compute state.
    */
    ArgumentTableAllocation CommandList::createArgumentTable(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets)
    {
        ArgumentTableAllocation allocation;
        if (!plan.valid || !plan.argumentBufferSize)
            return allocation;
        UploadAllocation upload = m_ArgumentTableManager.suballocate(size_t(plan.argumentBufferSize), 256);
        if (!upload.buffer || !upload.cpuAddress)
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] Failed to allocate Metal shader argument table.");
            return allocation;
        }
        allocation.buffer = upload.buffer;
        allocation.offset = upload.offset;
        allocation.cpuAddress = upload.cpuAddress;
        ++m_ArgumentTableAllocationCount;
        std::memset(upload.cpuAddress, 0, size_t(plan.argumentBufferSize));
        for (const auto& table : plan.descriptorTables)
        {
            uint64_t address = 0;
            if (table.descriptorCount)
                address = upload.buffer.gpuAddress + upload.offset + table.descriptorOffset;
            else
            {
                if (table.layoutIndex >= bindingSets.size() || !bindingSets[table.layoutIndex] ||
                    bindingSets[table.layoutIndex]->getDesc())
                {
                    m_RecordingFailed = true;
                    m_Context.error("[nvrhi] Missing Metal unbounded descriptor table for '" + plan.debugName +
                        "', slot " + std::to_string(table.slot) + ", space " + std::to_string(table.space) + ".");
                    continue;
                }
                auto* descriptorTable = static_cast<DescriptorTable*>(bindingSets[table.layoutIndex]);
                auto snapshot = descriptorTable->getSnapshot(m_Context);
                if (!snapshot || !snapshot->buffer)
                {
                    m_RecordingFailed = true;
                    m_Context.error("[nvrhi] Cannot snapshot Metal unbounded descriptor table for '" + plan.debugName + "'.");
                    continue;
                }
                address = snapshot->buffer.gpuAddress;
                referenceDescriptorSnapshot(snapshot);
            }
            std::memcpy(static_cast<uint8_t*>(upload.cpuAddress) + table.byteOffset, &address, sizeof(address));
        }
        for (const MetalBindingPlanEntry& entry : plan.entries)
        {
            const MetalBindingResource* resource = findArgumentTableResource(bindingSets, entry);
            if (!resource)
            {
                m_RecordingFailed = true;
                m_Context.error("[nvrhi] Missing Metal shader binding for '" + plan.debugName +
                    "', slot " + std::to_string(entry.slot) + ", space " + std::to_string(entry.space) + ".");
                continue;
            }
            if (!resource->resource)
                continue;
            auto* encoded = reinterpret_cast<IRDescriptorTableEntry*>(static_cast<uint8_t*>(upload.cpuAddress) + entry.byteOffset);
            if (!encodeArgumentTableEntry(encoded, *resource))
            {
                m_RecordingFailed = true;
                m_Context.error("[nvrhi] Failed to encode Metal shader resource for '" + plan.debugName +
                    "', slot " + std::to_string(entry.slot) + ", space " + std::to_string(entry.space) + ".");
            }
        }
        return allocation;
    }

    ArgumentTableAllocation CommandList::getOrCreateArgumentTable(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets)
    {
        ArgumentTableAllocation allocation;
        if (!plan.valid || !plan.argumentBufferSize)
        {
            if (!plan.valid && traceMetalRuntime())
            {
                static int invalidPlanLogCount = 0;
                if (invalidPlanLogCount++ < 32)
                    m_Context.warning("[metal3-trace] missing reflected binding plan for " +
                        std::string(utils::ShaderStageToString(plan.stage)) + "; regular shader resources are not directly bound");
            }
            return allocation;
        }

        uint64_t volatileBufferVersion = 0;
        for (const MetalBindingPlanEntry& entry : plan.entries)
        {
            const MetalBindingResource* resource = findArgumentTableResource(bindingSets, entry);
            if (!resource || resource->type != ResourceType::VolatileConstantBuffer)
                continue;
            auto* buffer = static_cast<Buffer*>(resource->resource.Get());
            const auto current = m_VolatileBufferAllocations.find(buffer);
            if (current == m_VolatileBufferAllocations.end() || !current->second.allocation.buffer)
                return createArgumentTable(plan, bindingSets);
            volatileBufferVersion = std::max(volatileBufferVersion, current->second.version);
        }

        MetalArgumentTableCacheKey key = makeArgumentTableCacheKey(plan, bindingSets);
        auto cached = m_ArgumentTableCache.find(key);
        if (cached != m_ArgumentTableCache.end())
        {
            if (!cached->second.allocation.buffer || cached->second.volatileBufferVersion != volatileBufferVersion)
            {
                allocation = createArgumentTable(plan, bindingSets);
                if (!allocation.buffer)
                    return allocation;
                cached->second.allocation = allocation;
                cached->second.volatileBufferVersion = volatileBufferVersion;
            }
            return cached->second.allocation;
        }

        allocation = createArgumentTable(plan, bindingSets);
        if (!allocation.buffer)
            return allocation;

        m_ArgumentTableCache.emplace(std::move(key), MetalArgumentTableCacheEntry{ allocation, volatileBufferVersion });
        return allocation;
    }

    static bool isSamplerTable(const BindlessLayoutDesc& desc)
    {
        return desc.layoutType == BindlessLayoutDesc::LayoutType::MutableSampler ||
            (desc.layoutType == BindlessLayoutDesc::LayoutType::Immutable &&
                !desc.registerSpaces.empty() && desc.registerSpaces.front().type == ResourceType::Sampler);
    }

    void CommandList::bindDescriptorTables(id<MTLRenderCommandEncoder> encoder, const BindingSetVector& bindingSets,
        const MetalStageBindingPlan& plan, MTLRenderStages stages)
    {
        bool bound[2]{};
        for (size_t bindingIndex = 0; bindingIndex < bindingSets.size(); ++bindingIndex)
        {
            IBindingSet* bindingSet = bindingSets[bindingIndex];
            if (!bindingSet || bindingSet->getDesc())
                continue;
            auto* table = static_cast<DescriptorTable*>(bindingSet);
            const BindlessLayoutDesc& desc = *table->layout->getBindlessDesc();
            if ((desc.visibility & plan.stage) == ShaderType::None)
                continue;
            const bool sampler = isSamplerTable(desc);
            const bool direct = sampler ? plan.directlyIndexedSamplerHeap : plan.directlyIndexedResourceHeap;
            if (!direct && std::none_of(plan.descriptorTables.begin(), plan.descriptorTables.end(),
                [bindingIndex](const MscDescriptorTable& reflectedTable) {
                    return reflectedTable.descriptorCount == 0 && reflectedTable.layoutIndex == bindingIndex;
                }))
                continue;
            if (direct)
            {
                if (bound[sampler])
                {
                    m_RecordingFailed = true;
                    m_Context.error("[nvrhi] Metal supports one directly indexed table per heap per shader stage.");
                    return;
                }
                bound[sampler] = true;
            }
            auto snapshot = table->getSnapshot(m_Context);
            if (!snapshot)
                continue;
            referenceDescriptorSnapshot(snapshot);
            const NSUInteger index = sampler ? kIRSamplerHeapBindPoint : kIRDescriptorHeapBindPoint;
            if (direct)
            {
                bindGraphicsBuffer(encoder, snapshot->buffer, 0, index, stages);
            }
        }
        if ((plan.directlyIndexedResourceHeap && !bound[0]) || (plan.directlyIndexedSamplerHeap && !bound[1]))
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] Missing directly indexed Metal descriptor heap for '" + plan.debugName + "'.");
        }
    }

    void CommandList::bindDescriptorTables(id<MTLComputeCommandEncoder> encoder, const BindingSetVector& bindingSets,
        const MetalStageBindingPlan& plan)
    {
        bool bound[2]{};
        for (size_t bindingIndex = 0; bindingIndex < bindingSets.size(); ++bindingIndex)
        {
            IBindingSet* bindingSet = bindingSets[bindingIndex];
            if (!bindingSet || bindingSet->getDesc())
                continue;
            auto* table = static_cast<DescriptorTable*>(bindingSet);
            const BindlessLayoutDesc& desc = *table->layout->getBindlessDesc();
            if ((desc.visibility & ShaderType::Compute) == ShaderType::None)
                continue;
            const bool sampler = isSamplerTable(desc);
            const bool direct = sampler ? plan.directlyIndexedSamplerHeap : plan.directlyIndexedResourceHeap;
            if (!direct && std::none_of(plan.descriptorTables.begin(), plan.descriptorTables.end(),
                [bindingIndex](const MscDescriptorTable& reflectedTable) {
                    return reflectedTable.descriptorCount == 0 && reflectedTable.layoutIndex == bindingIndex;
                }))
                continue;
            if (direct)
            {
                if (bound[sampler])
                {
                    m_RecordingFailed = true;
                    m_Context.error("[nvrhi] Metal supports one directly indexed table per heap per shader stage.");
                    return;
                }
                bound[sampler] = true;
            }
            auto snapshot = table->getSnapshot(m_Context);
            if (!snapshot)
                continue;
            referenceDescriptorSnapshot(snapshot);
            if (direct)
                [encoder setBuffer:snapshot->buffer offset:0 atIndex:sampler ? kIRSamplerHeapBindPoint : kIRDescriptorHeapBindPoint];
        }
        if ((plan.directlyIndexedResourceHeap && !bound[0]) || (plan.directlyIndexedSamplerHeap && !bound[1]))
        {
            m_RecordingFailed = true;
            m_Context.error("[nvrhi] Missing directly indexed Metal descriptor heap for '" + plan.debugName + "'.");
        }
    }

    void CommandList::bindGraphicsArgumentTable(
        id<MTLRenderCommandEncoder> encoder,
        const BindingSetVector& bindingSets,
        const MetalStageBindingPlan& plan,
        MTLRenderStages stages)
    {
        bindDescriptorTables(encoder, bindingSets, plan, stages);
        const ArgumentTableAllocation allocation = getOrCreateArgumentTable(plan, bindingSets);
        if (!allocation.buffer)
            return;

        m_ReferencedNativeBuffers.push_back(allocation.buffer);

        bindGraphicsBuffer(encoder, allocation.buffer, allocation.offset, kIRArgumentBufferBindPoint, stages);

    }

    bool CommandList::bindGeometryEmulationVertexBuffers(
        id<MTLRenderCommandEncoder> encoder,
        const GraphicsPipeline& pipeline,
        const GraphicsState& state)
    {
        auto* inputLayout = static_cast<InputLayout*>(pipeline.desc.inputLayout.Get());
        if (!inputLayout)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] geometry-emulation pipeline has no input layout for IRRuntimeVertexBuffers");
            return false;
        }

        std::array<uint32_t, c_IrRuntimeVertexBufferCount> strides{};
        std::array<bool, c_IrRuntimeVertexBufferCount> requiredSlots{};
        for (const VertexAttributeDesc& attr : inputLayout->attributes)
        {
            if (attr.bufferIndex >= c_IrRuntimeVertexBufferCount)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] geometry-emulation vertex buffer slot exceeds IRRuntimeVertexBuffers capacity: slot=" +
                        std::to_string(attr.bufferIndex));
                return false;
            }

            requiredSlots[attr.bufferIndex] = true;
            if (strides[attr.bufferIndex] != 0 && strides[attr.bufferIndex] != attr.elementStride && traceMetalRuntime())
            {
                m_Context.warning("[metal3-trace] geometry-emulation input layout has mismatched strides for slot " +
                    std::to_string(attr.bufferIndex));
            }
            strides[attr.bufferIndex] = attr.elementStride;
        }

        IRRuntimeVertexBuffers vertexBuffers{};
        std::array<bool, c_IrRuntimeVertexBufferCount> boundSlots{};
        for (const VertexBufferBinding& vb : state.vertexBuffers)
        {
            if (vb.slot >= c_IrRuntimeVertexBufferCount)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] geometry-emulation vertex buffer slot exceeds IRRuntimeVertexBuffers capacity: slot=" +
                        std::to_string(vb.slot));
                return false;
            }

            if (!requiredSlots[vb.slot])
                continue;

            auto* buffer = static_cast<Buffer*>(vb.buffer);
            if (!buffer || !buffer->buffer)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] geometry-emulation vertex buffer is null at required slot " +
                        std::to_string(vb.slot));
                return false;
            }
            if (strides[vb.slot] == 0)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] geometry-emulation input layout has zero stride for slot " +
                        std::to_string(vb.slot));
                return false;
            }
            if (vb.offset > buffer->desc.byteSize)
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] geometry-emulation vertex buffer offset exceeds buffer size at slot " +
                        std::to_string(vb.slot));
                return false;
            }

            const uint64_t remainingBytes = buffer->desc.byteSize - vb.offset;
            vertexBuffers[vb.slot].addr = buffer->getGpuVirtualAddress() + vb.offset;
            vertexBuffers[vb.slot].length = uint32_t(std::min<uint64_t>(remainingBytes, std::numeric_limits<uint32_t>::max()));
            vertexBuffers[vb.slot].stride = strides[vb.slot];
            boundSlots[vb.slot] = true;

        }

        for (uint32_t slot = 0; slot < c_IrRuntimeVertexBufferCount; ++slot)
        {
            if (requiredSlots[slot] && !boundSlots[slot])
            {
                if (traceMetalRuntime())
                    m_Context.warning("[metal3-trace] geometry-emulation missing required vertex buffer slot " +
                        std::to_string(slot));
                return false;
            }
        }

        UploadAllocation allocation = m_UploadManager.suballocate(sizeof(IRRuntimeVertexBuffers), 256);
        if (!allocation.buffer || !allocation.cpuAddress)
            return false;

        std::memcpy(allocation.cpuAddress, vertexBuffers, sizeof(IRRuntimeVertexBuffers));
        m_ReferencedNativeBuffers.push_back(allocation.buffer);
        m_GeometryEmulationVertexBuffers = allocation.buffer;
        m_GeometryEmulationVertexBuffersOffset = allocation.offset;
        [encoder setObjectBuffer:allocation.buffer offset:allocation.offset atIndex:0];
        return true;
    }

    // this path basically exclusively exists for binding volatile contant buffers
    // regular resources go through the argument table path. For graphics,
    // volatile CB binding itself is driven by reflected per-stage plans so GS
    // emulation does not bind unused object/mesh/fragment stage buffers.
    void CommandList::applyBindingSets(const BindingSetVector& bindings)
    {
        for (IBindingSet* bindingSet : bindings)
        {
            referenceBindingSet(bindingSet);
            if (!bindingSet || !bindingSet->getDesc())
                continue;
            for (const MetalBindingResource& entry : static_cast<BindingSet*>(bindingSet)->entries)
            {
                if (!entry.resource)
                    continue;
                const void* written = nullptr;
                switch (entry.type)
                {
                case ResourceType::Texture_UAV:
                    written = &static_cast<Texture*>(entry.resource.Get())->stateExtension;
                    break;
                case ResourceType::TypedBuffer_UAV:
                case ResourceType::StructuredBuffer_UAV:
                case ResourceType::RawBuffer_UAV:
                    written = &static_cast<Buffer*>(entry.resource.Get())->stateExtension;
                    break;
                default:
                    break;
                }
                if (written && std::find(m_RenderEncoderWrites.begin(), m_RenderEncoderWrites.end(), written) == m_RenderEncoderWrites.end())
                    m_RenderEncoderWrites.push_back(written);
            }
        }
    }

    void CommandList::applyComputeBindings(id<MTLComputeCommandEncoder> encoder, const ComputeState& state)
    {
        for (IBindingSet* bindingSet : state.bindings)
            referenceBindingSet(bindingSet);
        if (state.pipeline)
            bindDescriptorTables(encoder, state.bindings, static_cast<ComputePipeline*>(state.pipeline)->computeBindingPlan);
    }

    void CommandList::referenceBindingSet(IBindingSet* bindingSet)
    {
        if (bindingSet && std::find(m_ReferencedBindingSets.begin(), m_ReferencedBindingSets.end(), bindingSet) == m_ReferencedBindingSets.end())
            m_ReferencedBindingSets.push_back(bindingSet);
    }

    void CommandList::referenceBuffer(IBuffer* buffer)
    {
        if (buffer && buffer->getDesc().cpuAccess != CpuAccessMode::None &&
            std::find(m_ReferencedMappableBuffers.begin(), m_ReferencedMappableBuffers.end(), buffer) ==
                m_ReferencedMappableBuffers.end())
            m_ReferencedMappableBuffers.emplace_back(buffer);
    }

    void CommandList::referenceDescriptorSnapshot(const std::shared_ptr<DescriptorTableSnapshot>& snapshot)
    {
        if (std::find(m_ReferencedDescriptorSnapshots.begin(), m_ReferencedDescriptorSnapshots.end(), snapshot) !=
            m_ReferencedDescriptorSnapshots.end())
            return;
        m_ReferencedDescriptorSnapshots.push_back(snapshot);
    }

    void CommandList::copyTexture(IStagingTexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice)
    {
        (void)dest;
        (void)destSlice;
        (void)src;
        (void)srcSlice;
        unsupported(__func__);
    }

    void CommandList::copyTexture(ITexture* dest, const TextureSlice& destSlice, IStagingTexture* src, const TextureSlice& srcSlice)
    {
        (void)dest;
        (void)destSlice;
        (void)src;
        (void)srcSlice;
        unsupported(__func__);
    }

    void CommandList::rebindArgumentTables()
    {
        m_ArgumentTablesDirty = false;
        if (m_RenderEncoder && m_CurrentGraphicsStateValid)
        {
            auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
            if (!pipeline)
                return;
            const BindingSetVector& bindings = m_CurrentGraphicsState.bindings;
            if (pipeline->usesGeometryEmulation)
            {
                bindGraphicsArgumentTable(m_RenderEncoder, bindings, pipeline->objectBindingPlan, MTLRenderStageObject);
                bindGraphicsArgumentTable(m_RenderEncoder, bindings, pipeline->meshBindingPlan, MTLRenderStageMesh);
            }
            else
                bindGraphicsArgumentTable(m_RenderEncoder, bindings, pipeline->vertexBindingPlan, MTLRenderStageVertex);
            if (pipeline->desc.PS)
                bindGraphicsArgumentTable(m_RenderEncoder, bindings, pipeline->fragmentBindingPlan, MTLRenderStageFragment);
        }
        else if (m_RenderEncoder && m_CurrentMeshletStateValid)
        {
            auto* pipeline = static_cast<MeshletPipeline*>(m_CurrentMeshletState.pipeline);
            if (!pipeline)
                return;
            const BindingSetVector& bindings = m_CurrentMeshletState.bindings;
            if (pipeline->hasObjectStage)
                bindGraphicsArgumentTable(m_RenderEncoder, bindings, pipeline->objectBindingPlan, MTLRenderStageObject);
            bindGraphicsArgumentTable(m_RenderEncoder, bindings, pipeline->meshBindingPlan, MTLRenderStageMesh);
            if (pipeline->desc.PS)
                bindGraphicsArgumentTable(m_RenderEncoder, bindings, pipeline->fragmentBindingPlan, MTLRenderStageFragment);
        }
        else if (m_ComputeEncoder && m_CurrentComputeStateValid)
        {
            auto* pipeline = static_cast<ComputePipeline*>(m_CurrentComputeState.pipeline);
            if (!pipeline)
                return;
            const ArgumentTableAllocation allocation = getOrCreateArgumentTable(pipeline->computeBindingPlan, m_CurrentComputeState.bindings);
            if (!allocation.buffer)
                return;
            m_ReferencedNativeBuffers.push_back(allocation.buffer);
            [m_ComputeEncoder setBuffer:allocation.buffer offset:allocation.offset atIndex:kIRArgumentBufferBindPoint];
        }
    }

    void CommandList::setPushConstants(const void* data, size_t byteSize)
    {
        m_PushConstantSize = std::min(byteSize, m_PushConstants.size());
        if (data && m_PushConstantSize)
            std::memcpy(m_PushConstants.data(), data, m_PushConstantSize);
        m_BindingStatesDirty = true;
    }

    void CommandList::setRayTracingState(const rt::State& state) { (void)state; unsupported(__func__); }
    void CommandList::dispatchRays(const rt::DispatchRaysArguments& args) { (void)args; unsupported(__func__); }
    void CommandList::buildOpacityMicromap(rt::IOpacityMicromap* omm, const rt::OpacityMicromapDesc& desc) { (void)omm; (void)desc; unsupported(__func__); }
    void CommandList::executeMultiIndirectClusterOperation(const rt::cluster::OperationDesc& desc) { (void)desc; unsupported(__func__); }
    void CommandList::convertCoopVecMatrices(coopvec::ConvertMatrixLayoutDesc const* convertDescs, size_t numDescs) { (void)convertDescs; (void)numDescs; unsupported(__func__); }
    void CommandList::clearSamplerFeedbackTexture(ISamplerFeedbackTexture* texture) { (void)texture; unsupported(__func__); }
    void CommandList::decodeSamplerFeedbackTexture(IBuffer* buffer, ISamplerFeedbackTexture* texture, Format format) { (void)buffer; (void)texture; (void)format; unsupported(__func__); }
    void CommandList::setSamplerFeedbackTextureState(ISamplerFeedbackTexture* texture, ResourceStates stateBits) { (void)texture; (void)stateBits; unsupported(__func__); }
    void CommandList::bindGraphicsBuffer(id<MTLRenderCommandEncoder> encoder, id<MTLBuffer> buffer,
        NSUInteger offset, NSUInteger index, MTLRenderStages stages)
    {
        if (index >= m_VertexBufferBindings.size())
        {
            m_RecordingFailed = true;
            m_Context.error("[metal3] Graphics buffer binding exceeds the Metal buffer argument limit.");
            return;
        }
        if (stages & MTLRenderStageVertex)
        {
            m_VertexBufferBindings[index] = { buffer, offset };
            [encoder setVertexBuffer:buffer offset:offset atIndex:index];
        }
        if (stages & MTLRenderStageFragment)
        {
            m_FragmentBufferBindings[index] = { buffer, offset };
            [encoder setFragmentBuffer:buffer offset:offset atIndex:index];
        }
        if (@available(macOS 13.0, *))
        {
            if (stages & MTLRenderStageObject)
                [encoder setObjectBuffer:buffer offset:offset atIndex:index];
            if (stages & MTLRenderStageMesh)
                [encoder setMeshBuffer:buffer offset:offset atIndex:index];
        }
    }

    void CommandList::annotateEncoder(id<MTLCommandEncoder> encoder, const char* operation)
    {
        if (encoder && (traceMetalRuntime() || encoderExecutionStatusEnabled()))
            encoder.label = [NSString stringWithFormat:@"%@ / %s",
                m_DebugGroups.empty() ? @"unscoped" : m_DebugGroups.back(), operation];
    }

    void CommandList::beginMarker(const char* name)
    {
        if (!name)
            return;
        NSString* label = [NSString stringWithUTF8String:name];
        [trackedCmdBuffer pushDebugGroup:label];
        if (traceMetalRuntime() || encoderExecutionStatusEnabled())
            m_DebugGroups.push_back(label);
    }

    void CommandList::endMarker()
    {
        [trackedCmdBuffer popDebugGroup];
        if (!m_DebugGroups.empty())
            m_DebugGroups.pop_back();
    }
    nvrhi::IDevice* CommandList::getDevice() { return m_Device; }
}
