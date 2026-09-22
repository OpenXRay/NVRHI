#pragma once

#include <mutex>

#include "nvrhi/common/aftermath.h"
#include "nvrhi/common/resource.h"
#include "../common/state-tracking.h"
#include "nvrhi/nvrhi.h"
#include <nvrhi/metal3.h>
#include <nvrhi/utils.h>
#include <dispatch/dispatch.h>
#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct IRDescriptorTableEntry;

#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
#include <tracy/TracyMetal.hmm>
#endif

namespace nvrhi::metal3 
{
    class Device;
    class CommandListLifetimeTracker;
    class Texture;
    class Buffer;
    class Shader;
    class Sampler;
    class InputLayout;
    class Framebuffer;
    class GraphicsPipeline;
    class ComputePipeline;
    class BindingLayout;
    class BindingSet;
    class MeshletPipeline;
    class EventQuery;

    // useful for shader reflection
    enum class MscArgumentType : uint8_t
    {
        SRV,
        UAV,
        CBV,
        Sampler
    };

    struct MscArgumentBinding
    {
        uint32_t index = 0;
        uint32_t slot = 0;
        uint32_t space = 0;
        MscArgumentType type = MscArgumentType::SRV;
        uint32_t byteOffset = ~0u;
        uint64_t sizeBytes = 24;
        uint32_t tableIndex = ~0u;
    };

    struct MscDescriptorTable
    {
        uint32_t byteOffset = 0;
        uint32_t descriptorCount = 0;
        uint32_t slot = 0;
        uint32_t space = 0;
        MscArgumentType type = MscArgumentType::SRV;
        uint32_t layoutIndex = ~0u;
        uint32_t descriptorOffset = 0;
    };

    struct MscShaderReflection
    {
        bool valid = false;
        bool needsFunctionConstants = false;
        bool directlyIndexedResourceHeap = false;
        bool directlyIndexedSamplerHeap = false;
        uint32_t resourceCount = 0;
        uint64_t argumentBufferSize = 0;
        uint32_t vertexOutputSizeInBytes = 0;
        uint32_t maxInputPrimitivesPerMeshThreadgroup = 0;
        uint32_t geometryInstanceCount = 1;
        uint32_t maxPayloadSizeInBytes = 0;
        std::string shaderType;
        std::string inputPrimitive;
        std::vector<MscArgumentBinding> topLevelArgumentBuffer;
        std::vector<MscDescriptorTable> descriptorTables;
        std::unordered_map<std::string, uint32_t> vertexInputAttributes;
    };
    // shader reflected data, first part basically the MscArgumentBinding i,e per resource
    struct MetalBindingPlanEntry
    {
        uint32_t argumentIndex = 0;
        uint32_t slot = 0;
        uint32_t space = 0;
        MscArgumentType argumentType = MscArgumentType::SRV;
        uint32_t byteOffset = ~0u;
        uint64_t sizeBytes = 24;

        uint32_t layoutIndex = ~0u;
        uint32_t layoutItemIndex = ~0u;
        ResourceType layoutType = ResourceType::None;
        bool layoutMatched = false;
    };
    // per shader stage binding plan; entries -> per resource plan
    struct MetalStageBindingPlan
    {
        ShaderType stage = ShaderType::None;
        std::string debugName;
        bool valid = false;
        bool directlyIndexedResourceHeap = false;
        bool directlyIndexedSamplerHeap = false;
        uint32_t resourceCount = 0;
        uint64_t argumentBufferSize = 0;
        std::vector<MetalBindingPlanEntry> entries;
        std::vector<MscDescriptorTable> descriptorTables;
    };

    struct MetalArgumentTableCacheKey
    {
        const MetalStageBindingPlan* plan = nullptr;
        size_t bindingSetCount = 0;
        std::array<const IBindingSet*, c_MaxBindingLayouts> bindingSets{};
        std::array<uint64_t, c_MaxBindingLayouts> bindingSetVersions{};

        bool operator==(const MetalArgumentTableCacheKey& other) const
        {
            return plan == other.plan &&
                bindingSetCount == other.bindingSetCount &&
                bindingSets == other.bindingSets &&
                bindingSetVersions == other.bindingSetVersions;
        }
    };

    struct MetalArgumentTableCacheKeyHash
    {
        size_t operator()(const MetalArgumentTableCacheKey& key) const
        {
            size_t hash = std::hash<const void*>()(key.plan);
            for (size_t index = 0; index < key.bindingSetCount; ++index)
            {
                hash ^= std::hash<const void*>()(key.bindingSets[index]) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
                hash ^= std::hash<uint64_t>()(key.bindingSetVersions[index]) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    struct ArgumentTableAllocation
    {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        void* cpuAddress = nullptr;
    };

    struct MetalArgumentTableCacheEntry
    {
        ArgumentTableAllocation allocation;
        uint64_t volatileBufferVersion = 0;
    };

    // implementation in metal3-constants.cpp
    MTLTextureType convertTextureDimension(TextureDimension dimension, uint32_t sampleCount);
    MTLResourceOptions convertCpuAccess(CpuAccessMode cpuAccess);
    MTLVertexFormat convertVertexFormat(Format format);
    MTLIndexType convertIndexFormat(Format format);
    MTLPrimitiveType convertPrimitiveType(PrimitiveType primitiveType);
    MTLCullMode convertCullMode(RasterCullMode cullMode);
    MTLWinding convertWinding(bool frontCounterClockwise);
    MTLCompareFunction convertCompareFunction(ComparisonFunc func);
    MTLSamplerAddressMode convertSamplerAddressMode(SamplerAddressMode mode);
    MTLBlendFactor convertBlendFactor(BlendFactor factor);
    MTLBlendOperation convertBlendOp(BlendOp op);

    class ResidencyRegistry;
    bool traceMetalRuntime();

    // metal 3 context
    struct MTL3Context
    {
        id<MTLDevice> device;
        id<MTLCommandQueue> commonQueue;
        id<MTLCommandQueue> queues[uint32_t(CommandQueue::Count)] = {};
        id<MTLCommandQueue> queue(CommandQueue index) const { return queues[uint32_t(index)] ? queues[uint32_t(index)] : commonQueue; }
        std::shared_ptr<ResidencyRegistry> residency;

        bool logBufferLifetime = false;
        uint32_t maxTextureDimension = 16384;
        IMessageCallback* messageCallback = nullptr;
        void error(const std::string& message) const;
        void warning(const std::string& message) const;
        void info(const std::string& message) const;
        void unsupported(const char* operation) const;
    };

    class ResidencyRegistry
    {
    public:
        explicit ResidencyRegistry(const MTL3Context& context);
        ~ResidencyRegistry();
        void add(id<MTLAllocation> allocation);
        void remove(id<MTLAllocation> allocation);
        void commit();

    private:
        std::mutex m_Mutex;
        id<MTLResidencySet> m_Set = nil;
        std::vector<id<MTLCommandQueue>> m_Queues;
        bool m_Dirty = false;
    };
    // created at time of shader creation, using reflection data
    MetalStageBindingPlan createMetalStageBindingPlan(ShaderType stage, const MscShaderReflection& reflection);
    // resolve at pipeline creation per stage
    MetalStageBindingPlan resolveMetalStageBindingPlan(const MetalStageBindingPlan& reflectedPlan, const BindingLayoutVector& pipelineLayouts);
    
    // basically a little allocator for temp CPU written, GPU read upload memory
    struct UploadAllocation
    {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        void* cpuAddress = nullptr;
    };

    struct VolatileBufferAllocation
    {
        UploadAllocation allocation;
        size_t writtenSize = 0;
        uint64_t version = 0;
    };

    // metal resource cache per nvrhi binding item
    struct MetalBindingResource
    {
        ResourceType type = ResourceType::None;
        uint32_t slot = 0;
        uint32_t arrayElement = 0;
        uint32_t registerSpace = 0;

        ResourceHandle resource;

        id<MTLTexture> texture = nil;
        id<MTLBuffer> buffer = nil;
        NSUInteger bufferOffset = 0;
        NSUInteger bufferSize = 0;
        id<MTLSamplerState> sampler = nil;
        float samplerMipBias = 0.f;
        uint32_t textureViewOffsetInElements = 0;

        id<MTLAccelerationStructure> accelerationStructure = nil;
        MTLResourceUsage usage = MTLResourceUsageRead;
    };

    bool encodeMetalBindingResource(IRDescriptorTableEntry* entry, const MetalBindingResource& resource);

    struct DescriptorTableSnapshot
    {
        id<MTLBuffer> buffer = nil;
        std::shared_ptr<ResidencyRegistry> residency;
        ~DescriptorTableSnapshot();
    };

    class UploadManager
    {
    public:
        UploadManager(const MTL3Context& context, size_t uploadChunkSize, size_t scratchMaxMem,
            bool isScratchBuffer, size_t minimumChunkSize = 4 * 1024 * 1024,
            size_t initialChunkCount = 0);
        ~UploadManager();
        void beginCommandBuffer();
        void submitCommandBuffer(id<MTLCommandBuffer> commandBuffer);
        void discardCommandBuffer();
        UploadAllocation suballocate(size_t size, size_t alignment);
        size_t getChunkCount() const { return m_Chunks.size(); }

    private:
        struct Chunk
        {
            id<MTLBuffer> buffer = nil;
            uint8_t* cpuAddress = nullptr;
            size_t size = 0;
            size_t offset = 0;
            uint64_t lastUsedSerial = 0;
        };

        const MTL3Context& m_Context;
        size_t m_DefaultChunkSize = 0;
        uint64_t m_SubmittedSerial = 0;
        uint64_t m_ActiveSerial = 0;
        size_t m_CurrentChunk = size_t(-1);
        std::shared_ptr<std::atomic<uint64_t>> m_CompletedSerial;
        std::vector<Chunk> m_Chunks;

        Chunk* findOrCreateChunk(size_t size, size_t alignment);
    };

    struct TransientBufferAllocation
    {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        NSUInteger size = 0;
        // null for private-storage pages
        void* cpuAddress = nullptr;
    };

    // Per-command-buffer temporary storage for indirect draw helpers. A slot is
    // reused only after Metal reports completion for the command buffer that
    // used it, so page contents and ICB commands stay valid for the GPU.
    class TransientIndirectResourcePool
    {
    public:
        struct Stats
        {
            size_t sharedPageCount = 0;
            size_t privatePageCount = 0;
            size_t sharedPagesCreated = 0;
            size_t privatePagesCreated = 0;
            size_t bytesReserved = 0;
            size_t indirectCommandBuffersReused = 0;
            size_t indirectCommandBuffersCreated = 0;
            bool usedOverflowSlot = false;
        };

        explicit TransientIndirectResourcePool(const MTL3Context& context);
        ~TransientIndirectResourcePool();
        void beginCommandBuffer();
        void submitCommandBuffer(id<MTLCommandBuffer> commandBuffer);
        void discardCommandBuffer();
        Stats getActiveStats() const;
        TransientBufferAllocation allocateShared(NSUInteger size, NSUInteger alignment = 256);
        TransientBufferAllocation allocatePrivate(NSUInteger size, NSUInteger alignment = 256);
        id<MTLIndirectCommandBuffer> acquireIndirectCommandBuffer(
            MTLIndirectCommandBufferDescriptor* descriptor, NSUInteger requiredCapacity,
            bool* wasCreated = nullptr, NSUInteger* capacity = nullptr);

    private:
        struct BufferPage
        {
            id<MTLBuffer> buffer = nil;
            NSUInteger capacity = 0;
            NSUInteger writeOffset = 0;
            uint8_t* cpuAddress = nullptr;  // only for shared pages
        };

        struct IndirectCommandBufferKey
        {
            // every descriptor property that changes ICB compatibility:
            MTLIndirectCommandType commandTypes{};
            bool inheritPipelineState = false;
            bool inheritBuffers = false;
            NSUInteger maxVertexBufferBindCount = 0;
            NSUInteger maxFragmentBufferBindCount = 0;
            NSUInteger maxObjectBufferBindCount = 0;
            NSUInteger maxMeshBufferBindCount = 0;

            bool operator==(const IndirectCommandBufferKey& other) const;
        };

        struct PooledIndirectCommandBuffer
        {
            IndirectCommandBufferKey key;
            NSUInteger capacity = 0;    // maxCommandCount used at creation
            id<MTLIndirectCommandBuffer> commandBuffer = nil;
        };

        struct FrameSlot
        {
            bool inFlight = false;
            bool isOverflow = false;
            Stats stats;
            std::vector<BufferPage> sharedPages;
            std::vector<BufferPage> privatePages;
            std::vector<PooledIndirectCommandBuffer> indirectCommandBuffers;
        };

        struct State
        {
            std::mutex mutex;
            std::vector<FrameSlot> slots;
        };

        const MTL3Context& m_Context;
        // Completion handlers retain State, not CommandList, so a submitted
        // command buffer cannot update a destroyed CommandList instance.
        std::shared_ptr<State> m_State;
        size_t m_ActiveSlot = size_t(-1);

        TransientBufferAllocation allocate(NSUInteger size, NSUInteger alignment,
            MTLResourceOptions options, std::vector<BufferPage> FrameSlot::*pages);
    };

    class Texture : public RefCounter<ITexture>
    {
    public:
        TextureDesc desc;
        TextureStateExtension stateExtension{desc};
        id<MTLTexture> texture = nil;
        NSUInteger memSize;
        NSUInteger memAlign;
        bool ownsTexture = true;
        std::shared_ptr<ResidencyRegistry> residency;
        ~Texture() override;

        std::mutex viewMutex;
        std::vector<id<MTLTexture>> views;
        id<MTLTexture> getView(Format format, TextureSubresourceSet subresources, TextureDimension dimension,
            std::optional<ComponentMapping> componentMapping = std::nullopt);
        const TextureDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
        Object getNativeView(ObjectType objectType, Format format = Format::UNKNOWN, TextureSubresourceSet subresources = AllSubresources, TextureDimension dimension = TextureDimension::Unknown, bool isReadOnlyDSV = false, std::optional<ComponentMapping> overrideComponentMapping = std::nullopt) override;
    };


    class Buffer : public RefCounter<IBuffer>
    {
    public:
        BufferDesc desc;
        BufferStateExtension stateExtension{desc};
        id<MTLBuffer> buffer = nil;
        bool ownsBuffer = true;
        std::shared_ptr<ResidencyRegistry> residency;
        std::array<uint64_t, uint32_t(CommandQueue::Count)> lastUseSubmissions{};
        ~Buffer() override;

        const BufferDesc& getDesc() const override { return desc; }
        GpuVirtualAddress getGpuVirtualAddress() const override { return buffer ? buffer.gpuAddress : 0; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class Shader : public RefCounter<IShader>
    {
    public:
        ShaderDesc desc;
        id<MTLLibrary> library = nil;
        id<MTLLibrary> stageInLibrary = nil;
        id<MTLFunction> function = nil;
        std::vector<uint8_t> bytecode;
        MscShaderReflection mscReflection;
        MetalStageBindingPlan reflectedBindingPlan;
        MTLSize computeThreadsPerGroup = MTLSizeMake(1, 1, 1);
        bool computeThreadsPerGroupValid = false;

        const ShaderDesc& getDesc() const override { return desc; }
        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
    };

    class ShaderLibrary : public RefCounter<IShaderLibrary>
    {
    public:
        id<MTLLibrary> library = nil;
        std::vector<uint8_t> bytecode;

        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
        ShaderHandle getShader(const char* entryName, ShaderType shaderType) override;
    };

    class Sampler : public RefCounter<ISampler>
    {
    public:
        SamplerDesc desc;
        id<MTLSamplerState> sampler = nil;
        const SamplerDesc& getDesc() const override { return desc; }
    };

    class InputLayout : public RefCounter<IInputLayout>
    {
    public:
        std::vector<VertexAttributeDesc> attributes;
        MTLVertexDescriptor* vertexDescriptor = nil;
        uint32_t getNumAttributes() const override { return uint32_t(attributes.size()); }
        const VertexAttributeDesc* getAttributeDesc(uint32_t index) const override;
    };

    class Framebuffer : public RefCounter<IFramebuffer>
    {
    public:
        FramebufferDesc desc;
        FramebufferInfoEx framebufferInfo;
        const FramebufferDesc& getDesc() const override { return desc; }
        const FramebufferInfoEx& getFramebufferInfo() const override { return framebufferInfo; }
    };

    class GraphicsPipeline : public RefCounter<IGraphicsPipeline>
    {
    public:
        GraphicsPipelineDesc desc;
        FramebufferInfo framebufferInfo;
        id<MTLRenderPipelineState> pipeline = nil;
        id<MTLDepthStencilState> depthStencilState = nil;
        MTLPrimitiveType primitiveType = MTLPrimitiveTypeTriangle;
        MTLCullMode cullMode = MTLCullModeNone;
        MTLWinding frontWinding = MTLWindingClockwise;
        bool usesGeometryEmulation = false;
        uint32_t geometryVertexSizeInBytes = 0;
        uint32_t geometryMaxInputPrimitivesPerMeshThreadgroup = 0;
        uint32_t geometryInstanceCount = 1;
        MetalStageBindingPlan vertexBindingPlan;
        // binding plan for object stage
        // for emulated geometry pipeline, the pipeline transforms to:
        // VS (object stage) -> GS -> PS
        MetalStageBindingPlan objectBindingPlan;
        MetalStageBindingPlan fragmentBindingPlan;
        MetalStageBindingPlan meshBindingPlan;
        const GraphicsPipelineDesc& getDesc() const override { return desc; }
        const FramebufferInfo& getFramebufferInfo() const override { return framebufferInfo; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class MeshletPipeline : public RefCounter<IMeshletPipeline>
    {
    public:
        MeshletPipelineDesc desc;
        FramebufferInfo framebufferInfo;
        id<MTLRenderPipelineState> pipeline = nil;
        id<MTLDepthStencilState> depthStencilState = nil;
        MTLCullMode cullMode = MTLCullModeNone;
        MTLWinding frontWinding = MTLWindingClockwise;
        bool hasObjectStage = false;
        MTLSize threadsPerObjectThreadgroup = MTLSizeMake(1, 1, 1);
        MTLSize threadsPerMeshThreadgroup = MTLSizeMake(1, 1, 1);
        MetalStageBindingPlan objectBindingPlan;
        MetalStageBindingPlan meshBindingPlan;
        MetalStageBindingPlan fragmentBindingPlan;
        const MeshletPipelineDesc& getDesc() const override { return desc; }
        const FramebufferInfo& getFramebufferInfo() const override { return framebufferInfo; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class AccelStruct : public RefCounter<rt::IAccelStruct>
    {
    public:
        class GeometryState
        {
        public:
            const void* vertexBuffer = nullptr;
            const void* indexBuffer = nullptr;
            uint64_t vertexOffset = 0;
            uint64_t indexOffset = 0;
            uint64_t vertexCount = 0;
            uint64_t indexCount = 0;
            uint32_t vertexStride = 0;
            uint32_t vertexFormat = 0;
            uint32_t indexFormat = 0;
            uint32_t geometryFlags = 0;
            bool indexed = false;
            bool useTransform = false;
            rt::AffineTransform transform{};
        };

        rt::AccelStructDesc desc;
        id<MTLAccelerationStructure> accelStruct = nil;
        id<MTLBuffer> gpuHeaderBuffer = nil;
        id<MTLBuffer> instanceContributionsBuffer = nil;
        std::shared_ptr<ResidencyRegistry> residency;
        bool compacted = false;
        bool allowCompaction = false;
        bool allowUpdate = false;
        bool built = false;
        uint32_t buildSerial = 0;
        MTLAccelerationStructureUsage builtUsage = MTLAccelerationStructureUsageNone;
        MTLAccelerationStructureSizes builtSizes = {};
        size_t builtInstanceCount = 0;
        std::vector<GeometryState> builtGeometry;
        std::vector<id<MTLBuffer>> builtGeometryBuffers;
        std::vector<id<MTLAccelerationStructure>> builtBottomLevelAS;
        std::vector<MTLAccelerationStructureUserIDInstanceDescriptor> builtInstances;
        ~AccelStruct() override;
        const rt::AccelStructDesc& getDesc() const override { return desc; }
        bool isCompacted() const override { return compacted; }
        uint64_t getDeviceAddress() const override { return gpuHeaderBuffer ? gpuHeaderBuffer.gpuAddress : 0; }
        Object getNativeObject(ObjectType objectType) override;
    };
    class ComputePipeline : public RefCounter<IComputePipeline>
    {
    public:
        ComputePipelineDesc desc;
        id<MTLComputePipelineState> pipeline = nil;
        MTLSize threadsPerGroup = MTLSizeMake(1, 1, 1);
        MetalStageBindingPlan computeBindingPlan;
        const ComputePipelineDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class BindingLayout : public RefCounter<IBindingLayout>
    {
    public:
        BindingLayoutDesc desc;
        BindlessLayoutDesc bindlessDesc;
        bool isBindless = false;
        const BindingLayoutDesc* getDesc() const override { return isBindless ? nullptr : &desc; }
        const BindlessLayoutDesc* getBindlessDesc() const override { return isBindless ? &bindlessDesc : nullptr; }
    };

    class BindingSet : public RefCounter<IBindingSet>
    {
    public:
        BindingSetDesc desc;
        BindingLayoutHandle layout;
        std::vector<ResourceHandle> resources;
        std::vector<MetalBindingResource> entries;
        uint64_t version = 1;
        const BindingSetDesc* getDesc() const override { return &desc; }
        IBindingLayout* getLayout() const override { return layout; }
    };

    class DescriptorTable : public RefCounter<IDescriptorTable>
    {
    public:
        BindingLayoutHandle layout;
        mutable std::mutex mutex;
        std::vector<MetalBindingResource> entries;
        std::shared_ptr<DescriptorTableSnapshot> snapshot;
        uint64_t version = 1;
        const BindingSetDesc* getDesc() const override { return nullptr; }
        IBindingLayout* getLayout() const override { return layout; }
        uint32_t getCapacity() const override;
        uint32_t getFirstDescriptorIndexInHeap() const override { return 0; }
        std::shared_ptr<DescriptorTableSnapshot> getSnapshot(const MTL3Context& context);
    };

    struct TrackedCommandBuffer
    {
        id<MTLCommandBuffer> commandBuffer = nil;
        std::vector<BindingSetHandle> bindingSets;
        std::vector<id<MTLBuffer>> buffers;
        std::vector<id<MTLResource>> resources;
        std::vector<std::shared_ptr<DescriptorTableSnapshot>> descriptorSnapshots;
        std::vector<ResourceHandle> stateResources;
    };

    struct TimerQueryPool
    {
        static constexpr NSUInteger QueriesPerPage = 256;
        static constexpr NSUInteger SamplesPerQuery = 4;

        struct Page
        {
            id<MTLCounterSampleBuffer> samples = nil;
            id<MTLBuffer> markers = nil;
            std::vector<NSUInteger> freeSamples;
        };

        std::mutex mutex;
        std::vector<Page> pages;
        id<MTLCounterSet> counterSet = nil;
        id<MTLComputePipelineState> markerPipeline = nil;
        bool timestampsInNanoseconds = false;
    };

    class TimerQuery : public RefCounter<ITimerQuery>
    {
    public:
        ~TimerQuery() override;
        std::shared_ptr<TimerQueryPool> pool;
        size_t pageIndex = 0;
        NSUInteger sampleIndex = 0;
        id<MTLCounterSampleBuffer> samples = nil;
        id<MTLBuffer> markers = nil;
        id<MTLCommandBuffer> commandBuffer = nil;
        bool started = false;
        bool ended = false;
        MTLTimestamp cpuStartTimestamp = 0;
        MTLTimestamp gpuStartTimestamp = 0;
    };

    class CommandListLifetimeTracker : public RefCounter<ICommandListLifetimeTracker>
    {
    public:
        CommandListLifetimeTracker(Device* device, CommandQueue queue, bool isDefault = false);
        ~CommandListLifetimeTracker() override;
        Object getNativeObject(ObjectType objectType) override;
        void runGarbageCollection() override;

    private:
        friend class Device;
        friend class CommandList;
        void collect();
        TrackedCommandBuffer acquireResourceStorage();
        Device* m_Device;
        CommandQueue m_Queue;
        bool m_IsDefault;
        std::deque<TrackedCommandBuffer> m_CommandBuffers;
        std::vector<TrackedCommandBuffer> m_ResourceStorage;
        size_t m_ResourceStorageHighWaterMark = 0;
    };

    class EventQuery : public RefCounter<IEventQuery>
    {
    public:
        explicit EventQuery(Device* owner) : device(owner) {}
        Object getNativeObject(ObjectType objectType) override
        {
            return objectType == ObjectTypes::Nvrhi_Metal3_EventQuery ? Object(this) : Object(nullptr);
        }
        Device* device;
        id<MTLCommandBuffer> commandBuffer = nil;
        std::array<uint64_t, uint32_t(CommandQueue::Count)> submissions{};
        bool failureReported = false;
    };

    // commandList impl
    class CommandList final : public RefCounter<nvrhi::metal3::ICommandList>
    {
    public:

        // TODO: Internal interface functions (metal 3 specific)
        CommandList(class Device* device, const MTL3Context& context, const CommandListParameters& params);
        ~CommandList() override;

        // IResource implementation
        Object getNativeObject(ObjectType objectType) override;

        // ICommandList implementation
        void open() override;
        void close() override;
        void clearState() override;
        
        void clearTextureFloat(ITexture* t, TextureSubresourceSet subresources, const Color& clearColor) override;
        void clearDepthStencilTexture(ITexture* t, TextureSubresourceSet subresources, bool clearDepth, float depth, bool clearStencil, uint8_t stencil) override;
        void clearTextureUInt(ITexture* t, TextureSubresourceSet subresources, uint32_t clearColor) override;
        void clearSamplerFeedbackTexture(ISamplerFeedbackTexture* texture) override;
        void decodeSamplerFeedbackTexture(IBuffer* buffer, ISamplerFeedbackTexture* texture, Format format) override;
        void setSamplerFeedbackTextureState(ISamplerFeedbackTexture* texture, ResourceStates stateBits) override;

        void copyTexture(ITexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice) override;
        //// staging texture path DNE ////
        void copyTexture(IStagingTexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice) override;
        void copyTexture(ITexture* dest, const TextureSlice& destSlice, IStagingTexture* src, const TextureSlice& srcSlice) override;
        //// staging texture path DNE /////
        void writeTexture(ITexture* dest, uint32_t arraySlice, uint32_t mipLevel, const void* data, size_t rowPitch, size_t depthPitch) override;
        void resolveTexture(ITexture* dest, const TextureSubresourceSet& dstSubresources, ITexture* src, const TextureSubresourceSet& srcSubresources) override;

        void writeBuffer(IBuffer* b, const void* data, size_t dataSize, uint64_t destOffsetBytes = 0) override;
        void clearBufferUInt(IBuffer* b, uint32_t clearValue) override;
        void copyBuffer(IBuffer* dest, uint64_t destOffsetBytes, IBuffer* src, uint64_t srcOffsetBytes, uint64_t dataSizeBytes) override;

        void setPushConstants(const void* data, size_t byteSize) override;

        void setGraphicsState(const GraphicsState& state) override;
        // for draw* commands:
        /* 
        // Metal Shader Converter vertex shaders read draw parameters from
        // the IR runtime bind points. Use the runtime wrapper for indirect
        // draws too; raw Metal draws do not populate those translated
        // D3D-style draw parameter bindings.
        */
        void draw(const DrawArguments& args) override;
        void drawIndexed(const DrawArguments& args) override;

        // supplying with draw count, just loops over the draw indices and issue
        // the draw calls one at a time, cpu side
        void drawIndirect(uint32_t offsetBytes, uint32_t drawCount) override;
        void drawIndexedIndirect(uint32_t offsetBytes, uint32_t drawCount) override;

        void drawIndexedIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount) override;

        void setComputeState(const ComputeState& state) override;
        void dispatch(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1) override;
        void dispatchIndirect(uint32_t offsetBytes) override;

        void setMeshletState(const MeshletState& state) override;
        void dispatchMesh(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1) override;
        void dispatchMeshIndirect(uint32_t offsetBytes, uint32_t maxDrawCount) override;
        void dispatchMeshIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount) override;

        void setRayTracingState(const rt::State& state) override;
        void dispatchRays(const rt::DispatchRaysArguments& args) override;

        void buildOpacityMicromap(rt::IOpacityMicromap* omm, const rt::OpacityMicromapDesc& desc) override;
        void copyRaytracingAccelerationStructure(rt::IAccelStruct* destination, rt::IAccelStruct* source) override;
        void buildBottomLevelAccelStruct(rt::IAccelStruct* as, const rt::GeometryDesc* pGeometries, size_t numGeometries, rt::AccelStructBuildFlags buildFlags) override;
        void compactBottomLevelAccelStructs() override;
        void buildTopLevelAccelStruct(rt::IAccelStruct* as, const rt::InstanceDesc* pInstances, size_t numInstances, rt::AccelStructBuildFlags buildFlags) override;
        void buildTopLevelAccelStructFromBuffer(rt::IAccelStruct* as, nvrhi::IBuffer* instanceBuffer, uint64_t instanceBufferOffset, size_t numInstances,
            rt::AccelStructBuildFlags buildFlags = rt::AccelStructBuildFlags::None) override;
        void executeMultiIndirectClusterOperation(const rt::cluster::OperationDesc& desc) override;

        void convertCoopVecMatrices(coopvec::ConvertMatrixLayoutDesc const* convertDescs, size_t numDescs) override;

        void beginTimerQuery(ITimerQuery* query) override;
        void endTimerQuery(ITimerQuery* query) override;

        void beginMarker(const char *name) override;
        void endMarker() override;

        void setEnableAutomaticBarriers(bool enable) override;
        void setResourceStatesForBindingSet(IBindingSet* bindingSet) override;

        void setEnableUavBarriersForTexture(ITexture* texture, bool enableBarriers) override;
        void setEnableUavBarriersForBuffer(IBuffer* buffer, bool enableBarriers) override;

        void beginTrackingTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits) override;
        void beginTrackingBufferState(IBuffer* buffer, ResourceStates stateBits) override;

        void setTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits) override;
        void setBufferState(IBuffer* buffer, ResourceStates stateBits) override;
        void setAccelStructState(rt::IAccelStruct* as, ResourceStates stateBits) override;
        
        void setPermanentTextureState(ITexture* texture, ResourceStates stateBits) override;
        void setPermanentBufferState(IBuffer* buffer, ResourceStates stateBits) override;

        void commitBarriers() override;

        ResourceStates getTextureSubresourceState(ITexture* texture, ArraySlice arraySlice, MipLevel mipLevel) override;
        ResourceStates getBufferState(IBuffer* buffer) override;

        nvrhi::IDevice* getDevice() override;
        const CommandListParameters& getDesc() override { return m_Desc; }

        // Metal3 specific methods
        id<MTLCommandBuffer> getNativeCommandBuffer() override;
        void encodeExternal(void (^encode)(id<MTLCommandBuffer> commandBuffer, id<MTLFence> fence)) override;
        void setTracyGpuScope(const char* name, const char* file, const char* function, uint32_t line, void* context) override;
        void clearTracyGpuScope() override;
    private:
        friend class Device;

        enum class RecordingState
        {
            Initial,
            Open,
            Closed,
            PendingSubmission,
            Submitted
        };

        RecordingState m_RecordingState = RecordingState::Initial;
        bool m_RecordingFailed = false;
        void unsupported(const char* operation);
        void submit();

        struct TracyGpuScopeDesc
        {
            const char* name = nullptr;
            const char* file = nullptr;
            const char* function = nullptr;
            uint32_t line = 0;
            void* context = nullptr;
            bool active = false;
        };
        const MTL3Context& m_Context;

        Device* m_Device;
        RefCountPtr<CommandListLifetimeTracker> m_LifetimeTracker;
        
        // i can prolly have Queue struct and pointer to object here
        // but its not really needed, and neither does metal 3 demand it.
        // would rather just have it inside m_Context like i have right now
        //Queue* m_Queue;
        UploadManager m_UploadManager;
        // Argument tables must use stable memory until command-buffer
        // completion, but must not create a native MTLBuffer per draw.
        UploadManager m_ArgumentTableManager;
        TransientIndirectResourcePool m_TransientIndirectResources;
        CommandListResourceStateTracker m_StateTracker;
        bool m_EnableAutomaticBarriers = true;
        
        CommandListParameters m_Desc;

        // Cache for user-provided state
        GraphicsState m_CurrentGraphicsState;
        ComputeState m_CurrentComputeState;
        MeshletState m_CurrentMeshletState;
        IFramebuffer* m_CurrentFramebuffer = nullptr;
        bool m_CurrentGraphicsStateValid = false;
        bool m_CurrentComputeStateValid = false;
        bool m_CurrentMeshletStateValid = false;
        bool m_GeometryEmulationDrawStateValid = false;
        // Holds the uploaded IRRuntimeVertexBuffers table for the active
        // geometry-emulation draw; object shaders read this to fetch app vertex buffers.
        id<MTLBuffer> m_GeometryEmulationVertexBuffers = nil;
        NSUInteger m_GeometryEmulationVertexBuffersOffset = 0;

        struct GraphicsBufferBinding
        {
            __unsafe_unretained id<MTLBuffer> buffer = nil;
            NSUInteger offset = 0;
        };
        std::array<GraphicsBufferBinding, 31> m_VertexBufferBindings{};
        std::array<GraphicsBufferBinding, 31> m_FragmentBufferBindings{};

        bool m_BindingStatesDirty = false;
        bool m_ArgumentTablesDirty = false;

        struct PendingClear
        {
            TextureHandle texture;
            MipLevel mipLevel = 0;
            ArraySlice arraySlice = 0;
            bool isDepthStencil = false;
            Color color;
            bool clearDepth = false;
            float depth = 0.f;
            bool clearStencil = false;
            uint8_t stencil = 0;
        };
        std::vector<PendingClear> m_PendingClears;
        std::vector<const void*> m_RenderEncoderWrites;
        std::vector<const void*> m_BlitEncoderWrites;

        id<MTLCommandBuffer> trackedCmdBuffer = nil;
        id<MTLRenderCommandEncoder> m_RenderEncoder = nil;
        id<MTLComputeCommandEncoder> m_ComputeEncoder = nil;
        id<MTLBlitCommandEncoder> m_BlitEncoder = nil;
        id<MTLAccelerationStructureCommandEncoder> m_AccelEncoder = nil;
        id<MTLFence> m_EncoderFence = nil;
        bool m_EncoderFenceArmed = false;

        std::vector<BindingSetHandle> m_ReferencedBindingSets;
        std::vector<id<MTLBuffer>> m_ReferencedNativeBuffers;
        std::vector<id<MTLResource>> m_ReferencedNativeResources;
        std::vector<std::shared_ptr<DescriptorTableSnapshot>> m_ReferencedDescriptorSnapshots;
        std::vector<ResourceHandle> m_ReferencedStateResources;
        std::vector<BufferHandle> m_ReferencedMappableBuffers;

        std::unordered_map<MetalArgumentTableCacheKey, MetalArgumentTableCacheEntry, MetalArgumentTableCacheKeyHash> m_ArgumentTableCache;
        uint64_t m_ArgumentTableAllocationCount = 0;
        size_t m_ArgumentTablePageCountAtOpen = 0;

        std::array<uint8_t, c_MaxPushConstantSize> m_PushConstants{};
        size_t m_PushConstantSize = 0;
        std::unordered_map<Buffer*, VolatileBufferAllocation> m_VolatileBufferAllocations;
        uint64_t m_VolatileBufferWriteVersion = 0;
        std::vector<NSString*> m_DebugGroups;
        void beginEncoding(id<MTLAccelerationStructureCommandEncoder> encoder, const char* operation);
        void endEncoding(id<MTLAccelerationStructureCommandEncoder> encoder);

        TracyGpuScopeDesc m_TracyGpuScope;
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        std::optional<tracy::MetalZoneScope> m_TracyEncoderScope;
        std::vector<std::unique_ptr<tracy::SourceLocationData>> m_TracySourceLocations;
#endif

        void endEncoding();
        void beginEncoding(id<MTLRenderCommandEncoder> encoder, const char* operation);
        void beginEncoding(id<MTLComputeCommandEncoder> encoder, const char* operation);
        void beginEncoding(id<MTLBlitCommandEncoder> encoder, const char* operation);
        void endEncoding(id<MTLRenderCommandEncoder> encoder);
        void endEncoding(id<MTLComputeCommandEncoder> encoder);
        void endEncoding(id<MTLBlitCommandEncoder> encoder);
        id<MTLBlitCommandEncoder> beginBlitEncoder();
        id<MTLComputeCommandEncoder> beginTransientComputeEncoder();
        void flushPendingClears();
        void encodeStandaloneClear(const PendingClear& clear);
        void rebindArgumentTables();
        void annotateEncoder(id<MTLCommandEncoder> encoder, const char* operation);
        id<MTLAccelerationStructureCommandEncoder> getOrCreateAccelEncoder(const char* operation);
        bool encodeTimerBoundary(TimerQuery* query, bool ending);
        void bindGraphicsBuffer(id<MTLRenderCommandEncoder> encoder, id<MTLBuffer> buffer,
            NSUInteger offset, NSUInteger index, MTLRenderStages stages);
        id<MTLRenderCommandEncoder> getOrCreateRenderEncoder();
        id<MTLComputeCommandEncoder> getOrCreateComputeEncoder();
        ArgumentTableAllocation getOrCreateArgumentTable(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets);
        ArgumentTableAllocation createArgumentTable(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets);
        void bindGraphicsArgumentTable(id<MTLRenderCommandEncoder> encoder, const BindingSetVector& bindingSets, const MetalStageBindingPlan& plan, MTLRenderStages stages);
        bool encodeArgumentTableEntry(IRDescriptorTableEntry* entry, const MetalBindingResource& resource);
        void bindDescriptorTables(id<MTLRenderCommandEncoder> encoder, const BindingSetVector& bindingSets, const MetalStageBindingPlan& plan, MTLRenderStages stages);
        void bindDescriptorTables(id<MTLComputeCommandEncoder> encoder, const BindingSetVector& bindingSets, const MetalStageBindingPlan& plan);
        bool bindGeometryEmulationVertexBuffers(id<MTLRenderCommandEncoder> encoder, const GraphicsPipeline& pipeline, const GraphicsState& state);
        void drawIndirectGeometryEmulation(uint32_t offsetBytes, uint32_t drawCount);
        void drawIndexedIndirectGeometryEmulation(uint32_t offsetBytes, uint32_t drawCount);
        void drawIndexedIndirectCountGeometryEmulation(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount);
        void applyGraphicsStateToEncoder(id<MTLRenderCommandEncoder> encoder, const GraphicsState& state);
        void applyMeshletStateToEncoder(id<MTLRenderCommandEncoder> encoder, const MeshletState& state);
        void applyBindingSets(const BindingSetVector& bindings);
        void applyComputeBindings(id<MTLComputeCommandEncoder> encoder, const ComputeState& state);
        void referenceBindingSet(IBindingSet* bindingSet);
        void referenceBuffer(IBuffer* buffer);
        void referenceDescriptorSnapshot(const std::shared_ptr<DescriptorTableSnapshot>& snapshot);
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        tracy::SourceLocationData* getOrCreateTracySourceLocation();
        void beginTracyRenderEncoderZone(MTLRenderPassDescriptor* desc);
        void beginTracyComputeEncoderZone(MTLComputePassDescriptor* desc);
#endif
    };

    // metal3 device
    /* 
     * device functions declarations of virtual nvrhi functions declared in nvrhi.h
     * 
    */

    class Device : public RefCounter<nvrhi::metal3::IDevice>
    {
    public:
        explicit Device(const DeviceDesc& desc);
        ~Device() override;
        
        // IResource impl
        Object getNativeObject(ObjectType objectType) override;

        // IDevice impl

        // Metal3: MTLHeap or resource heaps (closest to dx12/vulkan type descriptor heaps) not implemented yet, does nothing
        HeapHandle createHeap(const HeapDesc& d) override;

        TextureHandle createTexture(const TextureDesc& d) override;

        // Metal3: useful for texture allocated inside heaps, for manual/placed allocation; we create committed textures directly currently
        MemoryRequirements getTextureMemoryRequirements(ITexture* texture) override;
        // Metal 3: nothing like binding a texture memory in metal, does nothing, a stub
        bool bindTextureMemory(ITexture* texture, IHeap* heap, uint64_t offset) override;

        TextureHandle createHandleForNativeTexture(ObjectType objectType, Object texture, const TextureDesc& desc) override;

        StagingTextureHandle createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess) override;
        void *mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t *outRowPitch) override;
        void unmapStagingTexture(IStagingTexture* tex) override;

        void getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings) override;
        void updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue = CommandQueue::Graphics) override;

        SamplerFeedbackTextureHandle createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc) override;
        SamplerFeedbackTextureHandle createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture) override;

        BufferHandle createBuffer(const BufferDesc& d) override;
        void *mapBuffer(IBuffer* b, CpuAccessMode mapFlags) override;
        void unmapBuffer(IBuffer* b) override;

        // Metal3: useful for buffers allocated inside heaps, for manual/placed allocation; we create committed buffers directly currently
        MemoryRequirements getBufferMemoryRequirements(IBuffer* buffer) override;
        bool bindBufferMemory(IBuffer* buffer, IHeap* heap, uint64_t offset) override;

        BufferHandle createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc& desc) override;

        // Metal3: -> the way the current pipeline will handle shaders is by processing *.metallib files by shader reflection and 
        // storing arguments (index, slots, types, etc) for use later at resource binding/pipeline creation, execution stage. 
        // -> it also expects a reflection .json file for every shader. 
        // The path supported is using *metalshader-converter* for generating shader reflection data. The pipeline is as follows:
        // - *.hlsl -> *.dxil  (with dxc)
        // - *.dxil -> *.metallib (metalshader-converter)
        // - *.metallib -> *.reflection.json (with msc's reflection aergument)

        ShaderHandle createShader(const ShaderDesc& d, const void* binary, size_t binarySize) override;
        ShaderHandle createShaderSpecialization(IShader* baseShader, const ShaderSpecialization* constants, uint32_t numConstants) override;
        ShaderLibraryHandle createShaderLibrary(const void* binary, size_t binarySize) override;

        SamplerHandle createSampler(const SamplerDesc& d) override;

        InputLayoutHandle createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader) override;

        EventQueryHandle createEventQuery() override;
        void setEventQuery(IEventQuery* query, CommandQueue queue) override;
        bool pollEventQuery(IEventQuery* query) override;
        void waitEventQuery(IEventQuery* query) override;
        void resetEventQuery(IEventQuery* query) override;

        TimerQueryHandle createTimerQuery() override;
        bool pollTimerQuery(ITimerQuery* query) override;
        float getTimerQueryTime(ITimerQuery* query) override;
        void resetTimerQuery(ITimerQuery* query) override;

        GraphicsAPI getGraphicsAPI() override;

        FramebufferHandle createFramebuffer(const FramebufferDesc& desc) override;
        
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo) override;
        
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb) override;
        
        ComputePipelineHandle createComputePipeline(const ComputePipelineDesc& desc) override;

        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, FramebufferInfo const& fbinfo) override;

        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, IFramebuffer* fb) override;

        // Unsupported: RT backend is WIP
        rt::PipelineHandle createRayTracingPipeline(const rt::PipelineDesc& desc) override;

        BindingLayoutHandle createBindingLayout(const BindingLayoutDesc& desc) override;
        BindingLayoutHandle createBindlessLayout(const BindlessLayoutDesc& desc) override;

        BindingSetHandle createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout) override;
        DescriptorTableHandle createDescriptorTable(IBindingLayout* layout) override;

        void resizeDescriptorTable(IDescriptorTable* descriptorTable, uint32_t newSize, bool keepContents = true) override;
        bool writeDescriptorTable(IDescriptorTable* descriptorTable, const BindingSetItem& item) override;

        rt::OpacityMicromapHandle createOpacityMicromap(const rt::OpacityMicromapDesc& desc) override;
        rt::AccelStructHandle createAccelStruct(const rt::AccelStructDesc& desc) override;
        MemoryRequirements getAccelStructMemoryRequirements(rt::IAccelStruct* as) override;
        rt::cluster::OperationSizeInfo getClusterOperationSizeInfo(const rt::cluster::OperationParams& params) override;

        bool bindAccelStructMemory(rt::IAccelStruct* as, IHeap* heap, uint64_t offset) override;

        nvrhi::CommandListHandle createCommandList(const CommandListParameters& params = CommandListParameters()) override;
        uint64_t executeCommandLists(nvrhi::ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue = CommandQueue::Graphics) override;
        void queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance) override;
        bool waitForIdle() override;

        CommandListLifetimeTrackerHandle createCommandListLifetimeTracker(CommandQueue executionQueue) override;

        void runGarbageCollection() override;
        bool queryFeatureSupport(Feature feature, void* pInfo = nullptr, size_t infoSize = 0) override;
        FormatSupport queryFormatSupport(Format format) override;
        coopvec::DeviceFeatures queryCoopVecFeatures() override;
        coopvec::MatMulFormatSupport queryCoopVecMatMulFormatSupport(const coopvec::MatMulFormatCombo& combination) override;
        coopvec::TrainingFormatSupport queryCoopVecTrainingFormatSupport(coopvec::DataType componentType) override;
        size_t getCoopVecMatrixSize(coopvec::DataType type, coopvec::MatrixLayout layout, int rows, int columns) override;
        Object getNativeQueue(ObjectType objectType, CommandQueue queue) override;
        IMessageCallback* getMessageCallback() override { return m_Context.messageCallback; }

        // unused funcs, no aftermath sdk here :/
        bool isAftermathEnabled() override { return m_AftermathEnabled; }
        AftermathCrashDumpHelper& getAftermathCrashDumpHelper() override { return m_AftermathCrashDumpHelper; }

        // TODO: metal.h virtual funs implementation if any
        
    private:
        friend class CommandList;
        friend class CommandListLifetimeTracker;

        struct SubmittedCommandBuffer
        {
            id<MTLCommandBuffer> commandBuffer = nil;
            uint64_t instance = 0;
            bool lastInBatch = false;
        };

        struct QueueState
        {
            uint64_t submitted = 0;
            uint64_t completed = 0;
            uint64_t firstFailure = 0;
            std::deque<SubmittedCommandBuffer> pending;
            id<MTLEvent> event = nil;
            id<MTLFence> fence = nil;
            bool fenceArmed = false;
            std::vector<std::pair<id<MTLEvent>, uint64_t>> pendingWaits;
        };

        struct PendingCompaction
        {
            RefCountPtr<AccelStruct> accelStruct;
            id<MTLBuffer> sizeBuffer = nil;
            id<MTLCommandBuffer> buildCommandBuffer = nil;
            uint32_t buildSerial = 0;
        };

        void queueCompaction(AccelStruct* as, id<MTLBuffer> sizeBuffer, id<MTLCommandBuffer> buildCommandBuffer);
        void takeCompletedCompactions(std::vector<PendingCompaction>& completed);
        std::mutex m_CompactionMutex;
        std::vector<PendingCompaction> m_PendingCompactions;

        void updateCompletedSubmissions();
        bool submissionsSucceeded(const std::array<uint64_t, uint32_t(CommandQueue::Count)>& submissions) const;
        EventQuery* getEventQuery(IEventQuery* query);
        std::array<QueueState, uint32_t(CommandQueue::Count)> m_Queues;
        std::array<RefCountPtr<CommandListLifetimeTracker>, uint32_t(CommandQueue::Count)> m_DefaultLifetimeTrackers;
        std::shared_ptr<TimerQueryPool> m_TimerQueryPool = std::make_shared<TimerQueryPool>();

        CommandList* m_OpenImmediateCommandList = nullptr;
        bool m_CommandBufferConfigurationReported = false;

        bool m_AftermathEnabled;
        AftermathCrashDumpHelper m_AftermathCrashDumpHelper;

        MTL3Context m_Context;

        std::mutex m_Mutex;
    };
}
