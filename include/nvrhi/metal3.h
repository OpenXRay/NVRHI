#pragma once

#include <nvrhi/nvrhi.h>

#ifndef __OBJC__
#error "nvrhi/metal3.h uses native Metal Objective-C types and must be included from Objective-C++."
#endif

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

namespace nvrhi
{
    namespace ObjectTypes
    {
        constexpr ObjectType Nvrhi_Metal3_Device      = 0x00040101;
        constexpr ObjectType Nvrhi_Metal3_CommandList = 0x00040102;
        constexpr ObjectType Nvrhi_Metal3_EventQuery = 0x00040103;
        constexpr ObjectType Nvrhi_Metal3_LifetimeTracker = 0x00040104;
        constexpr ObjectType MTL3_Buffer              = 0x00040004;
        constexpr ObjectType MTL3_Sampler             = 0x00040005;
        constexpr ObjectType MTL3_RenderPipeline      = 0x00040006;
        constexpr ObjectType MTL3_ComputePipeline     = 0x00040007;
        constexpr ObjectType MTL3_Library             = 0x00040008;
        constexpr ObjectType MTL3_Function            = 0x00040009;
        constexpr ObjectType MTL3_AccelerationStructure = 0x0004000A;
    };
}

namespace nvrhi::metal3
{
    /* good approach i thought of earlier but following nvrhi conventions i am gonna do
     * it the getNativeObject way instead. no need to get them native here
    */
    class ICommandList : public nvrhi::ICommandList
    {
    public:
        virtual id<MTLCommandBuffer> getNativeCommandBuffer() = 0;
        virtual void encodeExternal(void (^encode)(id<MTLCommandBuffer> commandBuffer, id<MTLFence> fence)) = 0;
        virtual void setTracyGpuScope(const char* name, const char* file, const char* function, uint32_t line, void* context) = 0;
        virtual void clearTracyGpuScope() = 0;
    };

    typedef RefCountPtr<ICommandList> CommandListHandle;

    class IDevice : public nvrhi::IDevice
    {
    public:
        // add some metal backend specific functions functions if required
        // virtual id<MTLDevice> getNativeDevice() const = 0;
        // virtual id<MTLCommandQueue> getNativeCommandQueue(CommandQueue queue) const = 0;
    };

    typedef RefCountPtr<IDevice> DeviceHandle;

    struct DeviceDesc
    {
        IMessageCallback* errorCB = nullptr;

        id<MTLDevice> pDevice = nil;

        /* this is kinda confusing. With d3d12 and vulkan we need separate command queues
         * but with metal all work can be done with a single command queue -graphics, compute, copy all
         * the distinction is in the command buffers and command encoders used to record and commit work to gpu
         * -> create single command queue per app
         * -> create command buffer whenever needed on the fly (yes no need to cache it, its use and throw)
         * -> create command encoders on the fly dep on work (graphics, compute, etc)
         * -> record commands
         * -> commit command buffer. It frees the memory so we cant use it now
         * Internally, the command buffers are recorded in sequence to the queue, and then ran in the order on the GPU
         * alternatively, other queues for compute, asset streaming can be added to offload tasks
         * and not block the common queue
        */
        id<MTLCommandQueue> commonQueue = nil;
        id<MTLCommandQueue> computeQueue = nil;

        uint32_t maxTimerQueries = 256;
        bool logBufferLifetime = false;
    };

    NVRHI_API DeviceHandle createDevice(const DeviceDesc& desc);

    NVRHI_API MTLPixelFormat convertFormat(nvrhi::Format format);
}
