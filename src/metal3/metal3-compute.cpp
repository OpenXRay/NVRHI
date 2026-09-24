#include "metal3-backend.h"

namespace nvrhi::metal3
{
    ComputePipelineHandle Device::createComputePipeline(const ComputePipelineDesc& desc)
    {
        auto* cs = static_cast<Shader*>(desc.CS.Get());
        if (!cs || !cs->function)
            return nullptr;

        NSError* error = nil;
        MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
        descriptor.computeFunction = cs->function;
        id<MTLComputePipelineState> nativePipeline = createComputePipelineState(descriptor, &error);
        if (!nativePipeline)
        {
            std::string message = "[nvrhi] Failed to create Metal compute pipeline";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            m_Context.error(message);
            return nullptr;
        }

        ComputePipeline* pipeline = new ComputePipeline();
        pipeline->desc = desc;
        pipeline->pipeline = nativePipeline;
        if (cs->computeThreadsPerGroupValid)
        {
            pipeline->threadsPerGroup = cs->computeThreadsPerGroup;
        }
        else
        {
            // Metal does not expose the original HLSL [numthreads] shape from
            // the compiled pipeline. A 1D fallback preserves SV_DispatchThreadID.x
            // semantics much better than fabricating a rectangular group.

            /* uncommment for debug info. not really needed unless shader comp fails, which will be an issue long before pipeline creation*/
            // pipeline->threadsPerGroup = MTLSizeMake(std::max<NSUInteger>(1, nativePipeline.threadExecutionWidth), 1, 1);
            // m_Context.warning("[metal3] compute pipeline '" + cs->desc.debugName +
            //     "' missing numthreads metadata; using " +
            //     std::to_string(pipeline->threadsPerGroup.width) + "x1x1 fallback");
        }
        pipeline->computeBindingPlan = resolveMetalStageBindingPlan(cs->reflectedBindingPlan, desc.bindingLayouts);
        return ComputePipelineHandle::Create(pipeline);
    }

    Object ComputePipeline::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::MTL3_ComputePipeline)
            return Object((__bridge void*)pipeline);
        return nullptr;
    }
}
