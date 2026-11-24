#!/usr/bin/env python3
"""Run isolated NariQ Vulkan shaders. Requires numpy, vulkan, compiled .spv files.
Software Vulkan is useful for correctness only. This is not a full-model test.
"""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
import vulkan as v


def execute(shader, chunks, out_bytes, params):
    app = v.VkApplicationInfo(pApplicationName='NariQ isolated test', apiVersion=v.VK_MAKE_VERSION(1,1,0))
    instance = v.vkCreateInstance(v.VkInstanceCreateInfo(pApplicationInfo=app), None)
    devices = v.vkEnumeratePhysicalDevices(instance)
    if not devices:
        raise RuntimeError('No Vulkan devices')
    physical = devices[0]
    family = next(i for i, x in enumerate(v.vkGetPhysicalDeviceQueueFamilyProperties(physical)) if x.queueFlags & v.VK_QUEUE_COMPUTE_BIT)
    device = v.vkCreateDevice(physical, v.VkDeviceCreateInfo(pQueueCreateInfos=[v.VkDeviceQueueCreateInfo(queueFamilyIndex=family, queueCount=1, pQueuePriorities=[1.0])]), None)
    queue = v.vkGetDeviceQueue(device, family, 0)
    props = v.vkGetPhysicalDeviceMemoryProperties(physical)
    allocations = []
    for raw in chunks + [bytes(out_bytes)]:
        buffer = v.vkCreateBuffer(device, v.VkBufferCreateInfo(size=len(raw), usage=v.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sharingMode=v.VK_SHARING_MODE_EXCLUSIVE), None)
        req = v.vkGetBufferMemoryRequirements(device, buffer)
        flags = v.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | v.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        kind = next(i for i in range(props.memoryTypeCount) if req.memoryTypeBits & (1 << i) and props.memoryTypes[i].propertyFlags & flags == flags)
        memory = v.vkAllocateMemory(device, v.VkMemoryAllocateInfo(allocationSize=req.size, memoryTypeIndex=kind), None)
        v.vkBindBufferMemory(device, buffer, memory, 0)
        ptr = v.vkMapMemory(device, memory, 0, len(raw), 0)
        ptr[:] = raw
        v.vkUnmapMemory(device, memory)
        allocations.append((buffer, memory, len(raw)))
    bindings = [v.VkDescriptorSetLayoutBinding(binding=i, descriptorType=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=v.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(4)]
    ds_layout = v.vkCreateDescriptorSetLayout(device, v.VkDescriptorSetLayoutCreateInfo(pBindings=bindings), None)
    layout = v.vkCreatePipelineLayout(device, v.VkPipelineLayoutCreateInfo(pSetLayouts=[ds_layout], pPushConstantRanges=[v.VkPushConstantRange(stageFlags=v.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=16)]), None)
    module = v.vkCreateShaderModule(device, v.VkShaderModuleCreateInfo(codeSize=len(shader), pCode=shader), None)
    stage = v.VkPipelineShaderStageCreateInfo(stage=v.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName='main')
    info = v.VkComputePipelineCreateInfo(stage=stage, layout=layout)
    pipeline = v.vkCreateComputePipelines(device, v.VK_NULL_HANDLE, 1, [info], None)[0]
    pool = v.vkCreateDescriptorPool(device, v.VkDescriptorPoolCreateInfo(maxSets=1, pPoolSizes=[v.VkDescriptorPoolSize(type=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=4)]), None)
    ds = v.vkAllocateDescriptorSets(device, v.VkDescriptorSetAllocateInfo(descriptorPool=pool, pSetLayouts=[ds_layout]))[0]
    writes = [v.VkWriteDescriptorSet(dstSet=ds, dstBinding=i, descriptorCount=1, descriptorType=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[v.VkDescriptorBufferInfo(buffer=b, offset=0, range=size)]) for i, (b, m, size) in enumerate(allocations)]
    v.vkUpdateDescriptorSets(device, len(writes), writes, 0, None)
    command_pool = v.vkCreateCommandPool(device, v.VkCommandPoolCreateInfo(queueFamilyIndex=family), None)
    cmd = v.vkAllocateCommandBuffers(device, v.VkCommandBufferAllocateInfo(commandPool=command_pool, level=v.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
    v.vkBeginCommandBuffer(cmd, v.VkCommandBufferBeginInfo())
    v.vkCmdBindPipeline(cmd, v.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    v.vkCmdBindDescriptorSets(cmd, v.VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, [ds], 0, None)
    packed = np.asarray(params, dtype=np.uint32).tobytes()
    v.vkCmdPushConstants(cmd, layout, v.VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, v.ffi.from_buffer(packed))
    v.vkCmdDispatch(cmd, (params[1]+63)//64, params[3], 1)
    v.vkEndCommandBuffer(cmd)
    v.vkQueueSubmit(queue, 1, [v.VkSubmitInfo(pCommandBuffers=[cmd])], v.VK_NULL_HANDLE)
    v.vkQueueWaitIdle(queue)
    ptr = v.vkMapMemory(device, allocations[-1][1], 0, out_bytes, 0)
    result = bytes(ptr)
    v.vkUnmapMemory(device, allocations[-1][1])
    name = v.vkGetPhysicalDeviceProperties(physical).deviceName
    v.vkDestroyCommandPool(device, command_pool, None)
    v.vkDestroyDescriptorPool(device, pool, None)
    v.vkDestroyPipeline(device, pipeline, None)
    v.vkDestroyShaderModule(device, module, None)
    v.vkDestroyPipelineLayout(device, layout, None)
    v.vkDestroyDescriptorSetLayout(device, ds_layout, None)
    for b, m, _ in allocations:
        v.vkDestroyBuffer(device, b, None)
        v.vkFreeMemory(device, m, None)
    v.vkDestroyDevice(device, None)
    v.vkDestroyInstance(instance, None)
    return np.frombuffer(result, dtype=np.float32), name


def reference(codes, bits, K, N):
    # Independent bit-string oracle: no shared shifts or funnel logic with GLSL.
    out = np.empty((N,K), dtype=np.float32)
    for row in range(N):
        for col in range(K):
            r, c = row%16, col%16
            order = 8*(4*(r%8)+(c%8)//2)+4*(r//8)+2*(c//8)+c%2
            base = ((row//16)*(K//16)+col//16)*8*bits
            stream = ''.join(format(int(x), '032b') for x in codes[base:base+8*bits])
            end = (order+1)*bits
            state = int(''.join(stream[i%len(stream)] for i in range(end-16,end)), 2)
            h = (state*0x83DCD12D)&0xffffffff
            out[row,col] = sum((h>>s)&63 for s in (2,10,18,26))-126
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('shader_dir', type=Path)
    args = p.parse_args()
    results=[]
    for bits in range(1,7):
        K, N, batch = 48, 32, 3
        rng=np.random.default_rng(1234+bits)
        for pattern in ('random','zero','ones'):
            codes=rng.integers(0,2**32,size=(N*K*bits//32),dtype=np.uint32)
            if pattern=='zero': codes[:]=0
            if pattern=='ones': codes[:]=0xffffffff
            scales=rng.uniform(.01,.2,K//16).astype(np.float16)
            # Pad to 32-bit SSBO packing; the last scale exercises odd count.
            scales_padded=np.pad(scales,(0,len(scales)%2))
            x=rng.uniform(-1,1,(batch,K)).astype(np.float32)
            chunks=[codes.tobytes(),scales_padded.tobytes(),x.tobytes()]
            expected=reference(codes,bits,K,N)
            actual,name=execute((args.shader_dir/'decode.spv').read_bytes(),chunks,N*K*4,[K,N,bits,1])
            np.testing.assert_array_equal(actual.reshape(N,K),expected)
            actual,name=execute((args.shader_dir/'product.spv').read_bytes(),chunks,N*batch*4,[K,N,bits,batch])
            wanted=x.astype(np.float64)@(expected*np.repeat(scales.astype(np.float32),16)).astype(np.float64).T
            np.testing.assert_allclose(actual.reshape(batch,N),wanted,rtol=3e-5,atol=2e-4)
            results.append({'bits':bits,'pattern':pattern,'max_abs_error':float(np.max(np.abs(actual.reshape(batch,N)-wanted)))})
    print(json.dumps({'device':name,'cases':results,'passed':True,'scope':'isolated decode and unrotated product only'},indent=2))

if __name__=='__main__': main()
