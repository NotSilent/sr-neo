use crate::{
    resource_manager::{ResourceManager, VulkanResource},
    vulkan_engine::VulkanContext,
};

use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BufferIndex(u16);

impl From<usize> for BufferIndex {
    fn from(val: usize) -> Self {
        BufferIndex(val as u16)
    }
}

impl From<BufferIndex> for usize {
    fn from(val: BufferIndex) -> Self {
        val.0 as usize
    }
}

pub type BufferManager = ResourceManager<Buffer, (), BufferIndex>;

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
}

impl Buffer {
    pub fn new(
        ctx: &VulkanContext,
        allocator: &mut Allocator,
        alloc_size: usize,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        name: &str,
    ) -> Option<Buffer> {
        let create_info = vk::BufferCreateInfo::default()
            .size(alloc_size as u64)
            .usage(usage);

        let buffer = unsafe { ctx.create_buffer(&create_info, None).unwrap() };
        let requirements = unsafe { ctx.get_buffer_memory_requirements(buffer) };

        let allocation_info = AllocationCreateDesc {
            name,
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
        };

        let allocation = allocator.allocate(&allocation_info).unwrap();

        unsafe {
            ctx.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).ok()?;
        };

        Some(Buffer {
            buffer,
            allocation: allocation,
        })
    }
}

impl VulkanResource for Buffer {
    type Subresource = ();

    fn destroy(self, ctx: &VulkanContext, allocator: &mut Allocator) {
        unsafe { ctx.destroy_buffer(self.buffer, None) };
        allocator.free(self.allocation).unwrap();
    }
}
