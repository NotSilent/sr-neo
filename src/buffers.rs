use crate::resource_manager::{ResourceManager, VulkanResource};

use ash::{Device, vk};
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
    pub allocation: Option<Allocation>, // TODO: Drop Option<> somehow (maybe put in sparse vec and index for deletion?)
}

impl Buffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        alloc_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        name: &str,
    ) -> Buffer {
        let create_info = vk::BufferCreateInfo::default()
            .size(alloc_size)
            .usage(usage);

        let buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_info = AllocationCreateDesc {
            name,
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
        };

        let allocation = allocator.allocate(&allocation_info).unwrap();

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        };

        Buffer {
            buffer,
            allocation: Some(allocation),
        }
    }
}

impl VulkanResource for Buffer {
    type Subresource = ();

    fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe { device.destroy_buffer(self.buffer, None) };
        allocator.free(self.allocation.take().unwrap()).unwrap();
    }
}
