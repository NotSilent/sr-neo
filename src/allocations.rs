use ash::{Device, vk};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

pub struct AllocatedBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>, // TODO: Drop Option<> somehow (maybe put in sparse vec and index for deletion?)
}

impl AllocatedBuffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        alloc_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
    ) -> AllocatedBuffer {
        let create_info = vk::BufferCreateInfo::default()
            .size(alloc_size)
            .usage(usage);

        let buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_info = AllocationCreateDesc {
            name: "buffer", // TODO: Proper name
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator.allocate(&allocation_info).unwrap();

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        };

        AllocatedBuffer {
            buffer,
            allocation: Some(allocation),
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        let _ = allocator.free(self.allocation.take().unwrap());
        unsafe { device.destroy_buffer(self.buffer, None) };
    }
}

// Getters
impl AllocatedBuffer {
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn allocation(&self) -> Option<&Allocation> {
        self.allocation.as_ref()
    }
}
