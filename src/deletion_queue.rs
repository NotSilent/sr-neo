use ash::Device;
use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::allocations::AllocatedBuffer;

pub enum DeletionType {
    AllocatedBuffer(AllocatedBuffer),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
}

#[derive(Default)]
pub struct DeletionQueue {
    queue: Vec<DeletionType>,
}

impl DeletionQueue {
    pub fn push(&mut self, item: DeletionType) {
        self.queue.push(item);
    }

    pub fn flush(&mut self, device: &Device, allocator: &mut Allocator) {
        for item in self.queue.iter_mut().rev() {
            match item {
                DeletionType::AllocatedBuffer(allocated_buffer) => {
                    allocated_buffer.destroy(device, allocator);
                }
                DeletionType::DescriptorSetLayout(descriptor_set_layout) => unsafe {
                    device.destroy_descriptor_set_layout(*descriptor_set_layout, None);
                },
                DeletionType::Pipeline(pipeline) => unsafe {
                    device.destroy_pipeline(*pipeline, None);
                },
                DeletionType::PipelineLayout(pipeline_layout) => unsafe {
                    device.destroy_pipeline_layout(*pipeline_layout, None);
                },
            }
        }

        self.queue.clear();
    }
}
