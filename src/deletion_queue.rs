use ash::Device;
use ash::vk;

pub enum DeletionType {
    DescriptorSetLayout(vk::DescriptorSetLayout),
}

#[derive(Default)]
pub struct DeletionQueue {
    queue: Vec<DeletionType>,
}

impl DeletionQueue {
    pub fn push(&mut self, item: DeletionType) {
        self.queue.push(item);
    }

    pub fn flush(&mut self, device: &Device) {
        for item in self.queue.iter_mut().rev() {
            match item {
                DeletionType::DescriptorSetLayout(descriptor_set_layout) => unsafe {
                    device.destroy_descriptor_set_layout(*descriptor_set_layout, None);
                },
            }
        }

        self.queue.clear();
    }
}
