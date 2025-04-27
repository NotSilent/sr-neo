use ash::Device;
use ash::vk;

pub enum DeletionType {
    DescriptorSetLayout(vk::DescriptorSetLayout),
    Image(vk::Image),
    ImageView(vk::ImageView),
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

    pub fn flush(&mut self, device: &Device) {
        for item in self.queue.iter().rev() {
            match item {
                DeletionType::DescriptorSetLayout(descriptor_set_layout) => unsafe {
                    device.destroy_descriptor_set_layout(*descriptor_set_layout, None);
                },
                DeletionType::Image(image) => unsafe { device.destroy_image(*image, None) },
                DeletionType::ImageView(image_view) => unsafe {
                    device.destroy_image_view(*image_view, None)
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
