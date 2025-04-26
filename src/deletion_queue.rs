use ash::Device;
use ash::vk;

pub enum DeletionType {
    Image(vk::Image),
    ImageView(vk::ImageView),
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
                DeletionType::Image(image) => unsafe { device.destroy_image(*image, None) },
                DeletionType::ImageView(image_view) => unsafe {
                    device.destroy_image_view(*image_view, None)
                },
            }
        }

        self.queue.clear();
    }
}
