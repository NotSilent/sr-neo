use ash::vk;

use crate::images::Image;

pub struct RenderpassImageState {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub layout: vk::ImageLayout,
    pub stage_mask: vk::PipelineStageFlags2,
    pub access_mask: vk::AccessFlags2,
}

impl RenderpassImageState {
    pub fn new(image: &Image) -> Self {
        Self {
            image: image.image,
            image_view: image.image_view,
            layout: vk::ImageLayout::UNDEFINED,
            stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            access_mask: vk::AccessFlags2::NONE,
        }
    }
}
