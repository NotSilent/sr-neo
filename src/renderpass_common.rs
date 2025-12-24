use ash::vk;

use crate::{images::Image, vk_util};

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

    pub fn create_barrier(
        &self,
        target: &RenderpassImageState,
        aspect_mask: vk::ImageAspectFlags,
    ) -> vk::ImageMemoryBarrier2<'static> {
        vk_util::create_image_memory_barrier(
            self.image,
            self.layout,
            self.stage_mask,
            self.access_mask,
            target.layout,
            target.stage_mask,
            target.access_mask,
            aspect_mask,
        )
    }
}
