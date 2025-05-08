use crate::{images::Image, immediate_submit::ImmediateSubmit, vk_util};
use ash::{Device, vk};
use gpu_allocator::vulkan::Allocator;

pub fn image_white(
    device: &Device,
    allocator: &mut Allocator,
    immediate_submit: &ImmediateSubmit,
    access_mask: vk::AccessFlags2,
) -> Image {
    let white = vk_util::pack_u32(&[1.0, 1.0, 1.0, 1.0]);

    Image::new_with_data(
        device,
        allocator,
        immediate_submit,
        vk::Extent3D::default().width(1).height(1).depth(1),
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED,
        access_mask,
        false,
        &[white],
        "image_white",
    )
}

pub fn image_black(
    device: &Device,
    allocator: &mut Allocator,
    immediate_submit: &ImmediateSubmit,
    access_mask: vk::AccessFlags2,
) -> Image {
    let black = vk_util::pack_u32(&[0.0, 0.0, 0.0, 1.0]);

    Image::new_with_data(
        device,
        allocator,
        immediate_submit,
        vk::Extent3D::default().width(1).height(1).depth(1),
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED,
        access_mask,
        false,
        &[black],
        "image_black",
    )
}

pub fn image_error(
    device: &Device,
    allocator: &mut Allocator,
    immediate_submit: &ImmediateSubmit,
    access_mask: vk::AccessFlags2,
) -> Image {
    let magenta = vk_util::pack_u32(&[1.0, 0.0, 1.0, 1.0]);
    let black = vk_util::pack_u32(&[0.0, 0.0, 0.0, 1.0]);

    // TODO: 2x2
    let pixels = (0..16 * 16)
        .map(|i| {
            let x = i % 16;
            let y = i / 16;
            if (x % 2) ^ (y % 2) > 0 {
                magenta
            } else {
                black
            }
        })
        .collect::<Vec<u32>>();

    Image::new_with_data(
        device,
        allocator,
        immediate_submit,
        vk::Extent3D::default().width(16).height(16).depth(1),
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED,
        access_mask,
        false,
        &pixels,
        "image_error",
    )
}
