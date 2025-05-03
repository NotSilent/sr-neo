use ash::{
    Device,
    khr::{surface, swapchain},
    vk,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SwapchainError {
    #[error("Swapchain suboptimal")]
    Suboptimal,
    #[error("Swapchain out of date")]
    OutOfDate,
}

use crate::vk_util;

pub struct Swapchain {
    format: vk::Format,
    extent: vk::Extent2D,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub fn new(
        surface_instance: &surface::Instance,
        swapchain_device: &swapchain::Device,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        extent: vk::Extent2D,
        queue_family_indices: &[u32],
    ) -> Self {
        let format = vk::Format::B8G8R8A8_UNORM;

        let surface_format = unsafe {
            surface_instance
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0]
        };

        let surface_capabilities = unsafe {
            surface_instance.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .unwrap();

        let create_info = Self::swapchain_create_info(
            surface,
            &surface_format,
            &surface_capabilities,
            extent,
            queue_family_indices,
        );

        let swapchain = unsafe {
            swapchain_device
                .create_swapchain(&create_info, None)
                .unwrap()
        };

        let images = unsafe { swapchain_device.get_swapchain_images(swapchain).unwrap() };
        let image_views =
            Self::create_swapchain_image_views(device, surface_format.format, &images);

        Self {
            format,
            extent,
            handle: swapchain,
            images,
            image_views,
        }
    }

    pub fn destroy(&mut self, swapchain_device: &swapchain::Device, device: &Device) {
        for &image_view in &self.image_views {
            unsafe { device.destroy_image_view(image_view, None) };
        }

        self.image_views.clear();

        unsafe { swapchain_device.destroy_swapchain(self.handle, None) };
    }

    fn swapchain_create_info<'a>(
        surface: vk::SurfaceKHR,
        surface_format: &'a vk::SurfaceFormatKHR,
        surface_capabilities: &'a vk::SurfaceCapabilitiesKHR,
        extent: vk::Extent2D,
        queue_family_indices: &'a [u32],
    ) -> vk::SwapchainCreateInfoKHR<'a> {
        vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(3)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null())
    }

    fn create_swapchain_image_views(
        device: &Device,
        format: vk::Format,
        images: &[vk::Image],
    ) -> Vec<vk::ImageView> {
        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let image_view_create_info =
                    vk_util::image_view_create_info(format, image, vk::ImageAspectFlags::COLOR);

                unsafe {
                    device
                        .create_image_view(&image_view_create_info, None)
                        .unwrap()
                }
            })
            .collect();
        image_views
    }
}

// Getters
impl Swapchain {
    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn _image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }
}
