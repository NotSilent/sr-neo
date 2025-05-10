use ash::{
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

#[allow(clippy::struct_field_names)]
pub struct Swapchain {
    pub _format: vk::Format,
    pub extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
}

impl Swapchain {
    pub fn new(
        surface_instance: &surface::Instance,
        swapchain_device: &swapchain::Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        extent: vk::Extent2D,
    ) -> Self {
        let surface_format = unsafe {
            surface_instance
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0]
        };

        let surface_capabilities = unsafe {
            surface_instance.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .unwrap();

        let create_info =
            Self::swapchain_create_info(surface, &surface_format, &surface_capabilities, extent);

        let swapchain = unsafe {
            swapchain_device
                .create_swapchain(&create_info, None)
                .unwrap()
        };

        let images = unsafe { swapchain_device.get_swapchain_images(swapchain).unwrap() };

        Self {
            _format: surface_format.format,
            extent,
            swapchain,
            images,
        }
    }

    pub fn destroy(&mut self, swapchain_device: &swapchain::Device) {
        unsafe { swapchain_device.destroy_swapchain(self.swapchain, None) };
    }

    fn swapchain_create_info<'a>(
        surface: vk::SurfaceKHR,
        surface_format: &'a vk::SurfaceFormatKHR,
        surface_capabilities: &'a vk::SurfaceCapabilitiesKHR,
        extent: vk::Extent2D,
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
            // Not needed for exclusive
            // .queue_family_indices(queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null())
    }
}
