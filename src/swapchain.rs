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

pub struct SwapchainImage {
    pub index: u32,
    pub image: vk::Image,
}

#[allow(clippy::struct_field_names)]
pub struct Swapchain {
    surface_instance: surface::Instance,
    swapchain_device: swapchain::Device,

    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
}

// TODO: Part of DoubleBuffer?
impl Swapchain {
    pub fn new(
        surface_instance: surface::Instance,
        swapchain_device: swapchain::Device,
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
            surface_instance,
            swapchain_device,
            surface_format,
            extent,
            surface,
            swapchain,
            images,
        }
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
            self.surface_instance.destroy_surface(self.surface, None);
        }
    }

    pub fn recreate_swapchain(
        &mut self,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
    ) {
        unsafe {
            device.device_wait_idle().unwrap();

            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);

            let surface_capabilities = self
                .surface_instance
                .get_physical_device_surface_capabilities(physical_device, self.surface)
                .unwrap();

            self.extent = vk::Extent2D::default().width(width).height(height);

            let create_info = Self::swapchain_create_info(
                self.surface,
                &self.surface_format,
                &surface_capabilities,
                self.extent,
            );

            self.swapchain = self
                .swapchain_device
                .create_swapchain(&create_info, None)
                .unwrap();

            self.images = self
                .swapchain_device
                .get_swapchain_images(self.swapchain)
                .unwrap();
        }
    }

    pub fn acquire_next_image(
        &self,
        semaphore: vk::Semaphore,
    ) -> Result<SwapchainImage, SwapchainError> {
        let index = match unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                1_000_000_000,
                semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((swapchain_image_index, is_suboptimal)) => {
                if is_suboptimal {
                    return Err(SwapchainError::Suboptimal);
                }
                swapchain_image_index
            }
            Err(error) => {
                if error == vk::Result::ERROR_OUT_OF_DATE_KHR {
                    return Err(SwapchainError::OutOfDate);
                }

                panic!();
            }
        };

        Ok(SwapchainImage {
            index,
            image: self.images[index as usize],
        })
    }

    pub fn queue_present(
        &self,
        swapchain_image_index: u32,
        graphics_queue: vk::Queue,
        semaphore: vk::Semaphore,
    ) -> Result<(), SwapchainError> {
        let swapchains = [self.swapchain];
        let wait_semaphores = [semaphore];
        let image_indices = [swapchain_image_index];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&wait_semaphores)
            .image_indices(&image_indices);
        match unsafe {
            self.swapchain_device
                .queue_present(graphics_queue, &present_info)
        } {
            Ok(is_suboptimal) => {
                if is_suboptimal {
                    return Err(SwapchainError::Suboptimal);
                }

                Ok(())
            }
            Err(error) => {
                if error == vk::Result::ERROR_OUT_OF_DATE_KHR {
                    return Err(SwapchainError::OutOfDate);
                }
                panic!();
            }
        }
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
