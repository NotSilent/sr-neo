use std::{
    borrow::Cow,
    ffi::{self, c_char},
};

use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

mod vk_util {
    use ash::{Device, vk};

    pub fn create_fence(device: &Device, flags: vk::FenceCreateFlags) -> vk::Fence {
        let create_info = vk::FenceCreateInfo::default().flags(flags);

        unsafe {
            device
                .create_fence(&create_info, None)
                .expect("Failed to create Fence")
        }
    }

    pub fn create_semaphore(device: &Device) -> vk::Semaphore {
        let create_info = vk::SemaphoreCreateInfo::default();

        unsafe {
            device
                .create_semaphore(&create_info, None)
                .expect("Failed to create Semaphore")
        }
    }

    pub fn transition_image(
        device: &Device,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        current_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        // TODO: ALL_COMMANDS inefficent, make stage masks more accurate
        // https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
        let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .old_layout(current_layout)
            .new_layout(new_layout)
            .subresource_range(image_subresource_range(aspect_mask))
            .image(image);

        let binding = [image_barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&binding);

        unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
    }

    pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .base_array_layer(0)
            .layer_count(vk::REMAINING_ARRAY_LAYERS)
    }
}

#[derive(Default)]
struct FrameData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    swapchain_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

// TODO: runtime?
const FRAME_OVERLAP: usize = 2;

pub struct VulkanEngine {
    frame_number: u64,
    _stop_rendering: bool,
    _window_extent: vk::Extent2D,

    _entry: Entry,
    instance: Instance,
    debug_utils: debug_utils::Instance,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    _physical_device: vk::PhysicalDevice,
    device: Device,
    _graphics_queue_family_index: u32,
    graphics_queue: vk::Queue,

    surface_loader: surface::Instance,
    swapchain_loader: swapchain::Device,

    surface: vk::SurfaceKHR,

    swapchain: vk::SwapchainKHR,
    _swapchain_image_format: vk::Format,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    _swapchain_extent: vk::Extent2D,

    frames: [FrameData; FRAME_OVERLAP],
}

impl VulkanEngine {
    const _USE_VALIDATION_LAYERS: bool = false;

    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = unsafe { *p_callback_data };
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
        };

        println!(
            "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );

        vk::FALSE
    }

    #[allow(clippy::too_many_lines)]
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Self {
        let entry = Entry::linked();
        let instance = Self::create_instance(&entry, display_handle);
        let debug_utils = debug_utils::Instance::new(&entry, &instance);
        let debug_utils_messenger = Self::create_debug_utils_messenger(&debug_utils);
        let physical_device = Self::select_physical_device(&instance);

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let graphics_queue_family_index =
            Self::get_graphics_queue_family_index(&queue_families).unwrap();
        let device = Self::create_device(physical_device, &instance, graphics_queue_family_index);
        let graphics_queue = unsafe {
            device.get_device_queue2(
                &vk::DeviceQueueInfo2::default()
                    .queue_family_index(graphics_queue_family_index)
                    .queue_index(0), // TODO: 0?
            )
        };

        let surface_loader = surface::Instance::new(&entry, &instance);
        let swapchain_loader = swapchain::Device::new(&instance, &device);

        let surface = Self::create_surface(&entry, &instance, display_handle, window_handle);
        let surface_format = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0]
        };

        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .unwrap();

        let render_area =
            vk::Rect2D::default().extent(vk::Extent2D::default().width(width).height(height));

        let swapchain = Self::create_swapchain(
            &swapchain_loader,
            surface,
            surface_format,
            surface_capabilities.current_transform,
            render_area,
            &[graphics_queue_family_index],
        );

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };
        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images
            .iter()
            .map(|&image| {
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(
                        vk::ComponentMapping::default()
                            .r(vk::ComponentSwizzle::R)
                            .g(vk::ComponentSwizzle::G)
                            .b(vk::ComponentSwizzle::B)
                            .a(vk::ComponentSwizzle::A),
                    )
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                unsafe {
                    device
                        .create_image_view(&image_view_create_info, None)
                        .unwrap()
                }
            })
            .collect();

        // TODO: Abstract
        let mut frames: [FrameData; FRAME_OVERLAP] = [FrameData::default(), FrameData::default()];
        frames[0].command_pool = Self::create_command_pool(&device, graphics_queue_family_index);
        frames[0].command_buffer = Self::allocate_command_buffer(&device, frames[0].command_pool);
        frames[0].fence = vk_util::create_fence(&device, vk::FenceCreateFlags::SIGNALED);
        frames[0].swapchain_semaphore = vk_util::create_semaphore(&device);
        frames[0].render_semaphore = vk_util::create_semaphore(&device);
        frames[1].command_pool = Self::create_command_pool(&device, graphics_queue_family_index);
        frames[1].command_buffer = Self::allocate_command_buffer(&device, frames[1].command_pool);
        frames[1].fence = vk_util::create_fence(&device, vk::FenceCreateFlags::SIGNALED);
        frames[1].swapchain_semaphore = vk_util::create_semaphore(&device);
        frames[1].render_semaphore = vk_util::create_semaphore(&device);

        Self {
            frame_number: 0,
            _stop_rendering: false,
            _window_extent: vk::Extent2D {
                width: 1920,
                height: 1080,
            },

            _entry: entry,
            instance,
            debug_utils,
            debug_utils_messenger,
            _physical_device: physical_device,
            device,
            _graphics_queue_family_index: graphics_queue_family_index,
            graphics_queue,

            surface_loader,
            swapchain_loader,

            surface,
            swapchain,
            _swapchain_image_format: surface_format.format,
            swapchain_images,
            swapchain_image_views,
            _swapchain_extent: render_area.extent,
            frames,
        }
    }

    pub fn cleanup(&mut self) {
        unsafe { self.device.queue_wait_idle(self.graphics_queue).unwrap() };

        for frame in &self.frames {
            unsafe {
                self.device.destroy_command_pool(frame.command_pool, None);
                self.device.destroy_fence(frame.fence, None);
                self.device.destroy_semaphore(frame.render_semaphore, None);
                self.device
                    .destroy_semaphore(frame.swapchain_semaphore, None);
            }
        }

        for &image_view in &self.swapchain_image_views {
            unsafe { self.device.destroy_image_view(image_view, None) };
        }

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        };
    }

    pub fn draw(&mut self) {
        let current_frame = self.get_current_frame();
        unsafe {
            self.device
                .wait_for_fences(&[current_frame.fence], true, 1_000_000_000)
                .expect("Failed waiting for fences");

            self.device
                .reset_fences(&[current_frame.fence])
                .expect("Failed to reset fences");

            let (swapchain_image_index, _) = self
                .swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    1_000_000_000,
                    current_frame.swapchain_semaphore,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image");

            let cmd = current_frame.command_buffer;
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");

            let cmd_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            // Begin command buffer

            self.device
                .begin_command_buffer(cmd, &cmd_begin_info)
                .expect("Failed to begin command buffer");

            let swapchain_image = self.swapchain_images[swapchain_image_index as usize];

            vk_util::transition_image(
                &self.device,
                cmd,
                swapchain_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            let flash = f32::abs(f32::sin(self.frame_number as f32 / 120.0));
            let clear_value = vk::ClearColorValue {
                float32: [0.0, 0.0, flash, 1.0],
            };

            let clear_range = vk_util::image_subresource_range(vk::ImageAspectFlags::COLOR);
            self.device.cmd_clear_color_image(
                cmd,
                swapchain_image,
                vk::ImageLayout::GENERAL,
                &clear_value,
                &[clear_range],
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                swapchain_image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            self.device
                .end_command_buffer(cmd)
                .expect("Failed to end command buffer");

            // End command buffer

            let cmd_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(cmd)
                .device_mask(0)];

            let wait_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(current_frame.swapchain_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .device_index(0)
                .value(1)];

            let signal_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(current_frame.render_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                .device_index(0)
                .value(1)];

            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos)
                .command_buffer_infos(&cmd_infos);

            self.device
                .queue_submit2(self.graphics_queue, &[submit_info], current_frame.fence)
                .expect("Failed to queue submit");

            let swapchains = [self.swapchain];
            let wait_semaphores = [current_frame.render_semaphore];
            let image_indices = [swapchain_image_index];

            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchains)
                .wait_semaphores(&wait_semaphores)
                .image_indices(&image_indices);

            self.swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
                .expect("Failed to present queue");

            self.frame_number += 1;
        };
    }

    fn create_instance(entry: &Entry, display_handle: RawDisplayHandle) -> Instance {
        unsafe {
            let appinfo = vk::ApplicationInfo::default()
                .application_name(c"FutureAppName")
                .application_version(0)
                .engine_name(c"srneo")
                .engine_version(0)
                .api_version(vk::API_VERSION_1_3);

            let layer_names = [c"VK_LAYER_KHRONOS_validation"];

            let layers_names_raw: Vec<*const c_char> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let mut extension_names = ash_window::enumerate_required_extensions(display_handle)
                .unwrap()
                .to_vec();
            extension_names.push(debug_utils::NAME.as_ptr());

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names)
                .flags(vk::InstanceCreateFlags::default());

            entry
                .create_instance(&create_info, None)
                .expect("Instance creation failed")
        }
    }

    fn create_debug_utils_messenger(
        debug_utils: &debug_utils::Instance,
    ) -> vk::DebugUtilsMessengerEXT {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_callback));

        unsafe {
            debug_utils
                .create_debug_utils_messenger(&debug_info, None)
                .expect("Couldn't create debug utils messenger")
        }
    }

    fn select_physical_device(instance: &Instance) -> vk::PhysicalDevice {
        // TODO: Proper device selection
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };
        *physical_devices.first().unwrap()
    }

    fn create_device(
        physical_device: vk::PhysicalDevice,
        instance: &Instance,
        graphics_queue_family_index: u32,
    ) -> Device {
        let device_queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_family_index)
            .queue_priorities(&[1.0_f32]);

        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true);

        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let device_extension_names_raw = [ash::khr::swapchain::NAME.as_ptr()];

        let var_name = [device_queue_create_info];
        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features)
            .queue_create_infos(&var_name)
            .enabled_extension_names(&device_extension_names_raw);

        unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        }
    }

    fn get_graphics_queue_family_index(
        queue_families: &[vk::QueueFamilyProperties],
    ) -> Option<u32> {
        for (i, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                return Some(i as u32);
            }
        }

        None
    }

    fn create_surface(
        entry: &Entry,
        instance: &Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> vk::SurfaceKHR {
        unsafe {
            ash_window::create_surface(entry, instance, display_handle, window_handle, None)
                .unwrap()
        }
    }

    fn create_swapchain(
        swapchain: &swapchain::Device,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        surface_transform: vk::SurfaceTransformFlagsKHR,
        render_area: vk::Rect2D,
        queue_family_indices: &[u32],
    ) -> vk::SwapchainKHR {
        // TODO: based on capabilities
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(2)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(render_area.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(queue_family_indices)
            .pre_transform(surface_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        unsafe { swapchain.create_swapchain(&create_info, None).unwrap() }
    }

    fn get_current_frame(&self) -> &FrameData {
        self.frames
            .get(self.frame_number as usize % FRAME_OVERLAP)
            .unwrap()
    }

    fn create_command_pool(device: &Device, graphics_queue_family_index: u32) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        }
    }

    fn allocate_command_buffer(
        device: &Device,
        command_pool: vk::CommandPool,
    ) -> vk::CommandBuffer {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate command buffer")[0] // TODO: Safe
        }
    }
}
