use std::{
    borrow::Cow,
    ffi::{self, c_char},
};

use ash::{Device, Entry, Instance, ext::debug_utils, vk};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::vulkan_engine::{VulkanContext, VulkanResult};

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        return vk::FALSE;
    }

    let callback_data = unsafe { *p_callback_data };

    let cmd_labels = unsafe {
        if callback_data.cmd_buf_label_count > 0 {
            std::slice::from_raw_parts(
                callback_data.p_cmd_buf_labels,
                callback_data.cmd_buf_label_count as usize,
            )
        } else {
            &[]
        }
    };

    let test_cmd_label = if cmd_labels.is_empty() {
        c""
    } else {
        unsafe { cmd_labels[0].label_name_as_c_str().unwrap() }
    }
    .to_string_lossy();

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
        "{test_cmd_label}\n{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}

pub fn create_instance(
    entry: &Entry,
    display_handle: RawDisplayHandle,
) -> VulkanResult<ash::Instance> {
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

        entry.create_instance(&create_info, None)
    }
}

pub fn create_debug_utils_messenger(
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
        .pfn_user_callback(Some(vulkan_debug_callback));

    unsafe {
        debug_utils
            .create_debug_utils_messenger(&debug_info, None)
            .expect("Couldn't create debug utils messenger")
    }
}

pub fn select_physical_device(instance: &Instance) -> vk::PhysicalDevice {
    // TODO: Proper device selection
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
    };
    *physical_devices.first().unwrap()
}

pub fn get_graphics_queue_family_index(
    queue_families: &[vk::QueueFamilyProperties],
) -> Option<u32> {
    for (i, queue_family) in queue_families.iter().enumerate() {
        if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            return Some(i as u32);
        }
    }

    None
}

pub fn create_device(
    physical_device: vk::PhysicalDevice,
    instance: &Instance,
    graphics_queue_family_index: u32,
) -> Device {
    let device_queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(graphics_queue_family_index)
        .queue_priorities(&[1.0_f32]);

    let physical_device_features = vk::PhysicalDeviceFeatures::default().multi_draw_indirect(true);

    let mut vulkan_11_features =
        vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);

    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .host_query_reset(true)
        .descriptor_binding_storage_buffer_update_after_bind(true)
        .descriptor_binding_update_unused_while_pending(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_variable_descriptor_count(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .runtime_descriptor_array(true);

    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true)
        .shader_demote_to_helper_invocation(true);

    let device_extension_names_raw = [ash::khr::swapchain::NAME.as_ptr()];

    let var_name = [device_queue_create_info];
    let device_create_info = vk::DeviceCreateInfo::default()
        .push_next(&mut vulkan_11_features)
        .push_next(&mut vulkan_12_features)
        .push_next(&mut vulkan_13_features)
        .queue_create_infos(&var_name)
        .enabled_extension_names(&device_extension_names_raw)
        .enabled_features(&physical_device_features);

    unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .unwrap()
    }
}

pub fn create_surface(
    ctx: &VulkanContext,
    display_handle: RawDisplayHandle,
    window_handle: RawWindowHandle,
) -> Result<vk::SurfaceKHR, vk::Result> {
    unsafe {
        ash_window::create_surface(
            &ctx.entry,
            &ctx.instance,
            display_handle,
            window_handle,
            None,
        )
    }
}

pub fn create_allocator(ctx: &VulkanContext) -> Allocator {
    Allocator::new(&AllocatorCreateDesc {
        instance: ctx.instance.clone(),
        device: ctx.device.clone(),
        physical_device: ctx.physical_device,
        debug_settings: AllocatorDebugSettings {
            log_memory_information: false,
            log_leaks_on_shutdown: true,
            store_stack_traces: false,
            log_allocations: false,
            log_frees: false,
            log_stack_traces: false,
        },
        buffer_device_address: true,
        allocation_sizes: AllocationSizes::default(),
    })
    .unwrap()
}
