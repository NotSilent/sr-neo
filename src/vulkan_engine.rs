use std::{f32, mem::ManuallyDrop};

use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};

use gpu_allocator::{MemoryLocation, vulkan::Allocator};

use nalgebra::{Matrix4, Vector2, Vector3, Vector4, vector};
use thiserror::Error;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{
    buffers::{Buffer, BufferIndex, BufferManager},
    camera::{Camera, InputManager},
    default_resources,
    descriptors::{DescriptorAllocatorGrowable, DescriptorLayoutBuilder, PoolSizeRatio},
    double_buffer::{self, DoubleBuffer},
    draw::{DrawCommands, DrawContext, DrawRecord},
    forward_pass::{self},
    geometry_pass,
    gltf_loader::GLTFLoader,
    images::{ImageIndex, ImageManager},
    immediate_submit::ImmediateSubmit,
    lightning_pass,
    materials::{
        MasterMaterial, MasterMaterialIndex, MasterMaterialManager, MaterialConstants,
        MaterialInstanceIndex, MaterialInstanceManager, MaterialPass, MaterialResources,
    },
    meshes::MeshManager,
    renderpass_common::RenderpassImageState,
    shader_manager::ShaderManager,
    shadow_map_pass,
    swapchain::{Swapchain, SwapchainError},
    vk_init, vk_util,
};

#[derive(Default)]
pub struct GPUStats {
    pub draw_time: f64,
    pub draw_calls: u32,
    pub triangles: usize,
}

pub struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
    pub material_instance_index: MaterialInstanceIndex,
}

#[derive(Default)]
#[repr(C)]
pub struct SceneData {
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    inv_proj: Matrix4<f32>,
    view_proj: Matrix4<f32>,
    light_view: Matrix4<f32>,
    light_view_proj: Matrix4<f32>,
    sunlight_direction: Vector4<f32>, // w for sun power
    sunlight_color: Vector4<f32>,
    view_position: Vector3<f32>,
    _padding: f32,
    screen_size: Vector2<f32>,
}

impl SceneData {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Self>(self).cast::<u8>(),
                size_of::<Self>(),
            )
        }
    }
}

#[derive(Error, Debug)]
pub enum DrawError {
    #[error("{}", .0)]
    Swapchain(SwapchainError),
}

impl From<SwapchainError> for DrawError {
    fn from(value: SwapchainError) -> Self {
        DrawError::Swapchain(value)
    }
}

// TODO: Formats
#[derive(Default, Clone)]
#[repr(C)]
pub struct Vertex {
    pub position: Vector3<f32>,
    pub uv_x: f32,
    pub normal: Vector3<f32>,
    pub uv_y: f32,
    pub color: Vector4<f32>,
    pub tangent: Vector4<f32>,
}

// TODO: Probably can be part of Mesh
pub struct GPUMeshBuffers {
    pub index_buffer_index: BufferIndex,
    pub vertex_buffer_index: BufferIndex,
    pub vertex_buffer_address: vk::DeviceAddress,
}

pub struct ManagedResources {
    pub buffers: BufferManager,
    pub images: ImageManager,

    pub meshes: MeshManager,
    pub master_materials: MasterMaterialManager,
    pub material_instances: MaterialInstanceManager,
}

impl ManagedResources {
    // TODO: encapsulate access and properly manage removal
}

pub struct DefaultResources {
    pub image_white: ImageIndex,
    pub image_black: ImageIndex,
    pub image_error: ImageIndex,

    pub image_normal: ImageIndex,

    pub sampler_nearest: vk::Sampler,
    pub sampler_linear: vk::Sampler,

    pub geometry_pass_material: MasterMaterialIndex,
    pub shadow_map_pass_material: MasterMaterialIndex,
    pub transparent_material: MasterMaterialIndex,

    pub geometry_pass_material_instance: MaterialInstanceIndex,
}

pub struct VulkanEngine {
    frame_number: u64,

    _entry: Entry,
    instance: Instance,
    debug_utils: debug_utils::Instance,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    device: Device,
    _graphics_queue_family_index: u32,
    graphics_queue: vk::Queue,

    swapchain: Swapchain,

    double_buffer: DoubleBuffer,

    allocator: ManuallyDrop<Allocator>,

    shader_manager: ShaderManager,

    // TODO: PRIORITY: Leaking
    // Can never be cleared
    descriptor_allocator: DescriptorAllocatorGrowable,

    immediate_submit: ImmediateSubmit,

    gpu_scene_data_descriptor_layout: vk::DescriptorSetLayout,

    managed_resources: ManagedResources,
    default_resources: DefaultResources,

    main_draw_context: DrawContext,

    main_camera: Camera, // TODO: Shouldn't be part of renderer

    gltf_loader: GLTFLoader,
}

impl VulkanEngine {
    const _USE_VALIDATION_LAYERS: bool = false;

    const DIRECTIONAL_LIGHT_CLIP_LENGTH: f32 = 50.0;

    #[allow(clippy::too_many_lines)]
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        width: u32,
        height: u32,
        gltf_name: &str,
    ) -> Self {
        let entry = Entry::linked();
        let instance = vk_init::create_instance(&entry, display_handle);
        let debug_utils = debug_utils::Instance::new(&entry, &instance);
        let debug_utils_messenger = vk_init::create_debug_utils_messenger(&debug_utils);
        let physical_device = vk_init::select_physical_device(&instance);

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let graphics_queue_family_index =
            vk_init::get_graphics_queue_family_index(&queue_families).unwrap();
        let device =
            vk_init::create_device(physical_device, &instance, graphics_queue_family_index);
        let graphics_queue = unsafe {
            device.get_device_queue2(
                &vk::DeviceQueueInfo2::default()
                    .queue_family_index(graphics_queue_family_index)
                    .queue_index(0), // TODO: 0?
            )
        };

        let surface_instance = surface::Instance::new(&entry, &instance);
        let swapchain_device = swapchain::Device::new(&instance, &device);

        let surface = vk_init::create_surface(&entry, &instance, display_handle, window_handle);

        let swapchain = Swapchain::new(
            surface_instance,
            swapchain_device,
            physical_device,
            &device,
            surface,
            vk::Extent2D::default().width(width).height(height),
        );

        let mut allocator =
            vk_init::create_allocator(instance.clone(), device.clone(), physical_device);

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        // init descriptors
        let pool_ratios = vec![
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                ratio: 1,
            },
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                ratio: 1,
            },
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                ratio: 1,
            },
        ];

        let mut descriptor_allocator = DescriptorAllocatorGrowable::new(&device, 10, pool_ratios);

        let sampler_nearest_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        let sampler_linear_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);

        let default_sampler_nearest = unsafe {
            device
                .create_sampler(&sampler_nearest_create_info, None)
                .unwrap()
        };

        let default_sampler_linear = unsafe {
            device
                .create_sampler(&sampler_linear_create_info, None)
                .unwrap()
        };

        let mut image_manager = ImageManager::new();

        let mut shader_manager = ShaderManager::default();

        let gpu_scene_data_descriptor_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .build(
                &device,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            );

        let mut immediate_submit =
            ImmediateSubmit::new(&device, graphics_queue, graphics_queue_family_index);

        let mut buffer_manager = BufferManager::new();

        let mesh_manager = MeshManager::new();

        let image_white = default_resources::image_white(
            &device,
            &mut allocator,
            &immediate_submit,
            vk::AccessFlags2::SHADER_READ,
        );

        let image_black = default_resources::image_black(
            &device,
            &mut allocator,
            &immediate_submit,
            vk::AccessFlags2::SHADER_READ,
        );

        let image_error = default_resources::image_error(
            &device,
            &mut allocator,
            &immediate_submit,
            vk::AccessFlags2::SHADER_READ,
        );

        let image_normal = default_resources::image_normal(
            &device,
            &mut allocator,
            &immediate_submit,
            vk::AccessFlags2::SHADER_READ,
        );

        let mut master_material_manager = MasterMaterialManager::new();

        let double_buffer = DoubleBuffer::new(
            &device,
            &mut allocator,
            graphics_queue_family_index,
            width,
            height,
            properties.limits.timestamp_period,
            gpu_scene_data_descriptor_layout,
            &mut descriptor_allocator,
            default_sampler_linear,
            gpu_scene_data_descriptor_layout,
            &mut shader_manager,
        );

        let geometry_pass_shader =
            shader_manager.get_graphics_shader_combined(&device, "geometry_pass");
        let shadow_map_pass_shader =
            shader_manager.get_graphics_shader_combined(&device, "shadow_map_pass");
        let shader = shader_manager.get_graphics_shader_combined(&device, "forward_pass");

        let geometry_pass_material = MasterMaterial::new(
            &device,
            gpu_scene_data_descriptor_layout,
            &[double_buffer::COLOR_FORMAT, double_buffer::NORMAL_FORMAT],
            double_buffer::DEPTH_FORMAT,
            MaterialPass::Opaque,
            &geometry_pass_shader,
        );

        let shadow_map_pass_material = MasterMaterial::new_shadow_map(
            &device,
            gpu_scene_data_descriptor_layout,
            double_buffer::SHADOW_MAP_FORMAT,
            MaterialPass::Opaque,
            &shadow_map_pass_shader,
        );

        let default_transparent_material = MasterMaterial::new(
            &device,
            gpu_scene_data_descriptor_layout,
            &[double_buffer::DRAW_FORMAT],
            double_buffer::DEPTH_FORMAT,
            MaterialPass::Transparent,
            &shader,
        );

        let geometry_pass_material_index = master_material_manager.add(geometry_pass_material);
        let shadow_map_pass_material_index = master_material_manager.add(shadow_map_pass_material);
        let default_transparent_material_index =
            master_material_manager.add(default_transparent_material);

        let mut material_instance_manager = MaterialInstanceManager::new();

        let material_constants_buffer = Buffer::new(
            &device,
            &mut allocator,
            size_of::<MaterialConstants>(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "default_data_constants_buffer",
        );

        let material_constants = MaterialConstants {
            color_factors: vector![1.0, 1.0, 1.0, 1.0],
            metal_rough_factors: vector![1.0, 0.5, 0.0, 0.0],
        };

        vk_util::copy_data_to_allocation(
            &[material_constants],
            material_constants_buffer.allocation.as_ref().unwrap(),
        );

        let resources = MaterialResources {
            color_image_view: image_white.image_view,
            color_sampler: default_sampler_linear,
            normal_image_view: image_white.image_view,
            normal_sampler: default_sampler_linear,
            metal_rough_image_view: image_white.image_view,
            metal_rough_sampler: default_sampler_linear,
            data_buffer: material_constants_buffer.buffer,
            data_buffer_offset: 0,
        };

        buffer_manager.add(material_constants_buffer);

        let geometry_pass_material = master_material_manager.get_mut(geometry_pass_material_index);

        let geometry_pass_material_instance = geometry_pass_material.create_instance(
            &device,
            &resources,
            &mut descriptor_allocator,
            geometry_pass_material_index,
        );

        let geometry_pass_material_instance_index =
            material_instance_manager.add(geometry_pass_material_instance);

        let main_camera = Camera {
            position: vector![-6.0, 4.0, 0.0],
            velocity: Vector3::from_element(0.0),
            pitch: f32::consts::PI / -10.0,
            yaw: f32::consts::FRAC_PI_2,
        };

        let image_white_index = image_manager.add(image_white);
        let image_black_index = image_manager.add(image_black);
        let image_error_index = image_manager.add(image_error);
        let image_normal_index = image_manager.add(image_normal);

        let mut managed_resources = ManagedResources {
            buffers: buffer_manager,
            images: image_manager,
            meshes: mesh_manager,
            master_materials: master_material_manager,
            material_instances: material_instance_manager,
        };

        let default_resources = DefaultResources {
            image_white: image_white_index,
            image_black: image_black_index,
            image_error: image_error_index,
            image_normal: image_normal_index,
            sampler_nearest: default_sampler_nearest,
            sampler_linear: default_sampler_linear,
            geometry_pass_material: geometry_pass_material_index,
            transparent_material: default_transparent_material_index,
            geometry_pass_material_instance: geometry_pass_material_instance_index,
            shadow_map_pass_material: shadow_map_pass_material_index,
        };

        let gltf_loader = GLTFLoader::new(
            &device,
            std::path::PathBuf::from(gltf_name).as_path(),
            &mut allocator,
            &mut descriptor_allocator,
            &mut managed_resources,
            &default_resources,
            &mut immediate_submit,
        );

        Self {
            frame_number: 0,

            _entry: entry,
            instance,
            debug_utils,
            debug_utils_messenger,
            physical_device,
            device,
            _graphics_queue_family_index: graphics_queue_family_index,
            graphics_queue,

            swapchain,
            double_buffer,
            allocator: ManuallyDrop::new(allocator),

            shader_manager,

            descriptor_allocator,

            immediate_submit,

            gpu_scene_data_descriptor_layout,

            managed_resources,
            default_resources,

            main_draw_context: DrawContext::default(),

            main_camera,

            gltf_loader,
        }
    }

    // TODO: Shouldn't be part of renderer
    pub fn update(&mut self, input_manager: &InputManager) {
        self.main_camera.process_winit_events(input_manager);
    }

    #[allow(clippy::too_many_lines)]
    pub fn draw(&mut self, render_scale: f32) -> Result<GPUStats, DrawError> {
        let mut gpu_stats = GPUStats::default();

        let render_scale = render_scale.clamp(0.25, 1.0);

        let width = self.swapchain.extent.width as f32;
        let height = self.swapchain.extent.height as f32;

        self.update_scene();

        unsafe {
            let query_results = self
                .double_buffer
                .swap_buffer(&self.device, &mut self.allocator);

            gpu_stats.draw_time = query_results.draw_time;

            let synchronization_resources = self.double_buffer.get_synchronization_resources();
            let query_pool = self.double_buffer.get_query_pool();
            let cmd = self.double_buffer.get_command_buffer();
            let globals_descriptor_set = self.double_buffer.get_globals_descriptor_set();

            // TODO: encapsulate, into swapchain?
            let acquired_swapchain = self
                .swapchain
                .acquire_next_image(synchronization_resources.swapchain_semaphore)?;

            let frame_targets = self.double_buffer.get_frame_targets();
            let lightning_pass_description = self.double_buffer.get_lightning_pass_description();

            let draw_src = RenderpassImageState::new(&frame_targets.draw);
            let color_src = RenderpassImageState::new(&frame_targets.color);
            let normal_src = RenderpassImageState::new(&frame_targets.normal);
            let depth_src = RenderpassImageState::new(&frame_targets.depth);
            let shadow_map_src = RenderpassImageState::new(&frame_targets.shadow_map);

            let draw_width = self
                .swapchain
                .extent
                .width
                .min(frame_targets.draw.extent.width) as f32
                * render_scale;

            let draw_height = self
                .swapchain
                .extent
                .height
                .min(frame_targets.draw.extent.height) as f32
                * render_scale;

            let render_area = vk::Rect2D::default().extent(
                vk::Extent2D::default()
                    .width(draw_width as u32)
                    .height(draw_height as u32),
            );

            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");

            let cmd_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            // Begin command buffer

            self.device
                .begin_command_buffer(cmd, &cmd_begin_info)
                .expect("Failed to begin command buffer");

            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool,
                0,
            );

            // TODO: Prepare data before draw
            // TODO: split and sort commands in one function?
            let DrawRecord {
                opaque_commands,
                transparent_commands,
            } = DrawRecord::record_draw_commands(&self.main_draw_context, &self.managed_resources);

            let opaque_commads = DrawCommands::from(opaque_commands);
            let transparent_commads = DrawCommands::from(transparent_commands);

            self.double_buffer.set_globals(&Self::create_scene_data(
                &self.main_camera,
                self.frame_number,
                width,
                height,
            ));

            let mut write_data = self.double_buffer.upload_buffers(&self.device, cmd);

            let geometry_pass_output = geometry_pass::record(
                &self.device,
                cmd,
                render_area,
                color_src,
                normal_src,
                depth_src,
                globals_descriptor_set,
                &opaque_commads,
                &mut write_data,
                &mut gpu_stats,
            );

            let shadow_pass_master_material = self
                .managed_resources
                .master_materials
                .get(self.default_resources.shadow_map_pass_material);

            let shadow_map_pass_output = shadow_map_pass::record(
                &self.device,
                cmd,
                shadow_map_src,
                shadow_pass_master_material,
                globals_descriptor_set,
                &opaque_commads,
                &mut write_data,
                &mut gpu_stats,
            );

            let lightning_pass_output = lightning_pass::record(
                &self.device,
                cmd,
                render_area,
                draw_src,
                geometry_pass_output.color,
                geometry_pass_output.normal,
                geometry_pass_output.depth,
                shadow_map_pass_output.shadow_map,
                globals_descriptor_set,
                &lightning_pass_description,
            );

            let draw_image_state = forward_pass::record(
                &self.device,
                cmd,
                render_area,
                lightning_pass_output.draw,
                lightning_pass_output.depth,
                globals_descriptor_set,
                &transparent_commads,
                &mut write_data,
                &mut gpu_stats,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                draw_image_state.image,
                draw_image_state.layout,
                draw_image_state.stage_mask,
                draw_image_state.access_mask,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageAspectFlags::COLOR,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                acquired_swapchain.image,
                vk::ImageLayout::UNDEFINED,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageAspectFlags::COLOR,
            );

            vk_util::blit_image(
                &self.device,
                cmd,
                draw_image_state.image,
                acquired_swapchain.image,
                render_area.extent,
                self.swapchain.extent,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                acquired_swapchain.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::ImageAspectFlags::COLOR,
            );

            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool,
                1,
            );

            self.device
                .end_command_buffer(cmd)
                .expect("Failed to end command buffer");

            // End command buffer

            let cmd_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(cmd)
                .device_mask(0)];

            let wait_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(synchronization_resources.swapchain_semaphore)
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .device_index(0)
                .value(1)];

            let signal_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(acquired_swapchain.semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .device_index(0)
                .value(1)];

            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos)
                .command_buffer_infos(&cmd_infos);

            self.device
                .queue_submit2(
                    self.graphics_queue,
                    &[submit_info],
                    synchronization_resources.fence,
                )
                .expect("Failed to queue submit");

            self.swapchain.queue_present(
                acquired_swapchain.index,
                self.graphics_queue,
                acquired_swapchain.semaphore,
            )?;

            self.frame_number += 1;
        };

        Ok(gpu_stats)
    }

    // TODO: Drop frame counter in favor of delta time and light entities
    fn create_scene_data(camera: &Camera, frame_number: u64, width: f32, height: f32) -> SceneData {
        let view_position = camera.get_position();

        let view = camera.get_view();
        let proj = Camera::get_projection(height / width);
        let inv_proj = proj.try_inverse().unwrap();
        let view_proj = proj * view;

        let sin_x = f32::sin(frame_number as f32 / 200.0);
        let cos_z = f32::cos(frame_number as f32 / 200.0);
        let sunlight_direction = vector![sin_x, 4.0, cos_z].normalize().insert_row(3, 1.0);

        let light_view = Matrix4::look_at_rh(
            &From::from(
                view_position
                    + (sunlight_direction.xyz() * Self::DIRECTIONAL_LIGHT_CLIP_LENGTH / 2.0),
            ),
            &From::from(
                view_position
                    + -sunlight_direction.xyz() * Self::DIRECTIONAL_LIGHT_CLIP_LENGTH / 2.0,
            ),
            &Vector3::y(),
        );

        let light_proj = Camera::get_orthographic(
            -25.0,
            25.0,
            -25.0,
            25.0,
            Self::DIRECTIONAL_LIGHT_CLIP_LENGTH,
            0.01,
        );

        let light_view_proj = light_proj * light_view;

        SceneData {
            view: camera.get_view(),
            proj,
            inv_proj,
            view_proj,
            light_view,
            light_view_proj,
            sunlight_direction,
            sunlight_color: Vector4::from_element(1.0),
            view_position,
            _padding: 0.0,
            screen_size: vector![width, height],
        }
    }

    fn update_scene(&mut self) {
        self.main_camera.update();

        self.main_draw_context.render_objects.clear();

        self.gltf_loader.draw(&mut self.main_draw_context);
    }

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        self.swapchain
            .recreate_swapchain(&self.device, self.physical_device, width, height);
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe { self.device.queue_wait_idle(self.graphics_queue).unwrap() };

        let device = &self.device;
        let allocator = &mut self.allocator;

        self.double_buffer.destroy(device, allocator);

        let subresources = self.managed_resources.meshes.destroy(device, allocator);

        for resouce in subresources {
            self.managed_resources
                .buffers
                .remove(device, allocator, resouce.index_buffer_index);
            self.managed_resources
                .buffers
                .remove(device, allocator, resouce.vertex_buffer_index);
        }

        self.managed_resources
            .images
            .remove(device, allocator, self.default_resources.image_white);
        self.managed_resources
            .images
            .remove(device, allocator, self.default_resources.image_black);
        self.managed_resources
            .images
            .remove(device, allocator, self.default_resources.image_error);

        // TODO: Shouldn't be needed if all resources are removed properly
        self.managed_resources.images.destroy(device, allocator);

        self.managed_resources.buffers.destroy(device, allocator);

        self.managed_resources
            .master_materials
            .destroy(device, allocator);
        // ~Shouldn't be needed if all resources are removed properly

        self.shader_manager.destroy(device);

        self.descriptor_allocator.destroy(device);

        self.immediate_submit.destroy(device);

        unsafe {
            device.destroy_descriptor_set_layout(self.gpu_scene_data_descriptor_layout, None);
        };

        unsafe {
            device.destroy_sampler(self.default_resources.sampler_linear, None);
            device.destroy_sampler(self.default_resources.sampler_nearest, None);
        }

        dbg!(allocator.generate_report());

        unsafe { ManuallyDrop::drop(allocator) };

        unsafe {
            self.swapchain.destroy(device);
            device.destroy_device(None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        };
    }
}
