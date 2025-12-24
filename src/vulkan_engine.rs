use std::{f32, ffi::CStr, mem::ManuallyDrop, ops::Deref};

use ash::{
    Entry,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};

use gpu_allocator::{MemoryLocation, vulkan::Allocator};

use nalgebra::{Matrix4, Vector2, Vector3, Vector4, vector};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use thiserror::Error;

use crate::{
    buffers::{Buffer, BufferIndex, BufferManager},
    camera::Camera,
    default_resources, depth_pre_pass,
    descriptors::{DescriptorAllocatorGrowable, DescriptorLayoutBuilder, PoolSizeRatio},
    double_buffer::{self, DoubleBuffer},
    draw::{DrawContext, DrawRecord, IndexedIndirectData},
    forward_pass::{self},
    fxaa_pass, geometry_pass,
    images::{Image, ImageIndex, ImageManager},
    immediate_submit::ImmediateSubmit,
    lightning_pass,
    materials::{
        MasterMaterial, MasterMaterialIndex, MasterMaterialManager, MaterialDataManager,
        MaterialInstanceIndex, MaterialInstanceManager, MaterialPass,
    },
    meshes::MeshManager,
    renderpass_common::RenderpassImageState,
    resource_manager::VulkanResource,
    shader_manager::ShaderManager,
    shadow_map_pass,
    swapchain::{Swapchain, SwapchainError},
    vk_init, vk_util,
};

#[derive(Default)]
pub struct GPUStats {
    pub draw_time: f64,
    pub draw_calls: u32,
    pub triangles: u64,
}

#[derive(Debug)]
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

pub struct ManagedResources {
    pub buffers: BufferManager,
    pub images: ImageManager,

    pub meshes: MeshManager,
    pub master_materials: MasterMaterialManager,
    pub material_instances: MaterialInstanceManager,
    pub material_datas: MaterialDataManager,
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

    pub depth_pre_pass_material: MasterMaterialIndex,
    pub geometry_pass_material: MasterMaterialIndex,
    pub shadow_map_pass_material: MasterMaterialIndex,
    pub transparent_material: MasterMaterialIndex,

    pub geometry_pass_material_instance: MaterialInstanceIndex,
}

pub struct DebugLabel<'a> {
    #[cfg(debug_assertions)]
    ctx: &'a VulkanContext,
    #[cfg(debug_assertions)]
    cmd: vk::CommandBuffer,
}

impl<'a> DebugLabel<'a> {
    pub fn new(
        ctx: &'a VulkanContext,
        cmd: vk::CommandBuffer,
        label_name: &std::ffi::CStr,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            let label = vk::DebugUtilsLabelEXT::default().label_name(label_name);

            unsafe {
                ctx.debug_device.cmd_begin_debug_utils_label(cmd, &label);
            }

            Self { ctx, cmd }
        }

        #[cfg(not(debug_assertions))]
        {
            let _ = (device, cmd, label_name, color);
            Self {}
        }
    }
}

#[cfg(debug_assertions)]
impl Drop for DebugLabel<'_> {
    fn drop(&mut self) {
        unsafe {
            self.ctx.debug_device.cmd_end_debug_utils_label(self.cmd);
        }
    }
}

pub type VulkanResult<T> = Result<T, vk::Result>;

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,

    #[cfg(debug_assertions)]
    pub debug_instance: ash::ext::debug_utils::Instance,
    #[cfg(debug_assertions)]
    pub debug_device: ash::ext::debug_utils::Device,
    #[cfg(debug_assertions)]
    pub debug_utils_messenger: vk::DebugUtilsMessengerEXT,

    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,

    pub graphics_queue: vk::Queue,
    pub graphics_queue_index: u32,
}

impl Deref for VulkanContext {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl VulkanContext {
    fn new(display_handle: RawDisplayHandle) -> Option<Self> {
        let entry = Entry::linked();
        let instance = vk_init::create_instance(&entry, display_handle).ok()?;

        #[cfg(debug_assertions)]
        let debug_instance = debug_utils::Instance::new(&entry, &instance);
        #[cfg(debug_assertions)]
        let debug_utils_messenger = vk_init::create_debug_utils_messenger(&debug_instance);

        let physical_device = vk_init::select_physical_device(&instance);

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let graphics_queue_index =
            vk_init::get_graphics_queue_family_index(&queue_families).unwrap();
        let device = vk_init::create_device(physical_device, &instance, graphics_queue_index);

        let graphics_queue = unsafe {
            device.get_device_queue2(
                &vk::DeviceQueueInfo2::default()
                    .queue_family_index(graphics_queue_index)
                    .queue_index(0), // TODO: 0?
            )
        };

        #[cfg(debug_assertions)]
        let debug_device = debug_utils::Device::new(&instance, &device);

        Some(Self {
            entry,
            instance,

            #[cfg(debug_assertions)]
            debug_instance,
            #[cfg(debug_assertions)]
            debug_device,
            #[cfg(debug_assertions)]
            debug_utils_messenger,

            physical_device,
            device,
            graphics_queue,
            graphics_queue_index,
        })
    }

    pub fn create_debug_utils_label(
        &'_ self,
        cmd: vk::CommandBuffer,
        label_name: &CStr,
    ) -> DebugLabel<'_> {
        DebugLabel::new(self, cmd, label_name)
    }

    fn destroy(&self) {
        unsafe {
            self.destroy_device(None);

            #[cfg(debug_assertions)]
            self.debug_instance
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        };
    }
}

pub struct VulkanEngine {
    frame_number: u64,

    ctx: VulkanContext,

    // TODO: Part of context?
    swapchain: Swapchain,

    double_buffer: DoubleBuffer,

    allocator: ManuallyDrop<Allocator>,

    shader_manager: ShaderManager,

    // TODO: PRIORITY: Leaking
    // Can never be cleared
    descriptor_allocator: DescriptorAllocatorGrowable,

    immediate_submit: ImmediateSubmit,

    gpu_scene_data_descriptor_layout: vk::DescriptorSetLayout,

    pub managed_resources: ManagedResources,

    // TODO: Maybe not internal
    pub default_resources: DefaultResources,

    // TODO: MeshManager that manages that memory with some (buddy?) allocator
    index_buffer: BufferIndex,
    vertex_buffer: BufferIndex,
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
    ) -> Self {
        // TODO: Decide what type of result should be returned
        let ctx = VulkanContext::new(display_handle).unwrap();

        let surface_instance = surface::Instance::new(&ctx.entry, &ctx.instance);
        let swapchain_device = swapchain::Device::new(&ctx.instance, &ctx.device);

        let surface = vk_init::create_surface(&ctx, display_handle, window_handle).unwrap();

        let swapchain = Swapchain::new(
            surface_instance,
            swapchain_device,
            &ctx,
            surface,
            vk::Extent2D::default().width(width).height(height),
        )
        .unwrap();

        let mut allocator = vk_init::create_allocator(&ctx);

        // TODO: Move into context?
        let properties = unsafe {
            ctx.instance
                .get_physical_device_properties(ctx.physical_device)
        };

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
                ratio: 10000,
            },
            PoolSizeRatio {
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                ratio: 1,
            },
        ];

        let mut descriptor_allocator = DescriptorAllocatorGrowable::new(&ctx, 10, pool_ratios);

        let sampler_nearest_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);

        let sampler_linear_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);

        let default_sampler_nearest = unsafe {
            ctx.create_sampler(&sampler_nearest_create_info, None)
                .unwrap()
        };

        let default_sampler_linear = unsafe {
            ctx.create_sampler(&sampler_linear_create_info, None)
                .unwrap()
        };

        let mut image_manager = ImageManager::new();

        let mut shader_manager = ShaderManager::default();

        let gpu_scene_data_descriptor_layout = DescriptorLayoutBuilder::default()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER)
            .add_binding(1, vk::DescriptorType::STORAGE_BUFFER)
            .add_binding(2, vk::DescriptorType::STORAGE_BUFFER)
            .add_binding(3, vk::DescriptorType::STORAGE_BUFFER)
            .add_binding_indexed(4, 1024, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(
                &ctx,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            );

        let immediate_submit = ImmediateSubmit::new(&ctx);

        let buffer_manager = BufferManager::new();

        let mesh_manager = MeshManager::new();

        let image_white = default_resources::image_white(
            &ctx,
            &mut allocator,
            &immediate_submit,
            vk::AccessFlags2::SHADER_READ,
        );

        let image_black = default_resources::image_black(
            &ctx,
            &mut allocator,
            &immediate_submit,
            vk::AccessFlags2::SHADER_READ,
        );

        let image_error = default_resources::image_error(
            &ctx,
            &mut allocator,
            &immediate_submit,
            vk::AccessFlags2::SHADER_READ,
        );

        let image_normal = default_resources::image_normal(
            &ctx,
            &mut allocator,
            &immediate_submit,
            vk::AccessFlags2::SHADER_READ,
        );

        let mut master_material_manager = MasterMaterialManager::new();

        let depth_pre_pass_shader =
            shader_manager.get_graphics_shader_combined(&ctx, "depth_pre_pass");
        let geometry_pass_shader =
            shader_manager.get_graphics_shader_combined(&ctx, "geometry_pass");
        let shadow_map_pass_shader =
            shader_manager.get_graphics_shader_combined(&ctx, "shadow_map_pass");
        let shader = shader_manager.get_graphics_shader_combined(&ctx, "forward_pass");

        let depth_pre_pass_material = MasterMaterial::new_shadow_map(
            &ctx,
            gpu_scene_data_descriptor_layout,
            double_buffer::DEPTH_FORMAT,
            MaterialPass::Opaque,
            &depth_pre_pass_shader,
        );

        let geometry_pass_material = MasterMaterial::new(
            &ctx,
            gpu_scene_data_descriptor_layout,
            &[double_buffer::COLOR_FORMAT, double_buffer::NORMAL_FORMAT],
            double_buffer::DEPTH_FORMAT,
            MaterialPass::Opaque,
            &geometry_pass_shader,
        );

        let shadow_map_pass_material = MasterMaterial::new_shadow_map(
            &ctx,
            gpu_scene_data_descriptor_layout,
            double_buffer::SHADOW_MAP_FORMAT,
            MaterialPass::Opaque,
            &shadow_map_pass_shader,
        );

        let default_transparent_material = MasterMaterial::new(
            &ctx,
            gpu_scene_data_descriptor_layout,
            &[double_buffer::DRAW_FORMAT],
            double_buffer::DEPTH_FORMAT,
            MaterialPass::Transparent,
            &shader,
        );

        let depth_pre_pass_material_index = master_material_manager.add(depth_pre_pass_material);
        let geometry_pass_material_index = master_material_manager.add(geometry_pass_material);
        let shadow_map_pass_material_index = master_material_manager.add(shadow_map_pass_material);
        let default_transparent_material_index =
            master_material_manager.add(default_transparent_material);

        let mut material_instance_manager = MaterialInstanceManager::new();

        let geometry_pass_material_instance = MasterMaterial::create_instance(
            geometry_pass_material_index,
            0.into(), // TODO: Drop default instances
        );

        let geometry_pass_material_instance_index =
            material_instance_manager.add(geometry_pass_material_instance);

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
            material_datas: MaterialDataManager::new(&ctx, &mut allocator),
        };

        let default_resources = DefaultResources {
            image_white: image_white_index,
            image_black: image_black_index,
            image_error: image_error_index,
            image_normal: image_normal_index,
            sampler_nearest: default_sampler_nearest,
            sampler_linear: default_sampler_linear,
            depth_pre_pass_material: depth_pre_pass_material_index,
            geometry_pass_material: geometry_pass_material_index,
            transparent_material: default_transparent_material_index,
            geometry_pass_material_instance: geometry_pass_material_instance_index,
            shadow_map_pass_material: shadow_map_pass_material_index,
        };

        // TODO: Temp untill buddy implemented for buffer management
        const BUFFER_SIZE: usize = 258 * 1024 * 1024;

        let index_buffer = Buffer::new(
            &ctx,
            &mut allocator,
            BUFFER_SIZE,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "index_buffer",
        );

        let vertex_buffer = Buffer::new(
            &ctx,
            &mut allocator,
            BUFFER_SIZE,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "vertex_buffer",
        );

        let material_data_buffer = managed_resources.material_datas.get();

        let double_buffer = DoubleBuffer::new(
            &ctx,
            &mut allocator,
            width,
            height,
            properties.limits.timestamp_period,
            gpu_scene_data_descriptor_layout,
            &mut descriptor_allocator,
            default_sampler_linear,
            &mut shader_manager,
            &vertex_buffer,
            material_data_buffer,
            &managed_resources.images,
        );

        let index_buffer_index = managed_resources.buffers.add(index_buffer);
        let vertex_buffer_index = managed_resources.buffers.add(vertex_buffer);

        Self {
            frame_number: 0,

            ctx,

            swapchain,
            double_buffer,
            allocator: ManuallyDrop::new(allocator),

            shader_manager,

            descriptor_allocator,

            immediate_submit,

            gpu_scene_data_descriptor_layout,

            managed_resources,
            default_resources,

            index_buffer: index_buffer_index,
            vertex_buffer: vertex_buffer_index,
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn draw(
        &mut self,
        main_camera: &Camera,
        draw_context: &DrawContext,
        render_scale: f32,
    ) -> Result<GPUStats, DrawError> {
        let mut gpu_stats = GPUStats::default();

        let render_scale = render_scale.clamp(0.25, 1.0);

        let width = self.swapchain.extent.width as f32;
        let height = self.swapchain.extent.height as f32;

        unsafe {
            let query_results = self
                .double_buffer
                .swap_buffer(&self.ctx, &mut self.allocator);

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
            let lightning_pass_data = self.double_buffer.get_lightning_pass_data();
            let fxaa_pass_data = self.double_buffer.get_fxaa_pass_data();

            let draw_src = RenderpassImageState::new(&frame_targets.draw);
            let color_src = RenderpassImageState::new(&frame_targets.color);
            let normal_src = RenderpassImageState::new(&frame_targets.normal);
            let depth_src = RenderpassImageState::new(&frame_targets.depth);
            let shadow_map_src = RenderpassImageState::new(&frame_targets.shadow_map);
            let fxaa_src = RenderpassImageState::new(&frame_targets.fxaa);

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

            self.ctx
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");

            let cmd_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            // Begin command buffer

            self.ctx
                .begin_command_buffer(cmd, &cmd_begin_info)
                .expect("Failed to begin command buffer");

            self.ctx.cmd_write_timestamp(
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
            } = DrawRecord::record_draw_commands(draw_context, &self.managed_resources);

            self.double_buffer.set_globals(&Self::create_scene_data(
                main_camera,
                self.frame_number,
                width,
                height,
            ));

            let mut write_data = self.double_buffer.upload_buffers(&self.ctx, cmd);
            self.managed_resources.material_datas.upload(&self.ctx, cmd);

            let indexed_indirect_data = IndexedIndirectData::prepare(
                &opaque_commands,
                &transparent_commands,
                &mut write_data,
                &mut gpu_stats,
            );

            let index_buffer = self.managed_resources.buffers.get(self.index_buffer).buffer;

            let depth_pre_pass_master_material = self
                .managed_resources
                .master_materials
                .get(self.default_resources.depth_pre_pass_material);

            let depth_pre_pass_output = depth_pre_pass::record(
                &self.ctx,
                cmd,
                render_area,
                depth_src,
                depth_pre_pass_master_material,
                globals_descriptor_set,
                index_buffer,
                &indexed_indirect_data.opaque,
            );

            let geometry_pass_output = geometry_pass::record(
                &self.ctx,
                cmd,
                render_area,
                color_src,
                normal_src,
                depth_pre_pass_output.depth,
                globals_descriptor_set,
                index_buffer,
                &indexed_indirect_data.opaque,
            );

            let shadow_pass_master_material = self
                .managed_resources
                .master_materials
                .get(self.default_resources.shadow_map_pass_material);

            let shadow_map_pass_output = shadow_map_pass::record(
                &self.ctx,
                cmd,
                shadow_map_src,
                shadow_pass_master_material,
                globals_descriptor_set,
                index_buffer,
                &indexed_indirect_data.opaque,
            );

            let lightning_pass_output = lightning_pass::record(
                &self.ctx,
                cmd,
                render_area,
                draw_src,
                geometry_pass_output.color,
                geometry_pass_output.normal,
                geometry_pass_output.depth,
                shadow_map_pass_output.shadow_map,
                globals_descriptor_set,
                &lightning_pass_data,
            );

            let draw_image_state = forward_pass::record(
                &self.ctx,
                cmd,
                render_area,
                lightning_pass_output.draw,
                lightning_pass_output.depth,
                globals_descriptor_set,
                index_buffer,
                &indexed_indirect_data.transparent,
            );

            let fxaa_pass_output = fxaa_pass::record(
                &self.ctx,
                cmd,
                render_area,
                draw_image_state,
                fxaa_src,
                globals_descriptor_set,
                &fxaa_pass_data,
            );

            let fxaa_to_transfer_src = vk_util::create_image_memory_barrier(
                fxaa_pass_output.fxaa.image,
                fxaa_pass_output.fxaa.layout,
                fxaa_pass_output.fxaa.stage_mask,
                fxaa_pass_output.fxaa.access_mask,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageAspectFlags::COLOR,
            );

            let swapchain_to_transfer_dst = vk_util::create_image_memory_barrier(
                acquired_swapchain.image,
                vk::ImageLayout::UNDEFINED,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageAspectFlags::COLOR,
            );

            let image_memory_barriers = [fxaa_to_transfer_src, swapchain_to_transfer_dst];

            vk_util::pipeline_barrier(&self.ctx, cmd, &image_memory_barriers);

            vk_util::blit_image(
                &self.ctx,
                cmd,
                fxaa_pass_output.fxaa.image,
                acquired_swapchain.image,
                render_area.extent,
                self.swapchain.extent,
            );

            vk_util::pipeine_barrier_single_image(
                &self.ctx,
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

            self.ctx.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool,
                1,
            );

            self.ctx
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

            self.ctx
                .queue_submit2(
                    self.ctx.graphics_queue,
                    &[submit_info],
                    synchronization_resources.fence,
                )
                .expect("Failed to queue submit");

            self.swapchain.queue_present(
                acquired_swapchain.index,
                self.ctx.graphics_queue,
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

        let _sin_x = f32::sin(frame_number as f32 / 200.0);
        let _cos_z = f32::cos(frame_number as f32 / 200.0);
        let sunlight_direction = vector![1.0, 4.0, 1.0].normalize().insert_row(3, 3.0);

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

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        self.swapchain.recreate_swapchain(&self.ctx, width, height);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_image<T>(
        &mut self,
        extent: vk::Extent3D,
        format: vk::Format,
        image_usage: vk::ImageUsageFlags,
        access_flags: vk::AccessFlags2,
        access_mask: vk::ImageAspectFlags,
        mipmapped: bool,
        data: &[T],
        name: &str,
    ) -> ImageIndex {
        let image = Image::with_data(
            &self.ctx,
            &mut self.allocator,
            &self.immediate_submit,
            extent,
            format,
            image_usage,
            access_flags,
            access_mask,
            mipmapped,
            data,
            name,
        );

        self.managed_resources.images.add(image)
    }

    pub fn upload_buffers(&mut self, indices: &[u32], vertices: &[Vertex]) {
        let index_buffer_size = size_of_val(indices);
        let vertex_buffer_size = size_of_val(vertices);

        // TODO: Allocation separate?

        let mut staging = Buffer::new(
            &self.ctx,
            &mut self.allocator,
            index_buffer_size + vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "staging",
        );

        vk_util::copy_data_to_allocation(indices, staging.allocation.as_ref().unwrap());
        vk_util::copy_data_to_allocation_with_byte_offset(
            vertices,
            staging.allocation.as_ref().unwrap(),
            index_buffer_size,
        );

        self.immediate_submit.submit(&self.ctx, |cmd| {
            let index_regions = [vk::BufferCopy::default().size(index_buffer_size as u64)];

            unsafe {
                let index_buffer = &self.managed_resources.buffers.get(self.index_buffer);
                self.ctx
                    .cmd_copy_buffer(cmd, staging.buffer, index_buffer.buffer, &index_regions);
            };

            let vertex_regions: [vk::BufferCopy; 1] = [vk::BufferCopy::default()
                .src_offset(index_buffer_size as u64)
                .size(vertex_buffer_size as u64)];

            unsafe {
                let vertex_buffer = &self.managed_resources.buffers.get(self.vertex_buffer);

                self.ctx.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    vertex_buffer.buffer,
                    &vertex_regions,
                );
            }
        });

        staging.destroy(&self.ctx, &mut self.allocator);
    }

    pub fn update_images(&mut self) {
        self.double_buffer.update_images(
            &self.ctx,
            &self.default_resources,
            &self.managed_resources.images,
        );
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        let ctx = &self.ctx;
        let allocator = &mut self.allocator;

        unsafe { ctx.queue_wait_idle(self.ctx.graphics_queue).unwrap() };

        self.double_buffer.destroy(ctx, allocator);

        self.managed_resources
            .images
            .remove(ctx, allocator, self.default_resources.image_white);
        self.managed_resources
            .images
            .remove(ctx, allocator, self.default_resources.image_black);
        self.managed_resources
            .images
            .remove(ctx, allocator, self.default_resources.image_error);

        // TODO: Shouldn't be needed if all resources are removed properly
        self.managed_resources.images.destroy(ctx, allocator);

        self.managed_resources.buffers.destroy(ctx, allocator);

        self.managed_resources
            .master_materials
            .destroy(ctx, allocator);

        self.managed_resources
            .material_datas
            .destroy(ctx, allocator);
        // ~Shouldn't be needed if all resources are removed properly

        self.shader_manager.destroy(ctx);

        self.descriptor_allocator.destroy(ctx);

        self.immediate_submit.destroy(ctx);

        unsafe {
            ctx.destroy_descriptor_set_layout(self.gpu_scene_data_descriptor_layout, None);
        };

        unsafe {
            ctx.destroy_sampler(self.default_resources.sampler_linear, None);
            ctx.destroy_sampler(self.default_resources.sampler_nearest, None);
        }

        dbg!(allocator.generate_report());

        unsafe { ManuallyDrop::drop(allocator) };

        self.swapchain.destroy(ctx);
        self.ctx.destroy();
    }
}
