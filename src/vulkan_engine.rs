use std::mem::ManuallyDrop;

use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};

use gpu_allocator::{MemoryLocation, vulkan::Allocator};

use nalgebra::{Matrix4, Vector3, Vector4, vector};
use thiserror::Error;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{
    buffers::{Buffer, BufferIndex, BufferManager},
    camera::{Camera, InputManager},
    default_resources,
    descriptors::{
        DescriptorAllocatorGrowable, DescriptorLayoutBuilder, DescriptorWriter, PoolSizeRatio,
    },
    double_buffer::{self, DoubleBuffer},
    gltf_loader::GLTFLoader,
    images::{ImageIndex, ImageManager},
    immediate_submit::ImmediateSubmit,
    materials::{
        MasterMaterial, MasterMaterialIndex, MasterMaterialManager, MaterialConstants,
        MaterialInstanceIndex, MaterialInstanceManager, MaterialPass, MaterialResources,
    },
    meshes::{MeshIndex, MeshManager},
    shader_manager::ShaderManager,
    swapchain::{Swapchain, SwapchainError},
    vk_init, vk_util,
};

pub struct DrawCommand {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // TODO: Probably just index here and query + bind only when actualy before needed for drawing
    material_instance_set: vk::DescriptorSet,
    index_buffer: vk::Buffer,
    vertex_buffer_address: u64,
    world_matrix: Matrix4<f32>,
    surface_index_count: u32,
    surface_first_index: u32,
}

// TODO: Move this somewhere
impl DrawCommand {
    pub fn cmd_record_draw_command(
        device: &Device,
        cmd: vk::CommandBuffer,
        global_descriptor: vk::DescriptorSet,
        draw_commands: &[DrawCommand],
        draw_order: &[u32],
        gpu_stats: &mut GPUStats,
    ) {
        let mut last_material_set = vk::DescriptorSet::null();
        let mut last_pipeline = vk::Pipeline::null();
        let mut last_index_buffer = vk::Buffer::null();

        for index in draw_order {
            let command = &draw_commands[*index as usize];
            unsafe {
                if last_material_set != command.material_instance_set {
                    last_material_set = command.material_instance_set;

                    if last_pipeline != command.pipeline {
                        last_pipeline = command.pipeline;

                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            command.pipeline,
                        );

                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            command.pipeline_layout,
                            0,
                            &[global_descriptor],
                            &[],
                        );

                        // TODO: Dynamic state
                    }

                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        command.pipeline_layout,
                        1,
                        &[command.material_instance_set],
                        &[],
                    );
                }

                if last_index_buffer != command.index_buffer {
                    last_index_buffer = command.index_buffer;

                    device.cmd_bind_index_buffer(
                        cmd,
                        command.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                let push_constants = GPUPushDrawConstant {
                    world_matrix: command.world_matrix,
                    vertex_buffer: command.vertex_buffer_address,
                };

                device.cmd_push_constants(
                    cmd,
                    command.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    push_constants.as_bytes(),
                );

                device.cmd_draw_indexed(
                    cmd,
                    command.surface_index_count,
                    1,
                    command.surface_first_index,
                    0,
                    0,
                );

                gpu_stats.draw_calls += 1;
                gpu_stats.triangles += command.surface_index_count as usize / 3;
            }
        }
    }
}

struct DrawRecord {
    opaque_commands: Vec<DrawCommand>,
    transparent_commands: Vec<DrawCommand>,
}

impl DrawRecord {
    fn record_draw_commands(
        draw_context: &DrawContext,
        managed_resources: &ManagedResources,
    ) -> DrawRecord {
        let mut opaque_commands = vec![];
        let mut transparent_commands = vec![];

        for draw in &draw_context.render_objects {
            let mesh = managed_resources.meshes.get(draw.mesh_index);

            for surface in &mesh.surfaces {
                let material = managed_resources
                    .material_instances
                    .get(surface.material_instance_index);

                let master_material = managed_resources
                    .master_materials
                    .get(material.master_material_index);

                let pipeline_layout = master_material.pipeline_layout;
                let pipeline = master_material.pipeline;
                let material_instance_set = material.set;
                let index_buffer = managed_resources
                    .buffers
                    .get(mesh.buffers.index_buffer_index)
                    .buffer;
                let vertex_buffer_address = mesh.buffers.vertex_buffer_address;
                let world_matrix = draw.transform;
                let surface_first_index = surface.start_index;
                let surface_index_count = surface.count;

                let draw_command = DrawCommand {
                    pipeline_layout,
                    pipeline,
                    material_instance_set,
                    index_buffer,
                    vertex_buffer_address,
                    world_matrix,
                    surface_index_count,
                    surface_first_index,
                };

                if master_material.material_pass == MaterialPass::Opaque {
                    opaque_commands.push(draw_command);
                } else {
                    transparent_commands.push(draw_command);
                }
            }
        }

        Self {
            opaque_commands,
            transparent_commands,
        }
    }
}

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
pub struct DrawContext {
    pub render_objects: Vec<RenderObject>,
}

pub struct RenderObject {
    pub mesh_index: MeshIndex,

    pub transform: Matrix4<f32>,
}

#[derive(Default)]
struct GPUSceneData {
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    view_proj: Matrix4<f32>,
    ambient_color: Vector4<f32>,
    sunlight_direction: Vector4<f32>, // w for sun power
    sunlight_color: Vector4<f32>,
}

impl GPUSceneData {
    fn as_bytes(&self) -> &[u8] {
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

#[derive(Default, Clone)]
#[repr(C)]
pub struct Vertex {
    pub position: Vector3<f32>,
    pub uv_x: f32,
    pub normal: Vector3<f32>,
    pub uv_y: f32,
    pub color: Vector4<f32>,
}

// TODO: Probably can be part of Mesh
pub struct GPUMeshBuffers {
    pub index_buffer_index: BufferIndex,
    pub vertex_buffer_index: BufferIndex,
    pub vertex_buffer_address: vk::DeviceAddress,
}

#[repr(C)]
pub struct GPUPushDrawConstant {
    world_matrix: Matrix4<f32>,
    vertex_buffer: vk::DeviceAddress,
}

impl GPUPushDrawConstant {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Self>(self).cast::<u8>(),
                size_of::<Self>(),
            )
        }
    }
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

    pub sampler_nearest: vk::Sampler,
    pub sampler_linear: vk::Sampler,

    pub opaque_material: MasterMaterialIndex,
    pub transparent_material: MasterMaterialIndex,

    pub default_material_instance: MaterialInstanceIndex,
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

    scene_data: GPUSceneData,
    gpu_scene_data_descriptor_layout: vk::DescriptorSetLayout,

    managed_resources: ManagedResources,
    default_resources: DefaultResources,

    main_draw_context: DrawContext,

    main_camera: Camera, // TODO: Shouldn't be part of renderer

    query_pool: vk::QueryPool,

    gltf_loader: GLTFLoader,
}

impl VulkanEngine {
    const _USE_VALIDATION_LAYERS: bool = false;

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

        let double_buffer = DoubleBuffer::new(
            &device,
            &mut allocator,
            graphics_queue_family_index,
            width,
            height,
        );

        let mut image_manager = ImageManager::new();

        let mut shader_manager = ShaderManager::default();

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

        let mut master_material_manager = MasterMaterialManager::new();

        let default_opaque_material = MasterMaterial::new(
            &device,
            &mut shader_manager,
            gpu_scene_data_descriptor_layout,
            &[double_buffer::DRAW_FORMAT],
            double_buffer::DEPTH_FORMAT,
            MaterialPass::Opaque,
        );

        let default_transparent_material = MasterMaterial::new(
            &device,
            &mut shader_manager,
            gpu_scene_data_descriptor_layout,
            &[double_buffer::DRAW_FORMAT],
            double_buffer::DEPTH_FORMAT,
            MaterialPass::Transparent,
        );

        let default_opaque_material_index = master_material_manager.add(default_opaque_material);
        let default_transparent_material_index =
            master_material_manager.add(default_transparent_material);

        let mut material_instance_manager = MaterialInstanceManager::new();

        let material_constants_buffer = Buffer::new(
            &device,
            &mut allocator,
            size_of::<MaterialConstants>() as u64,
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
            metal_rough_image_view: image_white.image_view,
            metal_rough_sampler: default_sampler_linear,
            data_buffer: material_constants_buffer.buffer,
            data_buffer_offset: 0,
        };

        buffer_manager.add(material_constants_buffer);

        let default_opaque_material =
            master_material_manager.get_mut(default_opaque_material_index);

        let default_opaque_material_instance = default_opaque_material.create_instance(
            &device,
            &resources,
            &mut descriptor_allocator,
            default_opaque_material_index,
        );

        let default_opaque_material_instance_index =
            material_instance_manager.add(default_opaque_material_instance);

        let main_camera = Camera {
            position: vector![0.0, 0.0, 5.0],
            velocity: Vector3::from_element(0.0),
            pitch: 0.0,
            yaw: 0.0,
        };

        let query_pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);

        let query_pool = unsafe {
            device
                .create_query_pool(&query_pool_create_info, None)
                .unwrap()
        };

        let image_white_index = image_manager.add(image_white);
        let image_black_index = image_manager.add(image_black);
        let image_error_index = image_manager.add(image_error);

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
            sampler_nearest: default_sampler_nearest,
            sampler_linear: default_sampler_linear,
            opaque_material: default_opaque_material_index,
            transparent_material: default_transparent_material_index,
            default_material_instance: default_opaque_material_instance_index,
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

            scene_data: GPUSceneData::default(),
            gpu_scene_data_descriptor_layout,

            managed_resources,
            default_resources,

            main_draw_context: DrawContext::default(),

            main_camera,

            query_pool,

            gltf_loader,
        }
    }

    // TODO: Shouldn't be part of renderer
    pub fn update(&mut self, input_manager: &InputManager) {
        self.main_camera.process_winit_events(input_manager);
    }

    #[allow(clippy::too_many_lines)]
    fn draw_geometry(
        &mut self,
        cmd: vk::CommandBuffer,
        draw_extent: vk::Extent2D,
        gpu_stats: &mut GPUStats,
        draw_image_view: vk::ImageView,
        depth_image_view: vk::ImageView,
    ) {
        let gpu_scene_data_buffer = Buffer::new(
            &self.device,
            &mut self.allocator,
            size_of::<GPUSceneData>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "draw_geometry",
        );

        // TODO: part of allocated buffer?
        vk_util::copy_data_to_allocation(
            self.scene_data.as_bytes(),
            gpu_scene_data_buffer.allocation.as_ref().unwrap(),
        );

        // Part of DoubleBuffer/FrameBuffer
        let global_descriptor = self
            .double_buffer
            .allocate_set(&self.device, self.gpu_scene_data_descriptor_layout);

        let mut writer = DescriptorWriter::default();
        writer.write_buffer(
            0,
            gpu_scene_data_buffer.buffer,
            size_of::<GPUSceneData>() as u64,
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
        );

        writer.update_set(&self.device, global_descriptor);

        self.double_buffer.add_buffer(gpu_scene_data_buffer);

        let clear_color = Some(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [1.0, 0.0, 1.0, 1.0],
            },
        });

        let color_attachment = [vk_util::attachment_info(
            draw_image_view,
            clear_color,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )];

        let depth_attachment = vk_util::depth_attachment_info(
            depth_image_view,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        let rendering_info =
            vk_util::rendering_info(draw_extent, &color_attachment, &depth_attachment);

        let viewports = [vk::Viewport::default()
            .width(draw_extent.width as f32)
            .height(draw_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissors = [vk::Rect2D::default().extent(draw_extent)];

        unsafe {
            self.device.cmd_begin_rendering(cmd, &rendering_info);

            self.device.cmd_set_viewport(cmd, 0, &viewports);
            self.device.cmd_set_scissor(cmd, 0, &scissors);

            self.draw_meshes(cmd, global_descriptor, gpu_stats);
        };

        unsafe {
            self.device.cmd_end_rendering(cmd);
        }
    }

    fn draw_meshes(
        &mut self,
        cmd: vk::CommandBuffer,
        global_descriptor: vk::DescriptorSet,
        gpu_stats: &mut GPUStats,
    ) {
        let DrawRecord {
            opaque_commands,
            transparent_commands,
        } = DrawRecord::record_draw_commands(&self.main_draw_context, &self.managed_resources);

        let mut opaque_order: Vec<u32> = (0..opaque_commands.len() as u32).collect();

        opaque_order.sort_by(|lhs, rhs| {
            let lhs = &opaque_commands[*lhs as usize];
            let rhs = &opaque_commands[*rhs as usize];

            match lhs.material_instance_set.cmp(&rhs.material_instance_set) {
                std::cmp::Ordering::Equal => lhs.index_buffer.cmp(&rhs.index_buffer),
                other => other,
            }
        });

        DrawCommand::cmd_record_draw_command(
            &self.device,
            cmd,
            global_descriptor,
            &opaque_commands,
            &opaque_order,
            gpu_stats,
        );

        let mut transparent_order: Vec<u32> = (0..transparent_commands.len() as u32).collect();

        transparent_order.sort_by(|lhs, rhs| {
            let lhs = &transparent_commands[*lhs as usize];
            let rhs = &transparent_commands[*rhs as usize];

            // if lhs.material_instance_set < rhs.material_instance_set {
            //     lhs.index_buffer < rhs.index_buffer
            // } else {
            //     lhs.material_instance_set < rhs.material_instance_set
            // }

            match lhs.material_instance_set.cmp(&rhs.material_instance_set) {
                std::cmp::Ordering::Equal => lhs.index_buffer.cmp(&rhs.index_buffer),
                other => other,
            }
        });

        DrawCommand::cmd_record_draw_command(
            &self.device,
            cmd,
            global_descriptor,
            &transparent_commands,
            &transparent_order,
            gpu_stats,
        );
    }

    #[allow(clippy::too_many_lines)]
    pub fn draw(&mut self, render_scale: f32) -> Result<GPUStats, DrawError> {
        let mut gpu_stats = GPUStats::default();

        let render_scale = render_scale.clamp(0.25, 1.0);

        self.update_scene();

        unsafe {
            self.double_buffer
                .swap_buffer(&self.device, &mut self.allocator);

            let synchronization_resources = self.double_buffer.get_synchronization_resources();
            let cmd = self.double_buffer.get_command_buffer();

            // TODO: encapsulate, into swapchain?
            let acquired_swapchain = self
                .swapchain
                .acquire_next_image(synchronization_resources.swapchain_semaphore)?;

            let draw_image = self.double_buffer.get_draw_image();
            let depth_image = self.double_buffer.get_depth_image();

            let draw_width =
                self.swapchain.extent.width.min(draw_image.extent.width) as f32 * render_scale;

            let draw_height =
                self.swapchain.extent.height.min(draw_image.extent.height) as f32 * render_scale;

            let draw_extent = vk::Extent2D::default()
                .width(draw_width as u32)
                .height(draw_height as u32);

            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");

            let cmd_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            // Begin command buffer

            self.device
                .begin_command_buffer(cmd, &cmd_begin_info)
                .expect("Failed to begin command buffer");

            self.device.reset_query_pool(self.query_pool, 0, 2);
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                0,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                depth_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

            self.draw_geometry(
                cmd,
                draw_extent,
                &mut gpu_stats,
                draw_image.image_view,
                depth_image.image_view,
            );

            // TODO: If possible don't get it again here, but draw_geometry is currently &mut self
            let draw_image = self.double_buffer.get_draw_image();

            vk_util::transition_image(
                &self.device,
                cmd,
                draw_image.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                acquired_swapchain.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
            );

            vk_util::blit_image(
                &self.device,
                cmd,
                draw_image.image,
                acquired_swapchain.image,
                draw_extent,
                self.swapchain.extent,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                acquired_swapchain.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
            );

            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
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

            // let mut query_results: [u64; 2] = [0, 0];

            // self.device
            //     .get_query_pool_results(
            //         self.query_pool,
            //         0,
            //         &mut query_results,
            //         vk::QueryResultFlags::TYPE_64,
            //     )
            //     .unwrap();

            // let properties = self
            //     .instance
            //     .get_physical_device_properties(self.physical_device);

            // gpu_stats.draw_time = (query_results[1] as f64 - query_results[0] as f64)
            //     * f64::from(properties.limits.timestamp_period)
            //     / 1_000_000.0f64;

            self.frame_number += 1;
        };

        Ok(gpu_stats)
    }

    fn update_scene(&mut self) {
        self.main_camera.update();

        self.main_draw_context.render_objects.clear();

        // TODO: More global source of size/aspect ratio
        let draw_image = self.double_buffer.get_draw_image();

        self.gltf_loader.draw(&mut self.main_draw_context);

        self.scene_data.view = self.main_camera.get_view();
        self.scene_data.proj = Camera::get_projection(
            draw_image.extent.height as f32 / draw_image.extent.width as f32,
        );
        self.scene_data.view_proj = self.scene_data.proj * self.scene_data.view;

        self.scene_data.ambient_color = Vector4::from_element(0.1);
        self.scene_data.sunlight_color = Vector4::from_element(1.0);
        self.scene_data.sunlight_direction = vector![0.0, 1.0, 0.5, 1.0];
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
            device.destroy_query_pool(self.query_pool, None);
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
