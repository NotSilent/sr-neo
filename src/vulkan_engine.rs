use std::{
    borrow::Cow,
    cell::RefCell,
    ffi::{self, c_char},
};

use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk::{self, DescriptorPool, PipelineShaderStageCreateInfo},
};

use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};

use nalgebra::{Matrix4, Rotation3, Translation3, Vector3, Vector4, vector};
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{
    deletion_queue::{DeletionQueue, DeletionType},
    shader_manager::ShaderManager,
};
use crate::{pipeline_builder::PipelineBuilder, vk_util};

struct GeoSurface {
    start_index: u32,
    count: u32,
}

struct MeshAsset {
    name: String,
    surfaces: Vec<GeoSurface>,
    mesh_buffers: GPUMeshBuffers,
}

impl MeshAsset {
    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        self.mesh_buffers.index_buffer.destroy(device, allocator);
        self.mesh_buffers.vertex_buffer.destroy(device, allocator);
    }
}

struct ImmediateSubmit {
    graphics_queue: vk::Queue,
    fence: vk::Fence,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}

impl ImmediateSubmit {
    pub fn new(
        device: &Device,
        graphics_queue: vk::Queue,
        graphics_queue_family_index: u32,
    ) -> Self {
        let pool = vk_util::create_command_pool(device, graphics_queue_family_index);
        Self {
            graphics_queue,
            fence: vk_util::create_fence(device, vk::FenceCreateFlags::SIGNALED),
            pool,
            cmd: vk_util::allocate_command_buffer(device, pool),
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
        }
    }

    pub fn submit<F: Fn(vk::CommandBuffer)>(&self, device: &Device, record: F) {
        unsafe {
            device.reset_fences(&[self.fence]).unwrap();
            device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(self.cmd, &begin_info).unwrap();

            record(self.cmd);

            device.end_command_buffer(self.cmd).unwrap();

            let cmd_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(self.cmd)
                .device_mask(0)];

            let submit_infos = [vk::SubmitInfo2::default().command_buffer_infos(&cmd_infos)];

            device
                .queue_submit2(self.graphics_queue, &submit_infos, self.fence)
                .unwrap();

            device
                .wait_for_fences(&[self.fence], true, 1_000_000_000)
                .unwrap();
        }
    }
}

struct AllocatedBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>, // TODO: Drop Option<> somehow (maybe put in sparse vec and index for deletion?)
}

impl AllocatedBuffer {
    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        let _ = allocator.free(self.allocation.take().unwrap());
        unsafe { device.destroy_buffer(self.buffer, None) };
    }
}

#[derive(Default, Clone)]
#[repr(C)]
struct Vertex {
    position: Vector3<f32>,
    uv_x: f32,
    normal: Vector3<f32>,
    uv_y: f32,
    color: Vector4<f32>,
}

struct GPUMeshBuffers {
    index_buffer: AllocatedBuffer,
    vertex_buffer: AllocatedBuffer,
    vertex_buffer_address: vk::DeviceAddress,
}

#[repr(C)]
struct GPUPushDrawConstant {
    world_matrix: Matrix4<f32>,
    vertex_buffer: vk::DeviceAddress,
}

impl GPUPushDrawConstant {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Self>(self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

// TODO: std430
#[repr(C)]
struct ComputePushConstants {
    data1: Vector4<f32>,
    data2: Vector4<f32>,
    data3: Vector4<f32>,
    data4: Vector4<f32>,
}

impl ComputePushConstants {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Self>(self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[derive(Default)]
struct DestructorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl DestructorLayoutBuilder<'_> {
    fn add_binding(&mut self, binding: u32, descriptor_type: vk::DescriptorType) {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_count(1)
                .descriptor_type(descriptor_type),
        );
    }

    fn clear(&mut self) {
        self.bindings.clear();
    }

    fn build(
        &mut self,
        device: &Device,
        shader_stages: vk::ShaderStageFlags,
        /* , void* pNext = nullptr */ flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> vk::DescriptorSetLayout {
        for binding in &mut self.bindings {
            binding.stage_flags |= shader_stages;
        }

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings)
            .flags(flags);

        unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        }
    }
}

struct PoolSizeRatio {
    descriptor_type: vk::DescriptorType,
    ratio: f32, //TODO: why tf f32?
}

#[derive(Default)]
struct DescriptorAllocator {
    pool: DescriptorPool,
}

impl DescriptorAllocator {
    fn init_pool(&mut self, device: &Device, max_sets: u32, pool_ratios: &[PoolSizeRatio]) {
        let mut pool_sizes = Vec::new();
        for ratio in pool_ratios {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(ratio.descriptor_type)
                    .descriptor_count((ratio.ratio * max_sets as f32) as u32),
            );
        }

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        self.pool = unsafe { device.create_descriptor_pool(&create_info, None).unwrap() };
    }

    fn clear_descriptors(&self, device: &Device) {
        unsafe {
            device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        };
    }

    fn destroy_pool(&self, device: &Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }

    // TODO: Multiple layouts
    fn allocate(&self, device: &Device, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let binding = [layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&binding);

        unsafe {
            *device
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()
                .first()
                .unwrap()
        }
    }
}

// TODO: split into struct of arrays?
struct AllocatedImage {
    image: vk::Image,
    image_view: vk::ImageView,
    image_extent: vk::Extent3D,
    image_format: vk::Format,
    _allocation: Allocation,
}

#[derive(Default)]
struct FrameData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    swapchain_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    fence: vk::Fence,

    deletion_queue: RefCell<DeletionQueue>, // TODO: can this be done without RefCell?
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
    swapchain_extent: vk::Extent2D,

    frames: [FrameData; FRAME_OVERLAP],
    deletion_queue: DeletionQueue,
    allocator: Option<Allocator>,

    draw_image: AllocatedImage,
    draw_extent: vk::Extent2D,

    shader_manager: ShaderManager,

    descriptor_allocator: DescriptorAllocator,

    _draw_image_descriptor_layout: vk::DescriptorSetLayout,
    draw_image_descriptors: vk::DescriptorSet,

    gradient_pipeline_layout: vk::PipelineLayout,
    gradient_pipeline: vk::Pipeline,

    triangle_pipeline_layout: vk::PipelineLayout,
    triangle_pipeline: vk::Pipeline,

    immediate_submit: ImmediateSubmit,

    triangle_mesh_pipeline_layout: vk::PipelineLayout,
    triangle_mesh_pipeline: vk::Pipeline,

    rectangle: GPUMeshBuffers,

    mesh_assets: Vec<MeshAsset>,
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
        gltf_name: &str,
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
        frames[0].command_pool = vk_util::create_command_pool(&device, graphics_queue_family_index);
        frames[0].command_buffer =
            vk_util::allocate_command_buffer(&device, frames[0].command_pool);
        frames[0].fence = vk_util::create_fence(&device, vk::FenceCreateFlags::SIGNALED);
        frames[0].swapchain_semaphore = vk_util::create_semaphore(&device);
        frames[0].render_semaphore = vk_util::create_semaphore(&device);
        frames[1].command_pool = vk_util::create_command_pool(&device, graphics_queue_family_index);
        frames[1].command_buffer =
            vk_util::allocate_command_buffer(&device, frames[1].command_pool);
        frames[1].fence = vk_util::create_fence(&device, vk::FenceCreateFlags::SIGNALED);
        frames[1].swapchain_semaphore = vk_util::create_semaphore(&device);
        frames[1].render_semaphore = vk_util::create_semaphore(&device);

        let mut allocator =
            Self::create_allocator(instance.clone(), device.clone(), physical_device);

        let mut deletion_queue = DeletionQueue::default();

        let draw_image_extent = vk::Extent3D::default().width(width).height(height).depth(1);

        let format = vk::Format::R16G16B16A16_SFLOAT;

        let image_usages = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;

        let image_create_info = vk_util::image_create_info(format, image_usages, draw_image_extent);

        let image = unsafe { device.create_image(&image_create_info, None).unwrap() };
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let description = AllocationCreateDesc {
            name: "image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator.allocate(&description).unwrap();

        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();
        };

        let image_view_create_info =
            vk_util::image_view_create_info(format, image, vk::ImageAspectFlags::COLOR);
        let image_view = unsafe {
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };

        let draw_image = AllocatedImage {
            image,
            image_view,
            image_extent: draw_image_extent,
            image_format: format,
            _allocation: allocation,
        };

        let mut shader_manager = ShaderManager::new();

        // init descriptors
        let sizes = vec![PoolSizeRatio {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ratio: 1.0,
        }];

        let mut descriptor_allocator = DescriptorAllocator::default();
        descriptor_allocator.init_pool(&device, 10, &sizes);

        let mut builder = DestructorLayoutBuilder::default();
        builder.add_binding(0, vk::DescriptorType::STORAGE_IMAGE);

        let draw_image_descriptor_layout = builder.build(
            &device,
            vk::ShaderStageFlags::COMPUTE,
            vk::DescriptorSetLayoutCreateFlags::empty(),
        );

        let draw_image_descriptors =
            descriptor_allocator.allocate(&device, draw_image_descriptor_layout);

        let img_infos = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(draw_image.image_view)];

        let draw_image_write = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(draw_image_descriptors)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&img_infos);

        unsafe { device.update_descriptor_sets(&[draw_image_write], &[]) };
        // ~init descriptors

        // init pipelines
        let draw_image_descriptor_layouts = [draw_image_descriptor_layout];

        let push_constant_ranges = [vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<ComputePushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let compute_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&draw_image_descriptor_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let compute_layout = unsafe {
            device
                .create_pipeline_layout(&compute_layout_create_info, None)
                .unwrap()
        };

        let compute_shader = shader_manager.get_compute_shader(&device, "gradient_color");

        let stage_info = PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(compute_shader)
            .name(c"main");

        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .layout(compute_layout)
            .stage(stage_info);

        let compute_pipeline = *unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[compute_pipeline_create_info],
                None,
            )
        }
        .unwrap()
        .first()
        .unwrap();
        // ~init pipelines

        // triangle

        let triangle_shader = shader_manager.get_graphics_shader(&device, "colored_triangle");

        let triangle_pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
        let triangle_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&triangle_pipeline_layout_create_info, None)
                .unwrap()
        };

        let mut pipeline_builder = PipelineBuilder::default();
        #[allow(clippy::field_reassign_with_default)]
        {
            pipeline_builder.pipeline_layout = triangle_pipeline_layout;
        }
        pipeline_builder.set_shaders(triangle_shader.vert, triangle_shader.frag);
        pipeline_builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        pipeline_builder.set_polygon_mode(vk::PolygonMode::FILL);
        pipeline_builder.set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE); // TODO: COUNTER_CLOCKWISE
        pipeline_builder.set_multisampling_none();
        pipeline_builder.disable_blending();
        pipeline_builder.disable_depth_test();
        pipeline_builder.set_color_attachment_format(draw_image.image_format);
        pipeline_builder.set_depth_format(vk::Format::UNDEFINED);

        let triangle_pipeline = pipeline_builder.build_pipeline(&device);

        // ~triangle

        let immediate_submit =
            ImmediateSubmit::new(&device, graphics_queue, graphics_queue_family_index);

        // ~mesh

        let triangle_mesh_shader =
            shader_manager.get_graphics_shader(&device, "colored_triangle_mesh");

        let buffer_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(std::mem::size_of::<GPUPushDrawConstant>() as u32)];

        let triangle_mesh_pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&buffer_ranges);

        let triangle_mesh_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&triangle_mesh_pipeline_layout_create_info, None)
                .unwrap()
        };

        let mut mesh_pipeline_builder = PipelineBuilder::default();
        #[allow(clippy::field_reassign_with_default)]
        {
            mesh_pipeline_builder.pipeline_layout = triangle_mesh_pipeline_layout;
        }
        mesh_pipeline_builder.set_shaders(triangle_mesh_shader.vert, triangle_mesh_shader.frag);
        mesh_pipeline_builder.set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        mesh_pipeline_builder.set_polygon_mode(vk::PolygonMode::FILL);
        mesh_pipeline_builder.set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE); // TODO: COUNTER_CLOCKWISE
        mesh_pipeline_builder.set_multisampling_none();
        mesh_pipeline_builder.disable_blending();
        mesh_pipeline_builder.disable_depth_test();
        mesh_pipeline_builder.set_color_attachment_format(draw_image.image_format);
        mesh_pipeline_builder.set_depth_format(vk::Format::UNDEFINED);

        let triangle_mesh_pipeline = mesh_pipeline_builder.build_pipeline(&device);

        // ~mesh

        deletion_queue.push(DeletionType::Image(image));
        deletion_queue.push(DeletionType::ImageView(image_view));
        deletion_queue.push(DeletionType::DescriptorSetLayout(
            draw_image_descriptor_layout,
        ));
        deletion_queue.push(DeletionType::PipelineLayout(compute_layout));
        deletion_queue.push(DeletionType::Pipeline(compute_pipeline));
        deletion_queue.push(DeletionType::Pipeline(triangle_pipeline));
        deletion_queue.push(DeletionType::PipelineLayout(triangle_pipeline_layout));
        deletion_queue.push(DeletionType::Pipeline(triangle_mesh_pipeline));
        deletion_queue.push(DeletionType::PipelineLayout(triangle_mesh_pipeline_layout));

        let rect_indices = [0, 1, 2, 2, 1, 3];

        let mut vertex = Vertex::default();
        let mut rect_vertices = vec![];

        vertex.position = vector![0.5, -0.5, 0.0];
        vertex.color = vector![0.0, 0.0, 0.0, 1.0];

        rect_vertices.push(vertex.clone());

        vertex.position = vector![0.5, 0.5, 0.0];
        vertex.color = vector![0.5, 0.5, 0.5, 1.0];

        rect_vertices.push(vertex.clone());

        vertex.position = vector![-0.5, -0.5, 0.0];
        vertex.color = vector![1.0, 0.0, 0.0, 1.0];

        rect_vertices.push(vertex.clone());

        vertex.position = vector![-0.5, 0.5, 0.0];
        vertex.color = vector![0.0, 1.0, 0.0, 1.0];

        rect_vertices.push(vertex);

        let rectangle = Self::upload_mesh(
            &device,
            &mut allocator,
            &immediate_submit,
            &rect_indices,
            &rect_vertices,
        );

        deletion_queue.push(DeletionType::Buffer(rectangle.index_buffer.buffer));
        deletion_queue.push(DeletionType::Buffer(rectangle.vertex_buffer.buffer));

        let mesh_assets = Self::load_gltf_meshes(
            &device,
            &mut allocator,
            &immediate_submit,
            std::path::PathBuf::from(gltf_name).as_path(),
        );

        Self {
            frame_number: 0,
            _stop_rendering: false,
            _window_extent: vk::Extent2D { width, height },

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
            swapchain_extent: render_area.extent,
            frames,
            deletion_queue,
            allocator: Some(allocator),
            draw_image,
            draw_extent: vk::Extent2D::default(),

            shader_manager,

            descriptor_allocator,
            draw_image_descriptors,
            _draw_image_descriptor_layout: draw_image_descriptor_layout,

            gradient_pipeline_layout: compute_layout,
            gradient_pipeline: compute_pipeline,

            triangle_pipeline_layout,
            triangle_pipeline,

            immediate_submit,

            triangle_mesh_pipeline_layout,
            triangle_mesh_pipeline,

            rectangle,

            mesh_assets: mesh_assets.unwrap(),
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

        for mesh_asset in &mut self.mesh_assets {
            mesh_asset.destroy(&self.device, self.allocator.as_mut().unwrap());
        }

        self.shader_manager.destroy(&self.device);

        self.descriptor_allocator.destroy_pool(&self.device);

        self.immediate_submit.destroy(&self.device);

        self.deletion_queue.flush(&self.device);

        self.allocator.take();

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

    fn draw_background(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline_layout,
                0,
                &[self.draw_image_descriptors],
                &[],
            );

            let push_constants = ComputePushConstants {
                data1: vector![1.0, 0.0, 0.0, 1.0],
                data2: vector![0.0, 0.0, 1.0, 1.0],
                data3: vector![0.0, 0.0, 0.0, 1.0],
                data4: vector![0.0, 0.0, 0.0, 1.0],
            };

            self.device.cmd_push_constants(
                cmd,
                self.gradient_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants.as_bytes(),
            );

            self.device.cmd_dispatch(
                cmd,
                f32::ceil(self.draw_extent.width as f32 / 16.0) as u32,
                f32::ceil(self.draw_extent.height as f32 / 16.0) as u32,
                1,
            );
        }
    }

    fn get_projection(aspect_ratio: f32, fov: f32, near: f32, far: f32) -> Matrix4<f32> {
        let projection = Matrix4::new(
            aspect_ratio / (fov / 2.0).tan(),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / (fov / 2.0).tan(),
            0.0,
            0.0,
            0.0,
            0.0,
            far / (far - near),
            1.0,
            0.0,
            0.0,
            -(near * far) / (far - near),
            0.0,
        )
        .transpose();

        let view_to_clip =
            Rotation3::from_axis_angle(&Vector3::x_axis(), 180.0_f32.to_radians()).to_homogeneous();

        projection * view_to_clip
    }

    //#[allow(clippy::too_many_lines)]
    fn draw_geometry(&self, cmd: vk::CommandBuffer) {
        let Self {
            device,
            triangle_pipeline,
            draw_extent,
            triangle_mesh_pipeline_layout,
            triangle_mesh_pipeline,
            rectangle,
            mesh_assets,
            ..
        } = self;

        let color_attachment = [vk_util::attachment_info(
            self.draw_image.image_view,
            None,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )];

        let depth_attachment = vk::RenderingAttachmentInfo::default(); // TODO: unneccesary creation of empty struct

        let rendering_info =
            vk_util::rendering_info(self.draw_extent, &color_attachment, &depth_attachment);

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, *triangle_pipeline);
        }

        let viewports = [vk::Viewport::default()
            .width(draw_extent.width as f32)
            .height(draw_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissors = [vk::Rect2D::default().extent(*draw_extent)];

        unsafe {
            device.cmd_set_viewport(cmd, 0, &viewports);
            device.cmd_set_scissor(cmd, 0, &scissors);
            //device.cmd_draw(cmd, 3, 1, 0, 0);
        }

        let aspect_ratio = draw_extent.height as f32 / draw_extent.width as f32;
        let fov = 90_f32.to_radians();
        let near = 100.0;
        let far = 0.0;

        let projection = Self::get_projection(aspect_ratio, fov, near, far);

        let view = Translation3::new(0.0, 0.0, 0.0).inverse().to_homogeneous();

        let world = Translation3::new(1.0, 2.0, -5.0).to_homogeneous();

        let mut push_constants = GPUPushDrawConstant {
            world_matrix: projection * view * world,
            vertex_buffer: rectangle.vertex_buffer_address,
        };

        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                *triangle_mesh_pipeline,
            );

            device.cmd_push_constants(
                cmd,
                *triangle_mesh_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push_constants.as_bytes(),
            );

            device.cmd_bind_index_buffer(
                cmd,
                rectangle.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            //device.cmd_draw_indexed(cmd, 6, 1, 0, 0, 0);

            push_constants.vertex_buffer = mesh_assets[2].mesh_buffers.vertex_buffer_address;

            device.cmd_push_constants(
                cmd,
                *triangle_mesh_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push_constants.as_bytes(),
            );

            device.cmd_bind_index_buffer(
                cmd,
                mesh_assets[2].mesh_buffers.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            device.cmd_draw_indexed(
                cmd,
                mesh_assets[2].surfaces[0].count,
                1,
                mesh_assets[2].surfaces[0].start_index,
                0,
                0,
            );
        };

        unsafe {
            device.cmd_end_rendering(cmd);
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn draw(&mut self) {
        let current_frame_fence = self.get_current_frame().fence;
        let current_frame_command_buffer = self.get_current_frame().command_buffer;
        let current_frame_swapchain_semaphore = self.get_current_frame().swapchain_semaphore;
        let current_frame_render_semaphore = self.get_current_frame().render_semaphore;
        unsafe {
            self.device
                .wait_for_fences(&[current_frame_fence], true, 1_000_000_000)
                .expect("Failed waiting for fences");

            self.get_current_frame()
                .deletion_queue
                .borrow_mut()
                .flush(&self.device);

            self.device
                .reset_fences(&[current_frame_fence])
                .expect("Failed to reset fences");

            let (swapchain_image_index, _) = self
                .swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    1_000_000_000,
                    current_frame_swapchain_semaphore,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image");

            let cmd = current_frame_command_buffer;
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");

            let cmd_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.draw_extent = self
                .draw_extent
                .width(self.draw_image.image_extent.width)
                .height(self.draw_image.image_extent.height);

            // Begin command buffer

            self.device
                .begin_command_buffer(cmd, &cmd_begin_info)
                .expect("Failed to begin command buffer");

            vk_util::transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            self.draw_background(cmd);

            vk_util::transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            self.draw_geometry(cmd);

            vk_util::transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            let swapchain_image = self.swapchain_images[swapchain_image_index as usize];

            vk_util::transition_image(
                &self.device,
                cmd,
                swapchain_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            vk_util::copy_image_to_image(
                &self.device,
                cmd,
                self.draw_image.image,
                swapchain_image,
                self.draw_extent,
                self.swapchain_extent,
            );

            vk_util::transition_image(
                &self.device,
                cmd,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
                .semaphore(current_frame_swapchain_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .device_index(0)
                .value(1)];

            let signal_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(current_frame_render_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                .device_index(0)
                .value(1)];

            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos)
                .command_buffer_infos(&cmd_infos);

            self.device
                .queue_submit2(self.graphics_queue, &[submit_info], current_frame_fence)
                .expect("Failed to queue submit");

            let swapchains = [self.swapchain];
            let wait_semaphores = [current_frame_render_semaphore];
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

    fn create_allocator(
        instance: Instance,
        device: Device,
        physical_device: vk::PhysicalDevice,
    ) -> Allocator {
        Allocator::new(&AllocatorCreateDesc {
            instance,
            device,
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })
        .unwrap()
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

    fn create_buffer(
        device: &Device,
        allocator: &mut Allocator,
        alloc_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
    ) -> AllocatedBuffer {
        let create_info = vk::BufferCreateInfo::default()
            .size(alloc_size)
            .usage(usage);

        let buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_info = AllocationCreateDesc {
            name: "buffer", // TODO: Proper name
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator.allocate(&allocation_info).unwrap();

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        };

        AllocatedBuffer {
            buffer,
            allocation: Some(allocation),
        }
    }

    // TODO: Background thread, reuse staging
    fn upload_mesh(
        device: &Device,
        allocator: &mut Allocator,
        immediate_submit: &ImmediateSubmit,
        indices: &[u32],
        vertices: &[Vertex],
    ) -> GPUMeshBuffers {
        let index_buffer_size = std::mem::size_of_val(indices) as vk::DeviceSize;
        let vertex_buffer_size = std::mem::size_of_val(vertices) as vk::DeviceSize;

        let index_buffer = Self::create_buffer(
            device,
            allocator,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        );

        let vertex_buffer = Self::create_buffer(
            device,
            allocator,
            vertex_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        );

        let info = vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);
        let vertex_buffer_address = unsafe { device.get_buffer_device_address(&info) };

        // TODO: Allocation separate?

        let mut staging = Self::create_buffer(
            device,
            allocator,
            index_buffer_size + vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );

        unsafe {
            std::ptr::copy(
                indices.as_ptr(),
                staging
                    .allocation
                    .as_ref()
                    .unwrap()
                    .mapped_ptr()
                    .unwrap()
                    .cast()
                    .as_ptr(),
                indices.len(),
            );

            std::ptr::copy(
                vertices.as_ptr(),
                staging
                    .allocation
                    .as_ref()
                    .unwrap()
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr()
                    .cast::<u8>()
                    .add(index_buffer_size as usize)
                    .cast(),
                vertices.len(),
            );
        };

        immediate_submit.submit(device, |cmd| {
            let index_regions = [vk::BufferCopy::default().size(index_buffer_size)];

            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, index_buffer.buffer, &index_regions);
            };

            let vertex_regions = [vk::BufferCopy::default()
                .src_offset(index_buffer_size)
                .size(vertex_buffer_size)];

            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, vertex_buffer.buffer, &vertex_regions);
            }
        });

        staging.destroy(device, allocator);

        self::GPUMeshBuffers {
            index_buffer,
            vertex_buffer,
            vertex_buffer_address,
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn load_gltf_meshes(
        device: &Device,
        allocator: &mut Allocator,
        immediate_submit: &ImmediateSubmit,
        file_path: &std::path::Path,
    ) -> Option<Vec<MeshAsset>> {
        println!("Loading GLTF: {}", file_path.display());

        let (gltf, buffers, _images) = gltf::import(file_path).unwrap();

        let mut mesh_assets = vec![];
        let mut indices: Vec<u32> = vec![];
        let mut vertices: Vec<Vertex> = vec![];

        for mesh in gltf.meshes() {
            indices.clear();
            vertices.clear();
            let mut surfaces: Vec<GeoSurface> = vec![];

            // TODO: Pack same vertexes
            for primitive in mesh.primitives() {
                let start_index = indices.len();
                let count = primitive.indices().unwrap().count(); // ?

                surfaces.push(GeoSurface {
                    start_index: start_index as u32,
                    count: count as u32,
                });

                let initial_vtx = vertices.len();

                // Load indexes

                // TODO: Can this be cleaner?
                let reader = primitive
                    .reader(|buffer| buffers.get(buffer.index()).map(std::ops::Deref::deref));

                indices.reserve(count);

                reader.read_indices().unwrap().into_u32().for_each(|value| {
                    indices.push(value + initial_vtx as u32);
                });

                // Load POSITION
                vertices.reserve(count);

                for position in reader.read_positions().unwrap() {
                    let vertex = Vertex {
                        position: position.into(),
                        uv_x: 0.0,
                        normal: vector![1.0, 0.0, 0.0],
                        uv_y: 0.0,
                        color: Vector4::from_element(1.0),
                    };

                    vertices.push(vertex);
                }

                // Load NORMAL
                if let Some(normals) = reader.read_normals() {
                    let vertices = &mut vertices[initial_vtx..];

                    for (vertex, normal) in vertices.iter_mut().zip(normals.into_iter()) {
                        vertex.normal = normal.into();
                    }
                }

                // Load TEXCOORD_0
                if let Some(tex_coords) = reader.read_tex_coords(0) {
                    let vertices = &mut vertices[initial_vtx..];

                    for (vertex, [x, y]) in vertices.iter_mut().zip(tex_coords.into_f32()) {
                        vertex.uv_x = x;
                        vertex.uv_y = y;
                    }
                }

                // Load COLOR_0
                if let Some(colors) = reader.read_colors(0) {
                    let vertices = &mut vertices[initial_vtx..];

                    for (vertex, color) in vertices.iter_mut().zip(colors.into_rgba_f32()) {
                        vertex.color = color.into();
                    }
                }

                {
                    const OVERRIDE_COLORS: bool = true;
                    if OVERRIDE_COLORS {
                        for vertex in &mut vertices {
                            vertex.color = vertex.normal.push(1.0);
                        }
                    }
                }
            }

            mesh_assets.push(MeshAsset {
                name: mesh.name().unwrap().into(),
                surfaces,
                mesh_buffers: Self::upload_mesh(
                    device,
                    allocator,
                    immediate_submit,
                    &indices,
                    &vertices,
                ),
            });
        }

        Some(mesh_assets)
    }
}
