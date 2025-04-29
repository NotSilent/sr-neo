use ash::{Device, vk};

#[derive(Default)]
pub struct PipelineBuilder<'a> {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo<'a>,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState, // TODO: multiple
    multisampling: vk::PipelineMultisampleStateCreateInfo<'a>,
    pub pipeline_layout: vk::PipelineLayout,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    render_info: vk::PipelineRenderingCreateInfo<'a>,
    color_attachment_format: vk::Format,
}

impl PipelineBuilder<'_> {
    pub fn build_pipeline(&mut self, device: &Device) -> vk::Pipeline {
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let color_blend_attachments = [self.color_blend_attachment];
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments);

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut self.render_info)
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.pipeline_layout)
            .dynamic_state(&dynamic_info);

        unsafe {
            *device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()
                .first()
                .unwrap()
        }
    }

    pub fn set_shaders(&mut self, vertex: vk::ShaderModule, fragment: vk::ShaderModule) {
        self.shader_stages.clear();

        self.shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(c"main"),
        );
        self.shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(c"main"),
        );
    }

    pub fn set_input_topology(&mut self, topology: vk::PrimitiveTopology) {
        // TODO: set directly
        self.input_assembly.topology = topology;
        self.input_assembly.primitive_restart_enable = vk::FALSE;
    }

    pub fn set_polygon_mode(&mut self, polygon_mode: vk::PolygonMode) {
        self.rasterizer.polygon_mode = polygon_mode;
        self.rasterizer.line_width = 1.0;
    }

    pub fn set_cull_mode(&mut self, cull_mode: vk::CullModeFlags, front_face: vk::FrontFace) {
        self.rasterizer.cull_mode = cull_mode;
        self.rasterizer.front_face = front_face;
    }

    pub fn set_multisampling_none(&mut self) {
        let Self { multisampling, .. } = self;
        multisampling.sample_shading_enable = vk::FALSE;
        multisampling.rasterization_samples = vk::SampleCountFlags::TYPE_1;
        multisampling.min_sample_shading = 1.0;
        multisampling.alpha_to_coverage_enable = vk::FALSE;
        multisampling.alpha_to_one_enable = vk::FALSE;
    }

    pub fn disable_blending(&mut self) {
        self.color_blend_attachment.color_write_mask = vk::ColorComponentFlags::RGBA;
        self.color_blend_attachment.blend_enable = vk::FALSE;
    }

    pub fn set_color_attachment_format(&mut self, format: vk::Format) {
        self.color_attachment_format = format;
        let _ = self.render_info.color_attachment_formats(&[format]);
    }

    pub fn set_depth_format(&mut self, format: vk::Format) {
        self.render_info.depth_attachment_format = format;
    }

    pub fn disable_depth_test(&mut self) {
        let Self { depth_stencil, .. } = self;

        depth_stencil.depth_test_enable = vk::FALSE;
        depth_stencil.depth_write_enable = vk::FALSE;
        depth_stencil.depth_compare_op = vk::CompareOp::NEVER;
        depth_stencil.stencil_test_enable = vk::FALSE;
        depth_stencil.front = vk::StencilOpState::default();
        depth_stencil.back = vk::StencilOpState::default();
        depth_stencil.min_depth_bounds = 0.0;
        depth_stencil.max_depth_bounds = 1.0;
    }
}
