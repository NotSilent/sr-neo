use ash::{Device, vk};

#[derive(Default, Clone)]
pub struct PipelineBuilder<'a> {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo<'a>,
    color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    multisampling: vk::PipelineMultisampleStateCreateInfo<'a>,
    pipeline_layout: vk::PipelineLayout,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    render_info: vk::PipelineRenderingCreateInfo<'a>,
}

impl<'a> PipelineBuilder<'a> {
    pub fn build(mut self, device: &Device) -> vk::Pipeline {
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&self.color_blend_attachments);

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

    pub fn set_shaders(mut self, vertex: vk::ShaderModule, fragment: vk::ShaderModule) -> Self {
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

        self
    }

    pub fn set_input_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.input_assembly.topology = topology;
        self.input_assembly.primitive_restart_enable = vk::FALSE;

        self
    }

    pub fn set_polygon_mode(mut self, polygon_mode: vk::PolygonMode) -> Self {
        self.rasterizer.polygon_mode = polygon_mode;
        self.rasterizer.line_width = 1.0;

        self
    }

    pub fn set_cull_mode(
        mut self,
        cull_mode: vk::CullModeFlags,
        front_face: vk::FrontFace,
    ) -> Self {
        self.rasterizer.cull_mode = cull_mode;
        self.rasterizer.front_face = front_face;

        self
    }

    pub fn set_multisampling_none(mut self) -> Self {
        let Self { multisampling, .. } = &mut self;

        multisampling.sample_shading_enable = vk::FALSE;
        multisampling.rasterization_samples = vk::SampleCountFlags::TYPE_1;
        multisampling.min_sample_shading = 1.0;
        multisampling.alpha_to_coverage_enable = vk::FALSE;
        multisampling.alpha_to_one_enable = vk::FALSE;

        self
    }

    pub fn set_pipeline_layout(mut self, pipeline_layout: vk::PipelineLayout) -> Self {
        self.pipeline_layout = pipeline_layout;

        self
    }

    pub fn add_attachment(mut self) -> Self {
        let attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        self.color_blend_attachments.push(attachment);

        self
    }

    // pub fn _enable_blending_additive(mut self) -> Self {
    //     let Self {
    //         color_blend_attachment,
    //         ..
    //     } = &mut self;

    //     color_blend_attachment.color_write_mask = vk::ColorComponentFlags::RGBA;
    //     color_blend_attachment.blend_enable = vk::TRUE;
    //     color_blend_attachment.src_color_blend_factor = vk::BlendFactor::SRC_ALPHA;
    //     color_blend_attachment.dst_color_blend_factor = vk::BlendFactor::ONE;
    //     color_blend_attachment.color_blend_op = vk::BlendOp::ADD;
    //     color_blend_attachment.src_alpha_blend_factor = vk::BlendFactor::ONE;
    //     color_blend_attachment.dst_alpha_blend_factor = vk::BlendFactor::ZERO;
    //     color_blend_attachment.alpha_blend_op = vk::BlendOp::ADD;

    //     self
    // }

    pub fn add_blend_attachment(mut self) -> Self {
        let attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        self.color_blend_attachments.push(attachment);

        self
    }

    pub fn set_color_attachment_formats(mut self, formats: &'a [vk::Format]) -> Self {
        self.render_info = self.render_info.color_attachment_formats(formats);

        self
    }

    pub fn set_depth_format(mut self, format: vk::Format) -> Self {
        self.render_info.depth_attachment_format = format;

        self
    }

    pub fn disable_depth_test(mut self) -> Self {
        let Self { depth_stencil, .. } = &mut self;

        depth_stencil.depth_test_enable = vk::FALSE;
        depth_stencil.depth_write_enable = vk::FALSE;
        depth_stencil.depth_compare_op = vk::CompareOp::NEVER;
        depth_stencil.stencil_test_enable = vk::FALSE;
        depth_stencil.front = vk::StencilOpState::default();
        depth_stencil.back = vk::StencilOpState::default();
        depth_stencil.min_depth_bounds = 0.0;
        depth_stencil.max_depth_bounds = 1.0;

        self
    }

    pub fn enable_depth_test(
        mut self,
        depth_write_enable: vk::Bool32,
        compare_op: vk::CompareOp,
    ) -> Self {
        let Self { depth_stencil, .. } = &mut self;

        depth_stencil.depth_test_enable = vk::TRUE;
        depth_stencil.depth_write_enable = depth_write_enable;
        depth_stencil.depth_compare_op = compare_op;
        depth_stencil.stencil_test_enable = vk::FALSE;
        depth_stencil.front = vk::StencilOpState::default();
        depth_stencil.back = vk::StencilOpState::default();
        depth_stencil.min_depth_bounds = 0.0;
        depth_stencil.max_depth_bounds = 1.0;

        self
    }
}
