use ash::{Device, vk};

struct DescriptorBufferInfo {
    binding: u32,
    descriptor_type: vk::DescriptorType,
    buffer_info: vk::DescriptorBufferInfo,
}

struct DescriptorImageInfo {
    binding: u32,
    descriptor_type: vk::DescriptorType,
    image_info: vk::DescriptorImageInfo,
}

#[derive(Default)]
pub struct DescriptorWriter {
    buffer_infos: Vec<DescriptorBufferInfo>,
    image_infos: Vec<DescriptorImageInfo>,
}
impl DescriptorWriter {
    pub fn write_buffer(
        &mut self,
        binding: u32,
        buffer: vk::Buffer,
        size: u64,
        offset: u64,
        descriptor_type: vk::DescriptorType,
    ) {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(size);

        self.buffer_infos.push(DescriptorBufferInfo {
            binding,
            descriptor_type,
            buffer_info,
        });
    }

    pub fn write_image(
        &mut self,
        binding: u32,
        sampler: vk::Sampler,
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
        descriptor_type: vk::DescriptorType,
    ) {
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(image_view)
            .image_layout(image_layout);

        self.image_infos.push(DescriptorImageInfo {
            binding,
            descriptor_type,
            image_info,
        });
    }

    // TODO: refactor to create_set, allocate and return the new set
    pub fn update_set(&mut self, device: &Device, set: vk::DescriptorSet) {
        let mut writes = vec![];

        for buffer_info in &self.buffer_infos {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_binding(buffer_info.binding)
                .dst_set(set)
                .descriptor_type(buffer_info.descriptor_type);

            write.descriptor_count = 1;
            // TODO: How is this safe?
            write.p_buffer_info = &raw const buffer_info.buffer_info;

            writes.push(write);
        }

        for image_info in &self.image_infos {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_binding(image_info.binding)
                .dst_set(set)
                .descriptor_type(image_info.descriptor_type);

            write.descriptor_count = 1;
            write.p_image_info = &raw const image_info.image_info;

            writes.push(write);
        }

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        self.buffer_infos.clear();
        self.image_infos.clear();
    }
}

pub struct PoolSizeRatio {
    pub descriptor_type: vk::DescriptorType,
    pub ratio: u32,
}

// TODO: Better? idea
// track fullness per descriptor type
// one vec for full pools, only the biggest one is left after update
pub struct DescriptorAllocatorGrowable {
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}
impl DescriptorAllocatorGrowable {
    pub fn new(device: &Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) -> Self {
        let new_pool = Self::create_pool(device, max_sets, &pool_ratios);

        Self {
            ratios: pool_ratios,
            full_pools: vec![],
            ready_pools: vec![new_pool],
            sets_per_pool: max_sets * 2,
        }
    }

    pub fn destroy(&self, device: &Device) {
        for pool in &self.ready_pools {
            unsafe {
                device.destroy_descriptor_pool(*pool, None);
            }
        }

        for pool in &self.full_pools {
            unsafe {
                device.destroy_descriptor_pool(*pool, None);
            }
        }
    }

    pub fn clear_pools(&mut self, device: &Device) {
        for pool in &self.ready_pools {
            unsafe {
                device
                    .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())
                    .unwrap();
            }
        }

        for pool in &self.full_pools {
            unsafe {
                device
                    .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())
                    .unwrap();
            }

            self.ready_pools.push(*pool);
        }

        self.full_pools.clear();
    }

    pub fn allocate(
        &mut self,
        device: &Device,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let pool = self.get_pool(device);

        let layouts = [layout];
        let mut alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(set) => {
                self.ready_pools.push(pool);
                *set.first().unwrap()
            }
            Err(error) => {
                if error == vk::Result::ERROR_OUT_OF_POOL_MEMORY
                    || error == vk::Result::ERROR_FRAGMENTED_POOL
                {
                    self.full_pools.push(pool);

                    let pool = self.get_pool(device);
                    self.ready_pools.push(pool);

                    alloc_info.descriptor_pool = pool;

                    unsafe {
                        *device
                            .allocate_descriptor_sets(&alloc_info)
                            .unwrap()
                            .first()
                            .unwrap()
                    }
                } else {
                    panic!();
                }
            }
        }
    }

    fn get_pool(&mut self, device: &Device) -> vk::DescriptorPool {
        if let Some(pool) = self.ready_pools.pop() {
            return pool;
        }

        self.sets_per_pool = (self.sets_per_pool * 2).min(4092);

        Self::create_pool(device, self.sets_per_pool, &self.ratios)
    }

    fn create_pool(
        device: &Device,
        set_count: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> vk::DescriptorPool {
        let mut pool_sizes = vec![];
        for pool_ratio in pool_ratios {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(pool_ratio.descriptor_type)
                    .descriptor_count(pool_ratio.ratio * set_count),
            );
        }

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(set_count)
            .pool_sizes(&pool_sizes);

        unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
    }
}

#[derive(Default)]
pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl DescriptorLayoutBuilder<'_> {
    // TODO: Drop binding and increment?
    pub fn add_binding(mut self, binding: u32, descriptor_type: vk::DescriptorType) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_count(1)
                .descriptor_type(descriptor_type),
        );

        self
    }

    pub fn build(
        mut self,
        device: &Device,
        shader_stages: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayout {
        for binding in &mut self.bindings {
            binding.stage_flags |= shader_stages;
        }

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&self.bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        }
    }
}
