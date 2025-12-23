use ash::vk;

use crate::{vk_util, vulkan_engine::VulkanContext};

pub struct ImmediateSubmit {
    fence: vk::Fence,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}

impl ImmediateSubmit {
    pub fn new(ctx: &VulkanContext) -> Self {
        let pool = vk_util::create_command_pool(ctx);
        Self {
            fence: vk_util::create_fence(ctx, vk::FenceCreateFlags::SIGNALED),
            pool,
            cmd: vk_util::allocate_command_buffer(ctx, pool),
        }
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.destroy_fence(self.fence, None);
            ctx.destroy_command_pool(self.pool, None);
        }
    }

    // TODO: PRIORITY: It's getting annoying to pass it everywhere and keep track of when uploads should be made
    // Instead keep track of which buffers/images should be copied with src + dst and upload at once before frame begins
    pub fn submit<F: Fn(vk::CommandBuffer)>(&self, ctx: &VulkanContext, record: F) {
        unsafe {
            ctx.reset_fences(&[self.fence]).unwrap();
            ctx.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            ctx.begin_command_buffer(self.cmd, &begin_info).unwrap();

            record(self.cmd);

            ctx.end_command_buffer(self.cmd).unwrap();

            let cmd_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(self.cmd)
                .device_mask(0)];

            let submit_infos = [vk::SubmitInfo2::default().command_buffer_infos(&cmd_infos)];

            ctx.queue_submit2(ctx.graphics_queue, &submit_infos, self.fence)
                .unwrap();

            ctx.wait_for_fences(&[self.fence], true, 1_000_000_000)
                .unwrap();
        }
    }
}
