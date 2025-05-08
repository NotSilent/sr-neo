use ash::{Device, vk};

use crate::vk_util;

pub struct ImmediateSubmit {
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
