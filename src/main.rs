use vulkan_engine::VulkanEngine;
use winit::dpi::PhysicalSize;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod vulkan_engine;

#[derive(Default)]
struct State {
    window: Option<Window>,
    vulkan_engine: Option<VulkanEngine>,
    is_closing: bool,
}

impl ApplicationHandler for State {
    // This is a common indicator that you can create a window.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let width = 1920;
        let height = 1080;

        if let Ok(window) = event_loop.create_window(
            Window::default_attributes().with_inner_size(PhysicalSize::new(width, height)),
        ) {
            self.vulkan_engine = Some(VulkanEngine::new(
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                width,
                height,
            ));
            self.window = Some(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(new_size) => {
                if new_size.width == 0 || new_size.height == 0 {
                    // TODO: Stop rendering
                } else {
                    // TODO: Restart rendering
                }
            }
            WindowEvent::CloseRequested => {
                self.is_closing = true;

                if let Some(vulkan_engine) = &mut self.vulkan_engine {
                    vulkan_engine.cleanup();
                }

                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if !self.is_closing {
                    self.vulkan_engine.as_mut().unwrap().draw();
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut state = State::default();
    let _ = event_loop.run_app(&mut state);
}
