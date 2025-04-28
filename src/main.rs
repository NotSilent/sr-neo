use egui_winit::egui::{ViewportBuilder, ViewportInfo};
use egui_winit::{State, egui};
use vulkan_engine::VulkanEngine;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod deletion_queue;
mod shader_manager;
mod vk_util;
mod vulkan_engine;

#[derive(Default)]
struct App {
    state: Option<State>,
    window: Option<Window>,
    viewport_info: Option<ViewportInfo>,
    vulkan_engine: Option<VulkanEngine>,
    is_closing: bool,
}

impl ApplicationHandler for App {
    // This is a common indicator that you can create a window.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let width = 1920;
        let height = 1080;

        // if let Ok(window) = event_loop.create_window(
        //     Window::default_attributes().with_inner_size(PhysicalSize::new(width, height)),
        // ) {
        //     self.vulkan_engine = Some(VulkanEngine::new(
        //         window.display_handle().unwrap().as_raw(),
        //         window.window_handle().unwrap().as_raw(),
        //         width,
        //         height,
        //     ));
        //     self.window = Some(window);
        // }

        let egui_ctx = egui::Context::default();

        let window = egui_winit::create_window(
            &egui_ctx,
            event_loop,
            &ViewportBuilder::default()
                .with_inner_size(egui::Vec2::new(width as f32, height as f32)),
        )
        .unwrap();

        let state = State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            event_loop,
            None,
            None,
            None,
        );

        let mut viewport_info = ViewportInfo::default();

        egui_winit::update_viewport_info(&mut viewport_info, state.egui_ctx(), &window, true);

        let vulkan_engine = VulkanEngine::new(
            window.display_handle().unwrap().as_raw(),
            window.window_handle().unwrap().as_raw(),
            width,
            height,
        );

        self.state = Some(state);
        self.window = Some(window);
        self.viewport_info = Some(viewport_info);
        self.vulkan_engine = Some(vulkan_engine);
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
                    let viewport_info = self.viewport_info.as_mut().unwrap();
                    let state = self.state.as_mut().unwrap();
                    let window = self.window.as_ref().unwrap();

                    egui_winit::update_viewport_info(
                        viewport_info,
                        state.egui_ctx(),
                        window,
                        false,
                    );

                    let raw_input = state.take_egui_input(window);

                    let full_output = state.egui_ctx().run(raw_input, |ctx| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            ui.label("Hello world!");
                            if ui.button("Click me").clicked() {
                                // take some action here
                            }
                        });
                    });

                    state.handle_platform_output(window, full_output.platform_output);

                    let _clipped_primitives = state
                        .egui_ctx()
                        .tessellate(full_output.shapes, full_output.pixels_per_point);

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
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::default();
    let _ = event_loop.run_app(&mut app);
}
