use std::env;

use camera::{InputManager, KeyEvent};
use egui_winit::egui::{ViewportBuilder, ViewportInfo};
use egui_winit::{State, egui};
use nalgebra::vector;
use vulkan_engine::VulkanEngine;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod buffers;
mod camera;
mod default_resources;
mod descriptors;
mod double_buffer;
mod gltf_loader;
mod images;
mod immediate_submit;
mod materials;
mod meshes;
mod pipeline_builder;
mod resource_manager;
mod shader_manager;
mod swapchain;
mod vk_util;
mod vulkan_engine;

#[derive(Default)]
struct App<'a> {
    state: Option<State>,
    window: Option<Window>,
    viewport_info: Option<ViewportInfo>,
    vulkan_engine: Option<VulkanEngine>,
    is_minimized: bool,
    is_closing: bool,

    gltf_name: Option<&'a str>,

    input_manager: InputManager,
}

impl<'a> App<'a> {
    fn set_gltf_to_load(&mut self, gltf_name: &'a str) {
        self.gltf_name = Some(gltf_name);
    }
}

impl ApplicationHandler for App<'_> {
    // This is a common indicator that you can create a window.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let width = 1920;
        let height = 1080;

        let egui_ctx = egui::Context::default();

        let window = egui_winit::create_window(
            &egui_ctx,
            event_loop,
            &ViewportBuilder::default()
                .with_title("sr-neo")
                .with_inner_size(egui::Vec2::new(width as f32, height as f32))
                .with_resizable(true),
        )
        .unwrap();

        window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .unwrap_or(());
        window.set_cursor_visible(false);

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
            self.gltf_name.unwrap(),
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
                self.is_minimized = new_size.width == 0 || new_size.height == 0;

                self.vulkan_engine
                    .as_mut()
                    .unwrap()
                    .recreate_swapchain(new_size.width, new_size.height);
            }
            WindowEvent::CloseRequested => {
                self.is_closing = true;

                self.vulkan_engine.take();

                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if !self.is_closing {
                    if !self.is_minimized {
                        // let viewport_info = self.viewport_info.as_mut().unwrap();
                        // let state = self.state.as_mut().unwrap();
                        // let window = self.window.as_ref().unwrap();

                        // egui_winit::update_viewport_info(
                        //     viewport_info,
                        //     state.egui_ctx(),
                        //     window,
                        //     false,
                        // );

                        // let raw_input = state.take_egui_input(window);

                        // let full_output = state.egui_ctx().run(raw_input, |ctx| {
                        //     egui::CentralPanel::default().show(ctx, |ui| {
                        //         ui.label("Hello world!");
                        //         if ui.button("Click me").clicked() {
                        //             // take some action here
                        //         }
                        //     });
                        // });

                        // state.handle_platform_output(window, full_output.platform_output);

                        // let _clipped_primitives = state
                        //     .egui_ctx()
                        //     .tessellate(full_output.shapes, full_output.pixels_per_point);

                        let time_now = std::time::Instant::now();

                        let gpu_stats = if let Some(engine) = &mut self.vulkan_engine {
                            engine.update(&self.input_manager);
                            match engine.draw(1.0) {
                                Ok(gpu_stats) => gpu_stats,
                                Err(error) => match error {
                                    vulkan_engine::DrawError::Swapchain(_swapchain_error) => {
                                        let size = self.window.as_mut().unwrap().inner_size();
                                        engine.recreate_swapchain(size.width, size.height);

                                        if let Ok(gpu_stats) = engine.draw(1.0) {
                                            gpu_stats
                                        } else {
                                            println!("Swapchain broken beyond repair");
                                            panic!();
                                        }
                                    }
                                },
                            }
                        } else {
                            println!("No engine");
                            panic!();
                        };

                        // TODO: Should count since last redraw
                        let cpu_time = time_now.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / gpu_stats.draw_time;

                        self.window.as_mut().unwrap().set_title(&format!(
                            "sr-neo: CPU: {:.2} GPU: {:.2} DrawCalls: {} Triangles: {} FPS: {:.0}",
                            cpu_time,
                            gpu_stats.draw_time,
                            gpu_stats.draw_calls,
                            gpu_stats.triangles,
                            fps
                        ));

                        self.input_manager.clear();
                    }

                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    self.input_manager.key_events.push(KeyEvent {
                        key_code,
                        element_state: event.state,
                    });
                }

                if let PhysicalKey::Code(key_code) = event.physical_key {
                    if key_code == KeyCode::Escape {
                        self.is_closing = true;

                        self.vulkan_engine.take();

                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            // TODO: For some reason only getting whole numbers
            self.input_manager.mouse_delta = vector![delta.0, delta.1];
            // println!("{:.4}, {:.4}", delta.0, delta.1);
        }
    }
}

fn main() {
    let env_args: Vec<String> = env::args().collect();

    if let Some(gltf_name) = env_args.get(1) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

        let mut app = App::default();
        app.set_gltf_to_load(gltf_name);

        let _ = event_loop.run_app(&mut app);
    }
}
