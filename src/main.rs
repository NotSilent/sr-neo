use std::{env, f32};

use camera::{InputManager, KeyEvent};
use egui_winit::egui::{ViewportBuilder, ViewportInfo};
use egui_winit::{State, egui};
use nalgebra::{Vector3, vector};
use vulkan_engine::VulkanEngine;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use crate::camera::Camera;
use crate::draw::DrawContext;
use crate::gltf_loader::GLTFLoader;

mod buffers;
mod camera;
mod default_resources;
mod depth_pre_pass;
mod descriptors;
mod double_buffer;
mod draw;
mod forward_pass;
mod fxaa_pass;
mod geometry_pass;
mod gltf_loader;
mod images;
mod immediate_submit;
mod lightning_pass;
mod materials;
mod meshes;
mod pipeline_builder;
mod renderpass_common;
mod resource_manager;
mod shader_manager;
mod shadow_map_pass;
mod swapchain;
mod vk_init;
mod vk_util;
mod vulkan_engine;

struct App<'a> {
    state: Option<State>,
    window: Option<Window>,
    viewport_info: Option<ViewportInfo>,
    vulkan_engine: Option<VulkanEngine>,
    is_minimized: bool,
    is_closing: bool,

    gltf_name: Option<&'a str>,

    input_manager: InputManager,

    cpu_time: std::time::Instant,

    main_camera: Camera,
    draw_context: DrawContext,
    gltf_loader: GLTFLoader,
}

impl Default for App<'_> {
    fn default() -> Self {
        let main_camera = Camera {
            position: vector![-6.0, 4.0, 0.0],
            velocity: Vector3::from_element(0.0),
            pitch: f32::consts::PI / -10.0,
            yaw: f32::consts::FRAC_PI_2,
        };

        Self {
            state: None,
            window: None,
            viewport_info: None,
            vulkan_engine: None,
            is_minimized: false,
            is_closing: false,
            gltf_name: None,
            input_manager: InputManager::default(),
            cpu_time: std::time::Instant::now(),
            main_camera,
            draw_context: DrawContext::default(),
            gltf_loader: GLTFLoader::default(),
        }
    }
}

impl<'a> App<'a> {
    fn set_gltf_to_load(&mut self, gltf_name: &'a str) {
        self.gltf_name = Some(gltf_name);
    }

    fn update_scene(&mut self) {
        self.main_camera.update();
        self.draw_context.render_objects.clear();

        self.gltf_loader.draw(&mut self.draw_context);
    }
}

impl ApplicationHandler for App<'_> {
    // This is a common indicator that you can create a window.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let width = 1600;
        let height = 900;

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

        let mut vulkan_engine = VulkanEngine::new(
            window.display_handle().unwrap().as_raw(),
            window.window_handle().unwrap().as_raw(),
            window.inner_size().width,
            window.inner_size().height,
        );

        // TODO: drop unwrap
        self.gltf_loader.load(
            &mut vulkan_engine,
            std::path::PathBuf::from(&self.gltf_name.unwrap()).as_path(),
        );

        vulkan_engine.update_images();

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
                        let time_now = std::time::Instant::now();

                        self.update_scene();

                        let gpu_stats = if let Some(engine) = &mut self.vulkan_engine {
                            self.main_camera.process_winit_events(&self.input_manager);

                            match engine.draw(&self.main_camera, &self.draw_context, 1.0) {
                                Ok(gpu_stats) => gpu_stats,
                                Err(error) => match error {
                                    vulkan_engine::DrawError::Swapchain(_swapchain_error) => {
                                        let size = self.window.as_mut().unwrap().inner_size();
                                        engine.recreate_swapchain(size.width, size.height);

                                        if let Ok(gpu_stats) =
                                            engine.draw(&self.main_camera, &self.draw_context, 1.0)
                                        {
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

                        let cpu_time = self.cpu_time.elapsed().as_secs_f64() * 1000.0;
                        self.cpu_time = std::time::Instant::now();

                        let cpu_record_time = time_now.elapsed().as_secs_f64() * 1000.0;

                        // Result from previous frame
                        let fps = 1000.0 / gpu_stats.draw_time;

                        self.window.as_mut().unwrap().set_title(&format!(
                            "sr-neo: CPU: {:.2} Record: {:.2} GPU: {:.2} DrawCalls: {} Triangles: {} FPS: {:.0}",
                            cpu_time,
                            // TODO: Technicaly includes input manager update time
                            cpu_record_time,
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

                if let PhysicalKey::Code(key_code) = event.physical_key
                    && key_code == KeyCode::Escape
                {
                    self.is_closing = true;

                    self.vulkan_engine.take();

                    event_loop.exit();
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
