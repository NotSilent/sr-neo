use nalgebra::{Matrix4, Rotation3, Translation3, Vector2, Vector3, Vector4, vector};
use winit::{event::ElementState, keyboard::KeyCode};

// Move out of camera

#[derive(Default)]
pub struct InputManager {
    pub mouse_delta: Vector2<f64>,
    pub key_events: Vec<KeyEvent>,
}

impl InputManager {
    // TODO: keys should be kept until they are released
    pub fn clear(&mut self) {
        self.mouse_delta = vector![0.0, 0.0];
        self.key_events.clear();
    }
}

pub struct KeyEvent {
    pub key_code: KeyCode,
    pub element_state: ElementState,
}

#[derive(Default)]
pub struct Camera {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,

    pub pitch: f32,
    pub yaw: f32,
}

impl Camera {
    pub fn get_view(&self) -> Matrix4<f32> {
        let camera_translation = Translation3::from(self.position).to_homogeneous();
        let camera_rotation = self.get_rotation_matrix();

        (camera_translation * camera_rotation)
            .try_inverse()
            .unwrap()
    }

    pub fn get_rotation_matrix(&self) -> Matrix4<f32> {
        let pitch_rotation =
            Rotation3::from_axis_angle(&Vector3::x_axis(), self.pitch).to_homogeneous();
        let yaw_rotation =
            Rotation3::from_axis_angle(&-Vector3::y_axis(), self.yaw).to_homogeneous();

        yaw_rotation * pitch_rotation
    }

    pub fn process_winit_events(&mut self, input_manager: &InputManager) {
        let speed = 0.1;

        for key_event in &input_manager.key_events {
            match key_event.element_state {
                ElementState::Pressed => match key_event.key_code {
                    KeyCode::KeyA => self.velocity.x = -speed,
                    KeyCode::KeyD => self.velocity.x = 1.0 * speed,
                    KeyCode::KeyS => self.velocity.z = 1.0 * speed,
                    KeyCode::KeyW => self.velocity.z = -speed,
                    _ => {}
                },
                ElementState::Released => match key_event.key_code {
                    KeyCode::KeyD | KeyCode::KeyA => self.velocity.x = 0.0,
                    KeyCode::KeyS | KeyCode::KeyW => self.velocity.z = 0.0,
                    _ => {}
                },
            }
        }

        // TODO: delta_time
        self.yaw += (input_manager.mouse_delta.x / 50.0) as f32;
        self.pitch -= (input_manager.mouse_delta.y / 50.0) as f32;
    }

    pub fn update(&mut self) {
        let camera_rotation = self.get_rotation_matrix();
        let velocity: Vector4<f32> = self.velocity.push(0.0);
        let displacement: Vector4<f32> = camera_rotation * velocity;
        let displacement: Vector3<f32> = vector![displacement.x, displacement.y, displacement.z];
        self.position += displacement;
    }

    pub fn get_projection(aspect_ratio: f32) -> Matrix4<f32> {
        let fov = 90_f32.to_radians();
        let near = 1000.0;
        let far = 0.01;

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
}
