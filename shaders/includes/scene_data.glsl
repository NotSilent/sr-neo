layout(set = 0, binding = 0) uniform SceneData{   
	mat4 view;
	mat4 proj;
	mat4 inv_proj;
	mat4 view_proj;
    mat4 light_view;
    mat4 light_view_proj;
	vec4 sunlight_direction; //w for sun power
	vec4 sunlight_color;
	vec3 view_position;
	float padding;
	vec2 screen_size;
} scene_data;