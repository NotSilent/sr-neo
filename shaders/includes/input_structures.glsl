layout(set = 0, binding = 0) uniform SceneData{   
	mat4 view;
	mat4 proj;
	mat4 inv_proj;
	mat4 view_proj;
	vec4 ambient_color;
	vec4 sunlight_direction; //w for sun power
	vec4 sunlight_color;
	vec3 view_position;
	float padding;
	vec2 screen_size;
} sceneData;

layout(set = 1, binding = 0) uniform MaterialData{   
	vec4 color_factors;
	vec4 metal_rough_factors;
	
} material_data;

layout(set = 1, binding = 1) uniform sampler2D color_tex;
layout(set = 1, binding = 2) uniform sampler2D normal_tex;
layout(set = 1, binding = 3) uniform sampler2D metal_rough_tex;
