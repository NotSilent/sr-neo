struct Vertex 
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
	vec4 tangent;
};

struct Uniform
{
	mat4 world_matrix;
	uint material_data_index;
	// vec3 padding;
};

struct Material
{
	vec4 color_factors;
	vec4 metal_rough_factors;

	uint color_tex_index;
	uint normal_tex_index;
	uint metal_rough_tex_index;
	uint padding;
};

layout (set = 0, binding = 0) uniform SceneData {   
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

layout (set = 0, binding = 1) readonly buffer VertexData {
	Vertex vertices[];
} vertex_data;

layout (set = 0, binding = 2) readonly buffer UniformData {
	Uniform uniforms[];
} uniform_data;

layout (set = 0, binding = 3) readonly buffer MaterialData2 {
	Material materials[];
} material_data2;

layout (set = 0, binding = 4) uniform sampler2D samplers[];