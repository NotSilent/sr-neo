#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "scene_data.glsl"
#include "input_structures.glsl"

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out vec3 out_frag_position_tbn;
layout (location = 3) out vec3 out_light_direction_tbn;
layout (location = 4) out float out_light_power;
layout (location = 5) out vec3 out_view_position_tbn;

struct Vertex 
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
	vec4 tangent;
}; 

struct UniformData
{
	mat4 world_matrix;
};

layout(buffer_reference, std430) readonly buffer UniformBuffer{
	UniformData uniforms[];
};

layout(buffer_reference, std430) readonly buffer VertexBuffer{
	Vertex vertices[];
};

//push constants block
layout( push_constant ) uniform constants
{
	UniformBuffer uniform_buffer;
	VertexBuffer vertexBuffer;
	uint index;
} PushConstants;

void main() 
{
	UniformData uniform_data = PushConstants.uniform_buffer.uniforms[PushConstants.index + gl_DrawID];
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position, 1.0f);


	out_color = v.color.xyz * material_data.color_factors.xyz;	
	out_uv.x = v.uv_x;
	out_uv.y = v.uv_y;

	vec3 T = normalize(mat3(uniform_data.world_matrix) * v.tangent.xyz);
	vec3 N = normalize(mat3(uniform_data.world_matrix) * v.normal);
	// re-orthogonalize T with respect to N
	T = normalize(T - dot(T, N) * N);
	// then retrieve perpendicular vector B with the cross product of T and N
	vec3 B = cross(N, T) * v.tangent.w;

	// Transforms world-space to TBN space
	mat3 TBN = inverse(mat3(T, B, N));

	// Transform world-space LightDirection to TBN space
	out_frag_position_tbn = TBN * mat3(uniform_data.world_matrix) * position.xyz;
	out_light_direction_tbn = TBN * scene_data.sunlight_direction.xyz;//, 1.0;
	out_light_power = scene_data.sunlight_direction.w;
	out_view_position_tbn = TBN * scene_data.view_position;

	gl_Position =  scene_data.view_proj * uniform_data.world_matrix * position;
}
