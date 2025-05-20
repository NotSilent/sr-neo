#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "scene_data.glsl"
#include "input_structures.glsl"

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

	gl_Position =  scene_data.light_view_proj * uniform_data.world_matrix * position;
}
