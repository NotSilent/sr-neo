#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out mat3 out_tbn_to_view;

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

	mat3 TBN = mat3(T, B, N);

    out_tbn_to_view = mat3(sceneData.view) * TBN;

	gl_Position =  sceneData.view_proj * uniform_data.world_matrix * position;
}
