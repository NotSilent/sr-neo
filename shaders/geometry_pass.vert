#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "scene_data.glsl"
#include "input_structures.glsl"

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out mat3 out_tbn_to_view;
layout (location = 5) out uint out_material_index;

//push constants block
layout( push_constant ) uniform constants
{
	uint index;
} PushConstants;

void main() 
{
	Uniform uniform_data = uniform_data.uniforms[PushConstants.index + gl_DrawID];
	Vertex v = vertex_data.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position, 1.0f);

	out_color = v.color.xyz * material_data.color_factors.xyz;	
	out_uv.x = v.uv_x;
	out_uv.y = v.uv_y;
	out_material_index = uniform_data.material_data_index;

	vec3 T = normalize(mat3(uniform_data.world_matrix) * v.tangent.xyz);
	vec3 N = normalize(mat3(uniform_data.world_matrix) * v.normal);
	// re-orthogonalize T with respect to N
	T = normalize(T - dot(T, N) * N);
	// then retrieve perpendicular vector B with the cross product of T and N
	vec3 B = cross(N, T) * v.tangent.w;

	mat3 TBN = mat3(T, B, N);

    out_tbn_to_view = mat3(scene_data.view) * TBN;

	gl_Position =  scene_data.view_proj * uniform_data.world_matrix * position;
}
