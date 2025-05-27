#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "scene_data.glsl"

layout (location = 0) out vec2 out_uv;
layout (location = 1) out uint out_material_index;

layout( push_constant ) uniform constants
{
	uint index;
} PushConstants;

void main() 
{
	Uniform uniform_data = uniform_data.uniforms[PushConstants.index + gl_DrawID];
	Vertex v = vertex_data.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position, 1.0f);

	out_uv.x = v.uv_x;
	out_uv.y = v.uv_y;
	out_material_index = uniform_data.material_data_index;

	gl_Position =  scene_data.view_proj * uniform_data.world_matrix * position;
}
