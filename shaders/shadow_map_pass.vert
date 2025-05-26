#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "scene_data.glsl"

layout( push_constant ) uniform constants
{
	uint index;
} PushConstants;

void main() 
{
	Uniform uniform_data = uniform_data.uniforms[PushConstants.index + gl_DrawID];
	Vertex v = vertex_data.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position, 1.0f);

	gl_Position =  scene_data.light_view_proj * uniform_data.world_matrix * position;
}
