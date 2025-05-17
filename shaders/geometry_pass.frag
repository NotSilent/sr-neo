#version 460

#extension GL_GOOGLE_include_directive : require

#include "input_structures.glsl"
#include "pbr.glsl"

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_frag_position_tbn;
layout (location = 3) in vec3 in_light_direction_tbn; //
layout (location = 4) in float in_light_power; //
layout (location = 5) in vec3 in_view_position_tbn;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec4 out_normal;

// PBR calculations from: https://learnopengl.com/PBR/Lighting
// Modified to be in TBH space
void main()
{
	vec4 color = texture(color_tex,in_uv) * material_data.color_factors;
	vec4 normal = texture(normal_tex,in_uv);

	//TODO: This is temp for master material, should be different shader for masked
	if (color.w < 0.5) {
		discard;
	}
    
    out_color = color,
    out_normal = normal;
}
