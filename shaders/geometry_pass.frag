#version 460

#extension GL_GOOGLE_include_directive : require

#include "input_structures.glsl"

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in mat3 in_tbn_to_view;

layout (location = 0) out vec4 out_color;  // color + metallic
layout (location = 1) out vec4 out_normal; // normal + roughness

void main()
{
	vec4 color = pow(texture(color_tex,in_uv), vec4(2.2));

	//TODO: This is temp for master material, should be different shader for masked
	if (color.w < 0.5) {
		discard;
	}

	vec3 albedo = color.rgb * in_color;
	vec3 normal = texture(normal_tex,in_uv).rgb * 2.0 - 1.0;
	vec4 metal_rough = texture(metal_rough_tex,in_uv);
	
	vec3 view_space_normal = normalize(in_tbn_to_view * normal);
    
    out_color = vec4(albedo, metal_rough.b * material_data.metal_rough_factors.x);
    out_normal = vec4(view_space_normal, metal_rough.g * material_data.metal_rough_factors.y);
}
