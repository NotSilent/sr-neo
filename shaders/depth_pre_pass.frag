#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene_data.glsl"

layout (location = 0) in vec2 in_uv;
layout (location = 1) flat in uint in_material_index;

void main()
{
	Material material = material_data2.materials[nonuniformEXT(in_material_index)];

	float alpha = texture(samplers[material.color_tex_index],in_uv).a;

	//TODO: This is temp for masked material support
	if (alpha < 0.5) {
		discard;
	}
}
