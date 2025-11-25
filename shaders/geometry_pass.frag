#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "includes/scene_data.glsl"

layout(early_fragment_tests) in;

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in mat3 in_tbn_to_view;
layout(location = 5) flat in uint in_material_index;

layout(location = 0) out vec4 out_color; // color + metallic
layout(location = 1) out vec4 out_normal; // normal + roughness

void main()
{
    Material material = material_data2.materials[nonuniformEXT(in_material_index)];

    vec4 color = pow(texture(samplers[material.color_tex_index], in_uv), vec4(2.2));

    //TODO: This is temp for master material, should be different shader for masked
    if (color.w < 0.5) {
        discard;
    }

    vec3 albedo = color.rgb * in_color * material.color_factors.rgb;
    vec3 normal = texture(samplers[material.normal_tex_index], in_uv).rgb * 2.0 - 1.0;
    vec4 metal_rough = texture(samplers[material.metal_rough_tex_index], in_uv);

    vec3 view_space_normal = normalize(in_tbn_to_view * normal);

    out_color = vec4(albedo, metal_rough.b * material.metal_rough_factors.x);
    out_normal = vec4(view_space_normal, metal_rough.g * material.metal_rough_factors.y);
}
