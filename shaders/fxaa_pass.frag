#version 460

#extension GL_GOOGLE_include_directive : require

#include "scene_data.glsl"

layout (set = 1, binding = 0) uniform sampler2D color_tex;
 
layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec4 out_color;

void main()
{
	vec4 color = texture(color_tex,in_uv);

    

    out_color = color;
}
