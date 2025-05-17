#version 460

layout(set = 1, binding = 0) uniform sampler2D color_tex;  // color + metallic
layout(set = 1, binding = 1) uniform sampler2D normal_tex; // normal + roughness

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

void main()
{
	vec4 color = texture(color_tex,in_uv);
	vec4 normal = texture(normal_tex,in_uv);
    
    //out_color = vec4(vec3(1.0), 1.0);
    out_color = vec4(color.rgb, 1.0);
}
