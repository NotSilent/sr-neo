#version 460

#extension GL_GOOGLE_include_directive : require

#include "input_structures.glsl"
#include "pbr.glsl"

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_position;
layout (location = 3) in vec3 in_light_direction_tbn;
layout (location = 4) in float in_light_power;
layout (location = 5) in vec3 in_view_position_tbn;

layout (location = 0) out vec4 out_frag_color;

void main()
{
	vec4 color = texture(color_tex,in_uv);

	//TODO: This is temp for master material, should be different shader for masked
	if (color.w < 0.5) {
		discard;
	}

	vec3 normal = texture(normal_tex,in_uv).xyz * 2.0 - 1.0;
	vec3 ambient = 0.1 * color.rgb;

	float diff = max(dot(in_light_direction_tbn, normal), 0.0);
	vec3 diffuse = diff * color.rgb;

	vec3 view_dir = normalize(in_view_position_tbn - in_position);
	vec3 reflect_dir = reflect(-in_light_direction_tbn, normal);
    vec3 halfway_dir = normalize(in_light_direction_tbn + view_dir);  
    float spec = pow(max(dot(normal, halfway_dir), 0.0), 32.0);

    vec3 specular = vec3(0.2) * spec;
	
    out_frag_color = vec4(ambient + diffuse + spec, 1.0);
}
