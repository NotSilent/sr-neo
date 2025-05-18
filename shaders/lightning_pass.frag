#version 460

#extension GL_GOOGLE_include_directive : require

#include "pbr.glsl"

layout(set = 0, binding = 0) uniform SceneData{   
	mat4 view;
	mat4 proj;
	mat4 inv_proj;
	mat4 view_proj;
	vec4 ambient_color;
	vec4 sunlight_direction; //w for sun power
	vec4 sunlight_color;
	vec3 view_position;
	float padding;
	vec2 screen_size;
} scene_data;

layout (set = 1, binding = 0) uniform sampler2D color_tex;  // color + metallic
layout (set = 1, binding = 1) uniform sampler2D normal_tex; // normal + roughness
layout (set = 1, binding = 2) uniform sampler2D depth_tex;
 
layout (location = 0) in vec2 in_uv;
 
layout (location = 0) out vec4 out_color;

void main()
{
	vec4 color = texture(color_tex,in_uv);
	vec4 normal = texture(normal_tex,in_uv);
	vec4 depth = texture(depth_tex,in_uv);

	float metallic = color.a;
	float roughness = normal.a;

    vec2 ndc_xy = (gl_FragCoord.xy / scene_data.screen_size) * 2.0 - 1.0;
    float ndc_z = depth.r;
    vec4 ndc_pos = vec4(ndc_xy, ndc_z, 1.0);
    vec4 view_pos = scene_data.inv_proj * ndc_pos;
    vec3 pos =  view_pos.xyz / view_pos.w;

    out_color = vec4(pos, 1.0);

	vec3 view_position = (scene_data.view * vec4(scene_data.view_position, 1.0)).rgb;
	vec3 light_direction = normalize((mat3(scene_data.view) * (scene_data.sunlight_direction.rgb)));

	vec3 N = normal.rgb;
    vec3 V = normalize(view_position - pos);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, color.rgb, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    // for(int i = 0; i < 4; ++i) 
    // {
        // calculate per-light radiance
        vec3 L = light_direction;
        vec3 H = normalize(V + L);
        // float distance = length(lightPositions[i] - WorldPos);
        // attenuation = 1.0 / (distance * distance);
        vec3 radiance = vec3(1.0); // lightColors[i] * attenuation;

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);
           
        vec3 numerator    = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
        vec3 specular = numerator / denominator;
        
        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;	  

        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * color.rgb / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    // }   
    
    // ambient lighting (note that the next IBL tutorial will replace 
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.01) * color.rgb; // * ao;

    vec3 final_color = ambient + Lo;

    // HDR tonemapping
    final_color = final_color / (final_color + vec3(1.0));
    // gamma correct
    final_color = pow(final_color, vec3(1.0/2.2)); 

    out_color = vec4(final_color, 1.0);
}
