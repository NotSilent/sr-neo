#version 460

#extension GL_GOOGLE_include_directive : require

#include "scene_data.glsl"
#include "pbr.glsl"

layout (set = 1, binding = 0) uniform sampler2D color_tex;  // color + metallic
layout (set = 1, binding = 1) uniform sampler2D normal_tex; // normal + roughness
layout (set = 1, binding = 2) uniform sampler2D depth_tex;
layout (set = 1, binding = 3) uniform sampler2D shadow_map_tex;
 
layout (location = 0) in vec2 in_uv;
 
layout (location = 0) out vec4 out_color;

float shadow_contribution(vec4 light_space_pos, vec3 normal_view, vec3 sunlight_rection_view)
{
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;

    //float closest_depth = texture(shadow_map_tex, proj_coords.xy * 0.5 + 0.5).r;

    float current_depth = proj_coords.z;
    
    if(current_depth > 1.0)
    {
        return 0.0;
    }

    float bias = max(0.05 * (1.0 - dot(normal_view, sunlight_rection_view)), 0.005);

    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(shadow_map_tex, 0);

    int SAMPLES = 2;

    for(int x = -SAMPLES; x <= SAMPLES; ++x)
    {
        for(int y = -SAMPLES; y <= SAMPLES; ++y)
        {
            // TODO: Texture gather might reduce it by more than half
            // eg. instead of 5*5 samples, 8 gathers and one center sample could cover the same area
            float pcf_depth = texture(shadow_map_tex, proj_coords.xy * 0.5 + 0.5 + vec2(x, y) * texel_size).r; 
            shadow += current_depth > pcf_depth - bias ? 1.0 : 0.0;        
        }    
    }

    shadow /= pow(SAMPLES + SAMPLES - 1, 2.0);

    return shadow;
}


void main()
{
	vec4 color = texture(color_tex,in_uv);
	vec4 normal = texture(normal_tex,in_uv);
	vec4 depth = texture(depth_tex,in_uv);
	vec4 shadow = texture(shadow_map_tex,in_uv);

	float metallic = color.a;
	float roughness = normal.a;

    vec2 ndc_xy = (gl_FragCoord.xy / scene_data.screen_size) * 2.0 - 1.0;
    float ndc_z = depth.r;
    vec4 ndc_pos = vec4(ndc_xy, ndc_z, 1.0);
    vec4 view_pos = scene_data.inv_proj * ndc_pos;
    vec3 pos =  view_pos.xyz / view_pos.w;

    // TODO: SceneData
    mat4 inv_view = inverse(scene_data.view);

    vec4 world_pos = inv_view * vec4(pos, 1.0);
    vec4 light_space_pos = scene_data.light_view_proj * world_pos;

	vec3 view_position = (scene_data.view * vec4(scene_data.view_position, 1.0)).rgb;
	vec3 light_direction = normalize((mat3(scene_data.view) * (scene_data.sunlight_direction.rgb)));

    float shadow_contribution = shadow_contribution(light_space_pos, normal.xyz, light_direction);

	vec3 N = normal.rgb;
    vec3 V = normalize(view_position - pos);
 
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, color.rgb, metallic);

    vec3 Lo = vec3(0.0);
    
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
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;	  

    float NdotL = max(dot(N, L), 0.0);        

    Lo += (kD * color.rgb / PI + specular) * radiance * NdotL * shadow_contribution;

    vec3 ambient = vec3(0.01) * color.rgb; // * ao;

    vec3 final_color = ambient + Lo;
    final_color = final_color / (final_color + vec3(1.0));
    final_color = pow(final_color, vec3(1.0/2.2)); 

    out_color = vec4(final_color, 1.0);
}
