#version 460

#extension GL_GOOGLE_include_directive : require

#include "input_structures.glsl"
#include "pbr.glsl"

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_frag_position_tbn;
layout (location = 3) in vec3 in_light_direction_tbn;
layout (location = 4) in float in_light_power;
layout (location = 5) in vec3 in_view_position_tbn;

layout (location = 0) out vec4 out_frag_color;

// PBR calculations from: https://learnopengl.com/PBR/Lighting
// Modified to be in TBH space
void main()
{
	vec4 color = pow(texture(color_tex,in_uv), vec4(2.2));

	//TODO: This is temp for master material, should be different shader for masked
	if (color.w < 0.5) {
		discard;
	}

	vec3 normal = texture(normal_tex,in_uv).xyz * 2.0 - 1.0;
	vec3 albedo = color.rgb * in_color * material_data.color_factors.rgb;
	vec4 metal_rough = texture(metal_rough_tex,in_uv);

	float metallic = metal_rough.b * material_data.metal_rough_factors.x;
	float roughness = metal_rough.g * material_data.metal_rough_factors.y;

	vec3 N = normal;
    vec3 V = normalize(in_view_position_tbn - in_frag_position_tbn);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    // for(int i = 0; i < 4; ++i) 
    // {
        // calculate per-light radiance
        vec3 L = in_light_direction_tbn;
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
    vec3 ambient = vec3(0.03) * color.rgb; // * ao;

    vec3 final_color = ambient + Lo;

    // HDR tonemapping
    final_color = final_color / (final_color + vec3(1.0));
    // gamma correct
    final_color = pow(final_color, vec3(1.0/2.2)); 

    out_frag_color = vec4(final_color, color.a * in_color * material_data.color_factors.a);
}
