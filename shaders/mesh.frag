#version 460

#extension GL_GOOGLE_include_directive : require

#include "input_structures.glsl"

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inLightDirectionTBN;
layout (location = 3) in float inLightPower;

layout (location = 0) out vec4 outFragColor;

void main()
{
	vec3 normal = texture(normalTex,inUV).xyz * 2.0 -1.0;

	float lightValue = max(dot(normal, inLightDirectionTBN.xyz), 0.0f);

	vec3 color = inColor * texture(colorTex,inUV).xyz;
	vec3 ambient = color * sceneData.ambientColor.xyz;

	outFragColor = vec4(color * lightValue * inLightPower + ambient, 1.0f);
}
