#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outLightDirectionTBN;
layout (location = 3) out float outLightPower;

struct Vertex 
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
	vec4 tangent;
	vec2 metal_rough;
	vec2 padding;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

//push constants block
layout( push_constant ) uniform constants
{
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

void main() 
{
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position, 1.0f);

	gl_Position =  sceneData.viewproj * PushConstants.render_matrix * position;

	outColor = v.color.xyz * materialData.colorFactors.xyz;	
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;

	vec3 T = normalize(mat3(PushConstants.render_matrix) * v.tangent.xyz);
	vec3 N = normalize(mat3(PushConstants.render_matrix) * v.normal);
	// re-orthogonalize T with respect to N
	T = normalize(T - dot(T, N) * N);
	// then retrieve perpendicular vector B with the cross product of T and N
	vec3 B = cross(N, T) * v.tangent.w;

	// Transforms world-space to TBN space
	mat3 TBN = inverse(mat3(T, B, N));

	// Transform world-space LightDirection to TBN space
	outLightDirectionTBN = TBN * sceneData.sunlightDirection.xyz, 1.0;
	outLightPower = sceneData.sunlightDirection.w;
}
