layout(set = 1, binding = 0) uniform MaterialData{   
	vec4 color_factors;
	vec4 metal_rough_factors;
	
} material_data;

layout(set = 1, binding = 1) uniform sampler2D color_tex;
layout(set = 1, binding = 2) uniform sampler2D normal_tex;
layout(set = 1, binding = 3) uniform sampler2D metal_rough_tex;
