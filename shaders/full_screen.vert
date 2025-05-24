#version 460

layout (location = 0) out vec2 out_uv;

void main() 
{
    const vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(-1.0,  3.0),
        vec2( 3.0, -1.0)
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);

    // Map from position to UV coordinates (0 to 1)
    out_uv = gl_Position.xy * 0.5 + 0.5;
}
