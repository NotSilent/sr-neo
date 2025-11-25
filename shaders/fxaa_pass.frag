#version 460

#extension GL_GOOGLE_include_directive : require

#include "includes/scene_data.glsl"

layout(set = 1, binding = 0) uniform sampler2D color_tex;

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

const float EDGE_THRESHOLD_MIN = 0.0312;
const float EDGE_THRESHOLD_MAX = 0.125;
const uint ITERATIONS = 12;
const float SUBPIXEL_QUALITY = 0.75;

const bool DEBUG_EDGES = false;
const bool DEBUG_EDGE_DIRECTION = false;

float rgb_2_luma(vec3 rgb) {
    return sqrt(dot(rgb, vec3(0.299, 0.587, 0.114)));
}

const float OFFSET_PER_ITERATION[11] = float[](1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0);

float QUALITY(uint iteration) {
    if (iteration > 10) {
        return 8.0;
    }
    else
    {
        return OFFSET_PER_ITERATION[iteration];
    }
}

// https://blog.simonrodriguez.fr/articles/2016/07/implementing_fxaa.html
void main()
{
    vec2 inverse_screen_size = 1.0 / scene_data.screen_size;

    vec4 color = texture(color_tex, in_uv);

    vec3 color_center = color.rgb;

    // Luma at the current fragment
    float luma_center = rgb_2_luma(color_center);

    // Luma at the four direct neighbours of the current fragment.
    float luma_down = rgb_2_luma(textureOffset(color_tex, in_uv, ivec2(0, -1)).rgb);
    float luma_up = rgb_2_luma(textureOffset(color_tex, in_uv, ivec2(0, 1)).rgb);
    float luma_left = rgb_2_luma(textureOffset(color_tex, in_uv, ivec2(-1, 0)).rgb);
    float luma_right = rgb_2_luma(textureOffset(color_tex, in_uv, ivec2(1, 0)).rgb);

    // Find the maximum and minimum luma around the current fragment.
    float luma_min = min(luma_center, min(min(luma_down, luma_up), min(luma_left, luma_right)));
    float luma_max = max(luma_center, max(max(luma_down, luma_up), max(luma_left, luma_right)));

    // Compute the delta.
    float luma_range = luma_max - luma_min;

    // If the luma variation is lower that a threshold (or if we are in a really dark area), we are not on an edge, don't perform any AA.
    if (luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD_MAX))
    {
        out_color.rgb = color_center;
        return;
    }
    else if (DEBUG_EDGES)
    {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    // Query the 4 remaining corners lumas.
    float luma_down_left = rgb_2_luma(textureOffset(color_tex, in_uv, ivec2(-1, -1)).rgb);
    float luma_up_right = rgb_2_luma(textureOffset(color_tex, in_uv, ivec2(1, 1)).rgb);
    float luma_up_left = rgb_2_luma(textureOffset(color_tex, in_uv, ivec2(-1, 1)).rgb);
    float luma_down_right = rgb_2_luma(textureOffset(color_tex, in_uv, ivec2(1, -1)).rgb);

    // Combine the four edges lumas (using intermediary variables for future computations with the same values).
    float luma_down_up = luma_down + luma_up;
    float luma_left_right = luma_left + luma_right;

    // Same for corners
    float luma_left_corners = luma_down_left + luma_up_left;
    float luma_down_corners = luma_down_left + luma_down_right;
    float luma_right_corners = luma_down_right + luma_up_right;
    float luma_up_corners = luma_up_right + luma_up_left;

    // Compute an estimation of the gradient along the horizontal and vertical axis.
    float edge_horizontal = abs(-2.0 * luma_left + luma_left_corners) + abs(-2.0 * luma_center + luma_down_up) * 2.0 + abs(-2.0 * luma_right + luma_right_corners);
    float edge_vertical = abs(-2.0 * luma_up + luma_up_corners) + abs(-2.0 * luma_center + luma_left_right) * 2.0 + abs(-2.0 * luma_down + luma_down_corners);

    // Is the local edge horizontal or vertical ?
    bool is_horizontal = (edge_horizontal >= edge_vertical);

    if (DEBUG_EDGE_DIRECTION) {
        if (is_horizontal)
        {
            out_color = vec4(0.0, 1.0, 0.0, 1.0);
            return;
        }
        else
        {
            out_color = vec4(0.0, 0.0, 1.0, 1.0);
            return;
        }
    }

    // Select the two neighboring texels lumas in the opposite direction to the local edge.
    float luma_1 = is_horizontal ? luma_down : luma_left;
    float luma_2 = is_horizontal ? luma_up : luma_right;
    // Compute gradients in this direction.
    float gradient_1 = luma_1 - luma_center;
    float gradient_2 = luma_2 - luma_center;

    // Which direction is the steepest ?
    bool is_1_steepest = abs(gradient_1) >= abs(gradient_2);

    // Gradient in the corresponding direction, normalized.
    float gradient_scaled = 0.25 * max(abs(gradient_1), abs(gradient_2));

    // Choose the step size (one pixel) according to the edge direction.
    float step_length = is_horizontal ? inverse_screen_size.y : inverse_screen_size.x;

    // Average luma in the correct direction.
    float luma_local_average = 0.0;

    if (is_1_steepest)
    {
        // Switch the direction
        step_length = -step_length;
        luma_local_average = 0.5 * (luma_1 + luma_center);
    } else {
        luma_local_average = 0.5 * (luma_2 + luma_center);
    }

    // Shift UV in the correct direction by half a pixel.
    vec2 current_uv = in_uv;
    if (is_horizontal) {
        current_uv.y += step_length * 0.5;
    } else {
        current_uv.x += step_length * 0.5;
    }

    // Compute the offset per iteration step in the appropriate direction.
    vec2 offset = is_horizontal ? vec2(inverse_screen_size.x, 0.0) : vec2(0.0, inverse_screen_size.y);

    // Compute UVs on both sides of the edge for exploration.
    vec2 uv_1 = current_uv - offset;
    vec2 uv_2 = current_uv + offset;

    // Sample luma values at the edge extremities and compute their difference from the local average.
    float luma_end_1 = rgb_2_luma(texture(color_tex, uv_1).rgb) - luma_local_average;
    float luma_end_2 = rgb_2_luma(texture(color_tex, uv_2).rgb) - luma_local_average;

    // Check if either extremity has reached the edge (gradient threshold exceeded).
    bool reached_1 = abs(luma_end_1) >= gradient_scaled;
    bool reached_2 = abs(luma_end_2) >= gradient_scaled;
    bool reached_both = reached_1 && reached_2;

    // Continue exploring in the current direction if the edge is not yet reached.
    if (!reached_1) {
        uv_1 -= offset;
    }
    if (!reached_2) {
        uv_2 += offset;
    }

    // If both sides have not been reached, continue to explore.
    if (!reached_both) {
        for (int i = 2; i < ITERATIONS; i++) {
            // If needed, sample luma in the first direction and compute delta.
            if (!reached_1) {
                luma_end_1 = rgb_2_luma(texture(color_tex, uv_1).rgb) - luma_local_average;
            }

            // If needed, sample luma in the opposite direction and compute delta.
            if (!reached_2) {
                luma_end_2 = rgb_2_luma(texture(color_tex, uv_2).rgb) - luma_local_average;
            }

            // Determine if either direction has reached the edge based on gradient threshold.
            reached_1 = abs(luma_end_1) >= gradient_scaled;
            reached_2 = abs(luma_end_2) >= gradient_scaled;
            reached_both = reached_1 && reached_2;

            // If the edge hasn't been reached, continue exploring with adjusted step size.
            if (!reached_1) {
                uv_1 -= offset * QUALITY(i);
            }
            if (!reached_2) {
                uv_2 += offset * QUALITY(i);
            }

            // If both sides have reached the edge, stop the loop.
            if (reached_both) {
                break;
            }
        }
    }

    // Compute the distances to each extremity of the edge.
    float distance_1 = is_horizontal ? (in_uv.x - uv_1.x) : (in_uv.y - uv_1.y);
    float distance_2 = is_horizontal ? (uv_2.x - in_uv.x) : (uv_2.y - in_uv.y);

    // Determine which direction has the closer edge extremity.
    bool is_direction_1 = distance_1 < distance_2;
    float distance_final = min(distance_1, distance_2);

    // Total length of the edge.
    float edge_thickness = distance_1 + distance_2;

    // Compute UV offset toward the closer edge extremity.
    float pixel_offset = -distance_final / edge_thickness + 0.5;

    // Is the luma at the center smaller than the local average?
    bool is_luma_center_smaller = luma_center < luma_local_average;

    // If the center luma is smaller than its neighbor, the delta luma at the edge end should be positive.
    // (i.e., same variation direction, toward the closer side of the edge.)
    bool correct_variation = ((is_direction_1 ? luma_end_1 : luma_end_2) < 0.0) != is_luma_center_smaller;

    // If the luma variation is incorrect, cancel the offset.
    float final_offset = correct_variation ? pixel_offset : 0.0;

    // Sub-pixel shifting
    // Compute the full weighted average of luma over the 3x3 neighborhood.
    float luma_average = (1.0 / 12.0) * (2.0 * (luma_down_up + luma_left_right) + luma_left_corners + luma_right_corners);

    // Compute the ratio between the global average delta and the local luma range.
    float subpixel_offset_1 = clamp(abs(luma_average - luma_center) / luma_range, 0.0, 1.0);

    // Smoothstep-like curve to refine the subpixel offset.
    float subpixel_offset_2 = (-2.0 * subpixel_offset_1 + 3.0) * subpixel_offset_1 * subpixel_offset_1;

    // Final subpixel offset scaled by quality parameter.
    float subpixel_offset_final = subpixel_offset_2 * subpixel_offset_2 * SUBPIXEL_QUALITY;

    // Pick the larger of the two offsets.
    final_offset = max(final_offset, subpixel_offset_final);

    // Compute the final UV coordinates.
    vec2 final_uv = in_uv;
    if (is_horizontal) {
        final_uv.y += final_offset * step_length;
    } else {
        final_uv.x += final_offset * step_length;
    }

    vec3 final_color = texture(color_tex, final_uv).rgb;
    out_color = vec4(final_color, 1.0);
}
