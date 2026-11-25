// Multi-view Faster-GS rasterizer kernels.
// Derived from Faster-GS (https://github.com/nerficg-project/faster-gaussian-splatting, Apache-2.0):
// rasterization/include/kernels_forward.cuh and kernels_backward.cuh.
//
// Changes against Faster-GS:
//  - V cameras per call. Per-(view, Gaussian) arrays are [V * N] with index pv = v * N + n.
//  - preprocess: one thread per Gaussian loops over the views (parameters and SH loaded once).
//  - instance keys are view * n_tiles_view + tile, so one depth sort and one tile sort serve all views.
//  - blend / blend backward: grid z (forward) or the bucket's tile (backward) selects the view.
//  - backward: blend writes per-(view, Gaussian) 2D / colour / opacity gradients, then one thread per
//    Gaussian sums the views and writes every parameter gradient once. Scale and rotation gradients
//    are converted once from the summed dL/dcov3d.
//  - no densification info, abs-gradient or pruning scores.
#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "sh_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace faster_gs_mv::kernels {

    using namespace faster_gs::rasterization;
    using namespace faster_gs::rasterization::kernels;

    // Gaussians per backward bucket (one colour / transmittance checkpoint each) = 32 * kBucketPasses.
    // Checkpoint memory scales with 1 / kBucketPasses; the gradients do not change.
#ifndef FGS_BUCKET_PASSES
#define FGS_BUCKET_PASSES 8
#endif
    inline constexpr int kBucketPasses = FGS_BUCKET_PASSES;
    inline constexpr int kBucketSize = 32 * kBucketPasses;

    struct Camera {
        float4 r1, r2, r3;  // world-to-camera rows 0..2
        float fx, fy, cx, cy;
        float3 position;
    };

    __device__ inline Camera load_camera(const float4* __restrict__ w2c, const float4* __restrict__ intrinsics,
                                         const float3* __restrict__ cam_position, const uint v) {
        Camera c;
        c.r1 = w2c[4 * v + 0];
        c.r2 = w2c[4 * v + 1];
        c.r3 = w2c[4 * v + 2];
        const float4 k = intrinsics[v];
        c.fx = k.x; c.fy = k.y; c.cx = k.z; c.cy = k.w;
        c.position = cam_position[v];
        return c;
    }

    __device__ inline mat3x3_triu compute_cov3d(const float3& raw_scale, const mat3x3& R, mat3x3& RSS, float3& variance) {
        variance = expf(2.0f * raw_scale);
        RSS = {
            R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z,
            R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z,
            R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z
        };
        return {
            RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13,
            RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23,
            RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33,
            RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23,
            RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33,
            RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33,
        };
    }

    // ------------------------------------------------------------------------------------------
    // forward
    // ------------------------------------------------------------------------------------------

    __global__ void preprocess_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float4* __restrict__ intrinsics,
        const float3* __restrict__ cam_position,
        uint* __restrict__ depth_keys,
        uint* __restrict__ sort_values,
        uint* __restrict__ c_n_touched_tiles,
        ushort4* __restrict__ c_screen_bounds,
        float2* __restrict__ c_mean2d,
        float4* __restrict__ c_conic_opacity,
        float3* __restrict__ c_color,
        uint* __restrict__ c_local_view,
        uint* __restrict__ pv_to_c,
        uint2* __restrict__ counters,
        const uint capacity,
        const uint n_primitives,
        const uint n_views,
        const uint view_offset,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float width,
        const float height,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing)
    {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        uint primitive_idx = cg::this_grid().thread_rank();
        bool active_base = true;
        if (primitive_idx >= n_primitives) {
            active_base = false;
            primitive_idx = n_primitives - 1;
        }
        if (warp.ballot(active_base) == 0) return;

        // view-independent quantities
        const float3 mean3d = means[primitive_idx];
        const float opacity_base = sigmoid(opacities[primitive_idx]);
        if (config::original_opacity_interpretation && opacity_base < config::min_alpha_threshold) active_base = false;
        float quaternion_norm_sq = 1.0f;
        const mat3x3 R = convert_quaternion_to_rotation_matrix(rotations[primitive_idx], quaternion_norm_sq);
        if (quaternion_norm_sq < 1e-8f) active_base = false;
        mat3x3 RSS;
        float3 variance;
        const mat3x3_triu cov3d = compute_cov3d(scales[primitive_idx], R, RSS, variance);
        const int padded_width = static_cast<int>(grid_width * config::tile_width);
        const int padded_height = static_cast<int>(grid_height * config::tile_height);

        for (uint v = 0; v < n_views; v++) {
            bool active = active_base;
            const Camera cam = load_camera(w2c, intrinsics, cam_position, v);

            // z culling
            const float depth = cam.r3.x * mean3d.x + cam.r3.y * mean3d.y + cam.r3.z * mean3d.z + cam.r3.w;
            if (depth < near_plane || depth > far_plane) active = false;
            if (warp.ballot(active) == 0) continue;

            // 2d mean in normalized image coordinates
            const float x = (cam.r1.x * mean3d.x + cam.r1.y * mean3d.y + cam.r1.z * mean3d.z + cam.r1.w) / depth;
            const float y = (cam.r2.x * mean3d.x + cam.r2.y * mean3d.y + cam.r2.z * mean3d.z + cam.r2.w) / depth;

            // ewa splatting
            const float clip_left = (-0.15f * width - cam.cx) / cam.fx;
            const float clip_right = (1.15f * width - cam.cx) / cam.fx;
            const float clip_top = (-0.15f * height - cam.cy) / cam.fy;
            const float clip_bottom = (1.15f * height - cam.cy) / cam.fy;
            const float x_clipped = clamp(x, clip_left, clip_right);
            const float y_clipped = clamp(y, clip_top, clip_bottom);
            const float j11 = cam.fx / depth;
            const float j13 = -j11 * x_clipped;
            const float j22 = cam.fy / depth;
            const float j23 = -j22 * y_clipped;
            const float3 jw_r1 = make_float3(
                j11 * cam.r1.x + j13 * cam.r3.x,
                j11 * cam.r1.y + j13 * cam.r3.y,
                j11 * cam.r1.z + j13 * cam.r3.z
            );
            const float3 jw_r2 = make_float3(
                j22 * cam.r2.x + j23 * cam.r3.x,
                j22 * cam.r2.y + j23 * cam.r3.y,
                j22 * cam.r2.z + j23 * cam.r3.z
            );
            const float3 jwc_r1 = make_float3(
                jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
                jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
                jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33
            );
            const float3 jwc_r2 = make_float3(
                jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
                jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
                jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33
            );
            float3 cov2d = make_float3(dot(jwc_r1, jw_r1), dot(jwc_r1, jw_r2), dot(jwc_r2, jw_r2));
            const float determinant_raw = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
            const float kernel_size = proper_antialiasing ? config::dilation_proper_antialiasing : config::dilation;
            cov2d.x += kernel_size;
            cov2d.z += kernel_size;
            const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
            if (determinant < config::min_cov2d_determinant) active = false;
            const float3 conic = make_float3(cov2d.z / determinant, -cov2d.y / determinant, cov2d.x / determinant);
            float opacity = opacity_base;
            if (proper_antialiasing) {
                opacity *= sqrtf(fmaxf(determinant_raw / determinant, 0.0f));
                if (config::original_opacity_interpretation && opacity < config::min_alpha_threshold) active = false;
            }

            // 2d mean in screen space
            const float2 mean2d = make_float2(x * cam.fx + cam.cx, y * cam.fy + cam.cy);

            // compute bounds
            const float power_threshold = config::original_opacity_interpretation ? logf(opacity * config::min_alpha_threshold_rcp) : config::max_power_threshold;
            const float cutoff_factor = 2.0f * power_threshold;
            const float extent_x = fmaxf(sqrtf(cov2d.x * cutoff_factor) - 0.5f, 0.0f);
            const float extent_y = fmaxf(sqrtf(cov2d.z * cutoff_factor) - 0.5f, 0.0f);
            const ushort4 screen_bounds = make_ushort4(
                min(padded_width, max(0, __float2int_rd(mean2d.x - extent_x))),
                min(padded_width, max(0, __float2int_ru(mean2d.x + extent_x))),
                min(padded_height, max(0, __float2int_rd(mean2d.y - extent_y))),
                min(padded_height, max(0, __float2int_ru(mean2d.y + extent_y)))
            );
            const uint4 tile_bounds = convert_screen_bounds_to_tile_bounds(screen_bounds);
            const uint n_touched_tiles_max = (tile_bounds.y - tile_bounds.x) * (tile_bounds.w - tile_bounds.z);
            if (n_touched_tiles_max == 0) active = false;
            if (warp.ballot(active) == 0) continue;

            // exact number of tiles the primitive overlaps (warp-cooperative)
            const uint n_touched_tiles = compute_exact_n_touched_tiles(
                mean2d, conic, tile_bounds, power_threshold, n_touched_tiles_max, active);
            if (n_touched_tiles == 0 || !active) continue;

            // visible (view, Gaussian) pairs are stored compactly at slot c
            const uint c = atomicAdd(&counters[0].x, 1);
            atomicAdd(&counters[0].y, n_touched_tiles);
            if (c >= capacity) continue;  // storage too small: counted only, the host retries with the exact size
            c_n_touched_tiles[c] = n_touched_tiles;
            c_screen_bounds[c] = screen_bounds;
            c_mean2d[c] = mean2d;
            c_conic_opacity[c] = make_float4(conic, opacity);
            c_color[c] = convert_sh_to_color(
                sh_coefficients_0, sh_coefficients_rest, mean3d, cam.position,
                primitive_idx, active_sh_bases, total_sh_bases);
            c_local_view[c] = v;
            pv_to_c[(view_offset + v) * n_primitives + primitive_idx] = c;
            depth_keys[c] = __float_as_uint(depth);
            sort_values[c] = c;
        }
    }

    __global__ void apply_depth_ordering_cu(
        const uint* __restrict__ c_sorted,
        const uint* __restrict__ c_n_touched_tiles,
        uint* __restrict__ offsets,
        const uint n_visible)
    {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_visible) return;
        offsets[idx] = c_n_touched_tiles[c_sorted[idx]];
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    template <typename KeyT>
    __global__ void create_instances_cu(
        const uint* __restrict__ c_sorted,
        const uint* __restrict__ offsets,
        const ushort4* __restrict__ pv_screen_bounds,
        const float2* __restrict__ pv_mean2d,
        const float4* __restrict__ pv_conic_opacity,
        const uint* __restrict__ c_local_view,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_pv_indices,
        const uint grid_width,
        const uint n_tiles_view,
        const uint n_visible)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();
        const uint thread_rank = block.thread_rank();
        const uint warp_start = warp.meta_group_rank() * warp_size;
        const uint lane_idx = warp.thread_rank();
        const uint previous_lanes_mask = (1 << lane_idx) - 1;

        uint original_idx = thread_idx;
        bool active = true;
        if (original_idx >= n_visible) {
            active = false;
            original_idx = n_visible - 1;
        }
        if (warp.ballot(active) == 0) return;

        const uint pv = c_sorted[original_idx];
        const uint key_base = c_local_view[pv] * n_tiles_view;

        const uint4 tile_bounds = convert_screen_bounds_to_tile_bounds(pv_screen_bounds[pv]);
        const uint tile_bounds_width = tile_bounds.y - tile_bounds.x;
        const uint instance_count = (tile_bounds.w - tile_bounds.z) * tile_bounds_width;
        const float2 mean2d_shifted = pv_mean2d[pv] - 0.5f;
        const float4 conic_opacity = pv_conic_opacity[pv];
        const float3 conic = make_float3(conic_opacity);
        const float power_threshold = config::original_opacity_interpretation ? logf(conic_opacity.w * config::min_alpha_threshold_rcp) : config::max_power_threshold;

        uint current_write_offset = offsets[original_idx];

        for (uint instance_idx = 0; active && instance_idx < instance_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_x = tile_bounds.x + (instance_idx % tile_bounds_width);
            const uint tile_y = tile_bounds.z + (instance_idx / tile_bounds_width);
            if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                instance_keys[current_write_offset] = static_cast<KeyT>(key_base + tile_y * grid_width + tile_x);
                instance_pv_indices[current_write_offset] = pv;
                current_write_offset++;
            }
        }

        const bool compute_cooperatively = active && instance_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_tile_bounds[config::block_size_create_instances];
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
        __shared__ float4 collected_conic_power_threshold[config::block_size_create_instances];
        collected_tile_bounds[thread_rank] = make_ushort4(tile_bounds.x, tile_bounds.y, tile_bounds.z, tile_bounds.w);
        collected_mean2d_shifted[thread_rank] = mean2d_shifted;
        collected_conic_power_threshold[thread_rank] = make_float4(conic, power_threshold);

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint pv_coop = warp.shfl(pv, current_lane);
            const uint key_base_coop = warp.shfl(key_base, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);

            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 tile_bounds_coop = collected_tile_bounds[read_offset_shared];
            const float2 mean2d_shifted_coop = collected_mean2d_shifted[read_offset_shared];
            const float4 conic_power_threshold_coop = collected_conic_power_threshold[read_offset_shared];

            const uint tile_bounds_width_coop = static_cast<uint>(tile_bounds_coop.y - tile_bounds_coop.x);
            const uint instance_count_coop = tile_bounds_width_coop * static_cast<uint>(tile_bounds_coop.w - tile_bounds_coop.z);
            const float3 conic_coop = make_float3(conic_power_threshold_coop);
            const float power_threshold_coop = conic_power_threshold_coop.w;

            const uint remaining_instance_count = instance_count_coop - config::n_sequential_threshold;
            const uint n_iterations = div_round_up(remaining_instance_count, warp_size);
            for (uint i = 0; i < n_iterations; i++) {
                const uint instance_idx = i * warp_size + lane_idx + config::n_sequential_threshold;
                const uint tile_x = tile_bounds_coop.x + (instance_idx % tile_bounds_width_coop);
                const uint tile_y = tile_bounds_coop.z + (instance_idx / tile_bounds_width_coop);
                const bool write = instance_idx < instance_count_coop && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint write_ballot = warp.ballot(write);
                if (write) {
                    const uint write_offset = current_write_offset_coop + __popc(write_ballot & previous_lanes_mask);
                    instance_keys[write_offset] = static_cast<KeyT>(key_base_coop + tile_y * grid_width + tile_x);
                    instance_pv_indices[write_offset] = pv_coop;
                }
                current_write_offset_coop += __popc(write_ballot);
            }
            warp.sync();
        }
    }

    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* __restrict__ instance_keys,
        uint2* __restrict__ tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const KeyT instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const KeyT previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    __global__ void extract_bucket_counts_cu(
        const uint2* __restrict__ tile_instance_ranges,
        uint* __restrict__ tile_n_buckets,
        const uint n_tiles)
    {
        const uint tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (tile_idx >= n_tiles) return;
        const uint2 instance_range = tile_instance_ranges[tile_idx];
        tile_n_buckets[tile_idx] = div_round_up(instance_range.y - instance_range.x, static_cast<uint>(kBucketSize));
    }

    // grid (tiles_x, tiles_y, views). Per-pixel buffers and the image are [V, ...].
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_buckets_offset,
        const uint* __restrict__ instance_pv_indices,
        const ushort4* __restrict__ pv_screen_bounds,
        const float2* __restrict__ pv_mean2d,
        const float4* __restrict__ pv_conic_opacity,
        const float3* __restrict__ pv_color,
        const float3* __restrict__ bg_color,
        float* __restrict__ image,
        float* __restrict__ pixel_final_transmittances,
        uint* __restrict__ tile_max_n_processed,
        uint* __restrict__ pixel_n_processed,
        uint* __restrict__ bucket_tile_index,
        float4* __restrict__ bucket_color_transmittance,
        const uint width,
        const uint height,
        const uint grid_width,
        const uint n_tiles_view)
    {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        const dim3 group_index = block.group_index();
        const uint view = group_index.z;
        const uint thread_rank = block.thread_rank();
        const uint lane_idx = warp.thread_rank();
        const uint warp_idx = warp.meta_group_rank();
        const uint n_pixels = width * height;
        image += view * 3 * n_pixels;
        pixel_final_transmittances += view * n_pixels;
        pixel_n_processed += view * n_pixels;
        // setup subtiling
        const uint2 subtile_origin = make_uint2(
            group_index.x * config::tile_width + (warp_idx % config::subgrid_width) * config::subtile_width,
            group_index.y * config::tile_height + (warp_idx / config::subgrid_width) * config::subtile_height
        );
        const ushort4 subtile_bounds = make_ushort4(
            subtile_origin.x, subtile_origin.x + config::subtile_width,
            subtile_origin.y, subtile_origin.y + config::subtile_height
        );
        const uint2 pixel_coords = make_uint2(subtile_origin.x + lane_idx % config::subtile_width, subtile_origin.y + lane_idx / config::subtile_width);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        const uint local_idx = (pixel_coords.y - group_index.y * config::tile_height) * config::tile_width + (pixel_coords.x - group_index.x * config::tile_width);
        // setup tile info
        const uint tile_idx = view * n_tiles_view + group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_primitives_total = tile_range.y - tile_range.x;
        // setup bucket to tile mapping
        const int n_buckets = div_round_up(n_primitives_total, kBucketSize);
        uint bucket_offset = (tile_idx == 0) ? 0 : tile_buckets_offset[tile_idx - 1];
        for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank; n_buckets_remaining > 0; n_buckets_remaining -= config::block_size_blend, current_bucket_idx += config::block_size_blend) {
            if (current_bucket_idx < n_buckets) bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
        }
        __shared__ ushort4 collected_screen_bounds[config::block_size_blend];
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        float3 color_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        uint n_processed_and_used = 0;
        bool done = !inside;
        for (int n_primitives_remaining = n_primitives_total, current_fetch_idx = tile_range.x + thread_rank; n_primitives_remaining > 0; n_primitives_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_and(done)) break;
            if (current_fetch_idx < tile_range.y) {
                const uint pv = instance_pv_indices[current_fetch_idx];
                collected_screen_bounds[thread_rank] = pv_screen_bounds[pv];
                collected_mean2d[thread_rank] = pv_mean2d[pv];
                collected_conic_opacity[thread_rank] = scale_conic_for_exp2(pv_conic_opacity[pv]);
                collected_color[thread_rank] = fmaxf(pv_color[pv], 0.0f);
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_primitives_remaining);
            const uint n_processed_batch_start = n_primitives_total - n_primitives_remaining;
            for (int current_shmem_offset = 0; current_shmem_offset < current_batch_size; current_shmem_offset += 32) {
                // store current color and transmittance every kBucketSize Gaussians
                if (!done && (n_processed_batch_start + current_shmem_offset) % kBucketSize == 0) {
                    bucket_color_transmittance[bucket_offset * config::block_size_blend + local_idx] = make_float4(color_pixel, transmittance);
                    bucket_offset++;
                }
                // skip all Gaussians that do not overlap the subtile of this warp
                bool overlaps_subtile = false;
                if (current_shmem_offset + lane_idx < current_batch_size) {
                    const ushort4 screen_bounds = collected_screen_bounds[current_shmem_offset + lane_idx];
                    overlaps_subtile = screen_bounds.x < subtile_bounds.y && subtile_bounds.x < screen_bounds.y &&
                        screen_bounds.z < subtile_bounds.w && subtile_bounds.z < screen_bounds.w;
                }
                uint pending_primitives = warp.ballot(overlaps_subtile);
                while (!done && pending_primitives != 0u) {
                    const int j = current_shmem_offset + __ffs(static_cast<int>(pending_primitives)) - 1;
                    pending_primitives &= pending_primitives - 1;
                    const float4 conic_opacity = collected_conic_opacity[j];
                    const float3 conic = make_float3(conic_opacity);
                    const float opacity = conic_opacity.w;
                    const float2 delta = collected_mean2d[j] - pixel;
                    const float exponent = delta.x * (conic.x * delta.x + conic.y * delta.y) + conic.z * delta.y * delta.y;
                    const float gaussian = exp2f(fminf(exponent, 0.0f));
                    if (!config::original_opacity_interpretation && gaussian < config::min_alpha_threshold) continue;
                    const float alpha = opacity * gaussian;
                    if (config::original_opacity_interpretation && alpha < config::min_alpha_threshold) continue;
                    color_pixel += transmittance * alpha * collected_color[j];
                    transmittance *= 1.0f - alpha;
                    n_processed_and_used = n_processed_batch_start + j + 1;
                    if (transmittance < config::transmittance_threshold) done = true;
                }
            }
        }
        if (inside) {
            color_pixel += transmittance * bg_color[0];
            const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            image[pixel_idx] = color_pixel.x;
            image[n_pixels + pixel_idx] = color_pixel.y;
            image[2 * n_pixels + pixel_idx] = color_pixel.z;
            pixel_final_transmittances[pixel_idx] = transmittance;
            pixel_n_processed[pixel_idx] = n_processed_and_used;
        }
        constexpr int n_warps = config::block_size_blend / 32;
        __shared__ uint warp_max_n_processed[n_warps];
        n_processed_and_used = cg::reduce(warp, n_processed_and_used, cg::greater<uint>());
        if (lane_idx == 0) warp_max_n_processed[warp_idx] = n_processed_and_used;
        block.sync();
        if (thread_rank == 0) {
            for (int i = 1; i < n_warps; ++i) n_processed_and_used = max(n_processed_and_used, warp_max_n_processed[i]);
            tile_max_n_processed[tile_idx] = n_processed_and_used;
        }
    }

    // ------------------------------------------------------------------------------------------
    // backward
    // ------------------------------------------------------------------------------------------

    // based on https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404
    // Writes per-(view, Gaussian) gradients. grad_conic has 3 planes of stride `plane_stride`.
    __global__ void blend_backward_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_bucket_offsets,
        const uint* __restrict__ instance_pv_indices,
        const float2* __restrict__ pv_mean2d,
        const float4* __restrict__ pv_conic_opacity,
        const float3* __restrict__ pv_color,
        const float3* __restrict__ bg_color,
        const float* __restrict__ grad_image,
        const float* __restrict__ image,
        const float* __restrict__ pixel_final_transmittances,
        const uint* __restrict__ tile_max_n_processed,
        const uint* __restrict__ pixel_n_processed,
        const uint* __restrict__ bucket_tile_index,
        const float4* __restrict__ bucket_color_transmittance,
        float2* __restrict__ grad_mean2d,
        float* __restrict__ grad_conic,
        float* __restrict__ grad_opacity,
        float3* __restrict__ grad_color,
        const uint plane_stride,
        const uint width,
        const uint height,
        const uint grid_width,
        const uint n_tiles_view,
        const uint n_tiles,
        const bool proper_antialiasing)
    {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        const uint bucket_idx = block.group_index().x;
        const uint lane_idx = warp.thread_rank();

        if (bucket_idx >= tile_bucket_offsets[n_tiles - 1]) return;
        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        const uint max_n_processed = tile_max_n_processed[tile_idx];
        if (tile_bucket_idx * kBucketSize >= max_n_processed) return;

        const uint view = tile_idx / n_tiles_view;
        const uint local_tile_idx = tile_idx - view * n_tiles_view;
        const uint n_pixels = width * height;
        image += view * 3 * n_pixels;
        grad_image += view * 3 * n_pixels;
        pixel_final_transmittances += view * n_pixels;
        pixel_n_processed += view * n_pixels;

        const float3 background = bg_color[0];
        const uint2 tile_coords = {local_tile_idx % grid_width, local_tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        bucket_color_transmittance += bucket_idx * config::block_size_blend;
        __shared__ uint collected_last_contributor[32];
        __shared__ float4 collected_color_pixel_after_transmittance[32];
        __shared__ float4 collected_grad_info_pixel[32];
        // per-pixel state after the previous 32 Gaussians of this bucket (written by lane 31)
        __shared__ float4 handoff_color_pixel_after_transmittance[kBucketPasses > 1 ? config::block_size_blend : 1];

        // A bucket holds kBucketPasses groups of 32 Gaussians but one checkpoint: the warp walks the groups in
        // depth order, each group starting from the per-pixel state the previous group left behind.
        for (int pass = 0; pass < kBucketPasses; ++pass) {
            const int group_start = tile_bucket_idx * kBucketSize + pass * 32;
            if (group_start >= tile_n_primitives || group_start >= static_cast<int>(max_n_processed)) break;

            const int tile_primitive_idx = group_start + lane_idx;
            const int instance_idx = tile_instance_range.x + tile_primitive_idx;
            const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

            uint pv = 0;
            float2 mean2d = {0.0f, 0.0f};
            float3 conic = {0.0f, 0.0f, 0.0f};
            float3 conic_exp2 = {0.0f, 0.0f, 0.0f};
            float opacity = 0.0f;
            float3 color = {0.0f, 0.0f, 0.0f};
            float3 color_grad_factor = {0.0f, 0.0f, 0.0f};
            if (valid_primitive) {
                pv = instance_pv_indices[instance_idx];
                mean2d = pv_mean2d[pv];
                const float4 conic_opacity = pv_conic_opacity[pv];
                conic = make_float3(conic_opacity);
                conic_exp2 = make_float3(scale_conic_for_exp2(conic_opacity));
                opacity = conic_opacity.w;
                const float3 color_unclamped = pv_color[pv];
                color = fmaxf(color_unclamped, 0.0f);
                if (color_unclamped.x >= 0.0f) color_grad_factor.x = 1.0f;
                if (color_unclamped.y >= 0.0f) color_grad_factor.y = 1.0f;
                if (color_unclamped.z >= 0.0f) color_grad_factor.z = 1.0f;
            }

            float2 dL_dmean2d_accum = {0.0f, 0.0f};
            float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
            float dL_dopacity_accum = 0.0f;
            float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};

            uint last_contributor;
            float3 color_pixel_after;
            float transmittance;
            float3 grad_color_pixel;
            float grad_alpha_common;

            #pragma unroll
            for (int i = 0; i < config::block_size_blend + 31; ++i) {
                if (i % 32 == 0) {
                    const uint local_idx = i + lane_idx;
                    if (local_idx < config::block_size_blend) {
                        const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width, start_pixel_coords.y + local_idx / config::tile_width};
                        const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                        float3 color_pixel_w_bg, grad_color_pixel;
                        float final_transmittance = 0.0f;
                        uint n_processed = 0;
                        if (pixel_coords.x < width && pixel_coords.y < height) {
                            color_pixel_w_bg = make_float3(image[pixel_idx], image[n_pixels + pixel_idx], image[2 * n_pixels + pixel_idx]);
                            grad_color_pixel = make_float3(grad_image[pixel_idx], grad_image[n_pixels + pixel_idx], grad_image[2 * n_pixels + pixel_idx]);
                            final_transmittance = pixel_final_transmittances[pixel_idx];
                            n_processed = pixel_n_processed[pixel_idx];
                        }
                        if (pass == 0) {
                            const float4 color_transmittance = bucket_color_transmittance[local_idx];
                            collected_color_pixel_after_transmittance[lane_idx] = make_float4(
                                color_pixel_w_bg - final_transmittance * background - make_float3(color_transmittance),
                                color_transmittance.w
                            );
                        } else {
                            collected_color_pixel_after_transmittance[lane_idx] = handoff_color_pixel_after_transmittance[local_idx];
                        }
                        collected_grad_info_pixel[lane_idx] = make_float4(
                            grad_color_pixel,
                            final_transmittance * -dot(grad_color_pixel, background)
                        );
                        collected_last_contributor[lane_idx] = n_processed;
                    }
                    warp.sync();
                }

                if (i > 0) {
                    last_contributor = warp.shfl_up(last_contributor, 1);
                    color_pixel_after.x = warp.shfl_up(color_pixel_after.x, 1);
                    color_pixel_after.y = warp.shfl_up(color_pixel_after.y, 1);
                    color_pixel_after.z = warp.shfl_up(color_pixel_after.z, 1);
                    transmittance = warp.shfl_up(transmittance, 1);
                    grad_color_pixel.x = warp.shfl_up(grad_color_pixel.x, 1);
                    grad_color_pixel.y = warp.shfl_up(grad_color_pixel.y, 1);
                    grad_color_pixel.z = warp.shfl_up(grad_color_pixel.z, 1);
                    grad_alpha_common = warp.shfl_up(grad_alpha_common, 1);
                }

                const int idx = i - static_cast<int>(lane_idx);
                const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width, start_pixel_coords.y + idx / config::tile_width};
                const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;
                const bool in_tile = idx >= 0 && idx < config::block_size_blend;

                if (valid_primitive && valid_pixel && lane_idx == 0 && in_tile) {
                    const int current_shmem_index = i % 32;
                    last_contributor = collected_last_contributor[current_shmem_index];
                    const float4 color_pixel_after_transmittance = collected_color_pixel_after_transmittance[current_shmem_index];
                    color_pixel_after = make_float3(color_pixel_after_transmittance);
                    transmittance = color_pixel_after_transmittance.w;
                    const float4 grad_info_pixel = collected_grad_info_pixel[current_shmem_index];
                    grad_color_pixel = make_float3(grad_info_pixel);
                    grad_alpha_common = grad_info_pixel.w;
                }

                const bool skip = !valid_primitive || !valid_pixel || !in_tile || tile_primitive_idx >= last_contributor;
                if (!skip) {
                    const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
                    const float2 delta = mean2d - pixel;
                    const float exponent = delta.x * (conic_exp2.x * delta.x + conic_exp2.y * delta.y) + conic_exp2.z * delta.y * delta.y;
                    const float gaussian = exp2f(fminf(exponent, 0.0f));
                    const float alpha = opacity * gaussian;
                    const bool contributes = config::original_opacity_interpretation ? alpha >= config::min_alpha_threshold : gaussian >= config::min_alpha_threshold;
                    if (contributes) {
                        const float blending_weight = transmittance * alpha;
                        dL_dcolor_accum += blending_weight * grad_color_pixel * color_grad_factor;
                        color_pixel_after -= blending_weight * color;

                        const float one_minus_alpha = 1.0f - alpha;
                        const float one_minus_alpha_rcp = 1.0f / fmaxf(one_minus_alpha, config::one_minus_alpha_eps);
                        const float dL_dalpha_from_color = dot(transmittance * color - color_pixel_after * one_minus_alpha_rcp, grad_color_pixel);
                        const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
                        const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha;
                        dL_dopacity_accum += gaussian * dL_dalpha;

                        const float gaussian_grad_helper = -alpha * dL_dalpha;
                        dL_dconic_accum += 0.5f * gaussian_grad_helper * make_float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
                        dL_dmean2d_accum += gaussian_grad_helper * make_float2(
                            conic.x * delta.x + conic.y * delta.y,
                            conic.y * delta.x + conic.z * delta.y
                        );

                        transmittance *= one_minus_alpha;
                    }
                }
                // the last lane hands this pixel's state to the next group of 32
                if (kBucketPasses > 1 && lane_idx == 31 && valid_pixel && in_tile)
                    handoff_color_pixel_after_transmittance[idx] = make_float4(color_pixel_after, transmittance);
            }

            if (valid_primitive) {
                atomicAdd(&grad_mean2d[pv].x, dL_dmean2d_accum.x);
                atomicAdd(&grad_mean2d[pv].y, dL_dmean2d_accum.y);
                atomicAdd(&grad_conic[pv], dL_dconic_accum.x);
                atomicAdd(&grad_conic[plane_stride + pv], dL_dconic_accum.y);
                atomicAdd(&grad_conic[2 * plane_stride + pv], dL_dconic_accum.z);
                // non-AA: gradient w.r.t. the raw (logit) opacity. AA: w.r.t. the compensated opacity
                const float dL_dopacity = proper_antialiasing ? dL_dopacity_accum : opacity * (1.0f - opacity) * dL_dopacity_accum;
                atomicAdd(&grad_opacity[pv], dL_dopacity);
                atomicAdd(&grad_color[pv].x, dL_dcolor_accum.x);
                atomicAdd(&grad_color[pv].y, dL_dcolor_accum.y);
                atomicAdd(&grad_color[pv].z, dL_dcolor_accum.z);
            }
            warp.sync();
        }
    }

    // SH backward for one view, adding into the per-thread rest-coefficient gradients in shared memory.
    // Returns dL/dmean3d through the view direction. Adapted from sh_utils.cuh convert_sh_to_color_backward.
    __device__ inline float3 sh_backward_accumulate(
        const float3* __restrict__ coefficients_ptr,
        float3* __restrict__ grad_coefficients_ptr,
        const float3& position,
        const float3& cam_position,
        const float3& grad_color,
        const uint active_sh_bases)
    {
        float3 dcolor_dposition = make_float3(0.0f);
        if (active_sh_bases <= 1) return dcolor_dposition;
        auto [x_raw, y_raw, z_raw] = position - cam_position;
        auto [x, y, z] = normalize(make_float3(x_raw, y_raw, z_raw));
        grad_coefficients_ptr[0] += -C1 * y * grad_color;
        grad_coefficients_ptr[1] += C1 * z * grad_color;
        grad_coefficients_ptr[2] += -C1 * x * grad_color;
        float3 grad_direction_x = -C1 * coefficients_ptr[2];
        float3 grad_direction_y = -C1 * coefficients_ptr[0];
        float3 grad_direction_z = C1 * coefficients_ptr[1];
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            grad_coefficients_ptr[3] += C2a * xy * grad_color;
            grad_coefficients_ptr[4] += -C2a * yz * grad_color;
            grad_coefficients_ptr[5] += (C2b * zz - C2c) * grad_color;
            grad_coefficients_ptr[6] += -C2a * xz * grad_color;
            grad_coefficients_ptr[7] += C2d * (xx - yy) * grad_color;
            grad_direction_x = grad_direction_x + C2a * y * coefficients_ptr[3] - C2a * z * coefficients_ptr[6] + C2a * x * coefficients_ptr[7];
            grad_direction_y = grad_direction_y + C2a * x * coefficients_ptr[3] - C2a * z * coefficients_ptr[4] - C2a * y * coefficients_ptr[7];
            grad_direction_z = grad_direction_z - C2a * y * coefficients_ptr[4] + C2e * z * coefficients_ptr[5] - C2a * x * coefficients_ptr[6];
            if (active_sh_bases > 9) {
                grad_coefficients_ptr[8] += y * (C3a * yy - C3b * xx) * grad_color;
                grad_coefficients_ptr[9] += C3c * xy * z * grad_color;
                grad_coefficients_ptr[10] += y * (C3d - C3e * zz) * grad_color;
                grad_coefficients_ptr[11] += z * (C3f * zz - C3g) * grad_color;
                grad_coefficients_ptr[12] += x * (C3d - C3e * zz) * grad_color;
                grad_coefficients_ptr[13] += C3h * z * (xx - yy) * grad_color;
                grad_coefficients_ptr[14] += x * (C3b * yy - C3a * xx) * grad_color;
                grad_direction_x = grad_direction_x - C3i * xy * coefficients_ptr[8] + C3c * yz * coefficients_ptr[9]
                                   + (C3d - C3e * zz) * coefficients_ptr[12] + C3c * xz * coefficients_ptr[13]
                                   + C3b * (yy - xx) * coefficients_ptr[14];
                grad_direction_y = grad_direction_y + C3b * (yy - xx) * coefficients_ptr[8] + C3c * xz * coefficients_ptr[9]
                                   + (C3d - C3e * zz) * coefficients_ptr[10] - C3c * yz * coefficients_ptr[13]
                                   + C3i * xy * coefficients_ptr[14];
                grad_direction_z = grad_direction_z + C3c * xy * coefficients_ptr[9] - C3j * yz * coefficients_ptr[10]
                                   + (C3k * zz - C3g) * coefficients_ptr[11] - C3j * xz * coefficients_ptr[12]
                                   + C3h * (xx - yy) * coefficients_ptr[13];
            }
        }
        const float3 grad_direction = make_float3(
            dot(grad_direction_x, grad_color), dot(grad_direction_y, grad_color), dot(grad_direction_z, grad_color));
        const float xx_raw = x_raw * x_raw, yy_raw = y_raw * y_raw, zz_raw = z_raw * z_raw;
        const float xy_raw = x_raw * y_raw, xz_raw = x_raw * z_raw, yz_raw = y_raw * z_raw;
        const float norm_sq = xx_raw + yy_raw + zz_raw;
        dcolor_dposition = make_float3(
            (yy_raw + zz_raw) * grad_direction.x - xy_raw * grad_direction.y - xz_raw * grad_direction.z,
            -xy_raw * grad_direction.x + (xx_raw + zz_raw) * grad_direction.y - yz_raw * grad_direction.z,
            -xz_raw * grad_direction.x - yz_raw * grad_direction.y + (xx_raw + yy_raw) * grad_direction.z
        ) * rsqrtf(norm_sq * norm_sq * norm_sq);
        return dcolor_dposition;
    }

    // One thread per Gaussian: sums the per-view gradients of all views and writes each parameter gradient once.
    __global__ void preprocess_backward_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float4* __restrict__ intrinsics,
        const float3* __restrict__ cam_position,
        const uint* __restrict__ pv_to_c,
        const float2* __restrict__ grad_mean2d,
        const float* __restrict__ grad_conic,
        const float* __restrict__ grad_opacity_pv,
        const float3* __restrict__ grad_color,
        const uint plane_stride,
        float3* __restrict__ grad_means,
        float3* __restrict__ grad_scales,
        float4* __restrict__ grad_rotations,
        float* __restrict__ grad_opacities,
        float3* __restrict__ grad_sh_coefficients_0,
        float3* __restrict__ grad_sh_coefficients_rest,
        const uint n_primitives,
        const uint n_views,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float width,
        const float height,
        const bool proper_antialiasing)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();
        const uint lane_idx = warp.thread_rank();
        if (thread_idx - lane_idx >= n_primitives) return;
        const bool valid = thread_idx < n_primitives;
        const uint primitive_idx = valid ? thread_idx : n_primitives - 1;

        extern __shared__ float3 collected_grad_sh_coefficients_rest[];
        float3* grad_sh_rest_primitive = collected_grad_sh_coefficients_rest + block.thread_rank() * total_sh_bases;
        for (uint i = 0; i < total_sh_bases; i++) grad_sh_rest_primitive[i] = make_float3(0.0f);

        float3 dL_dmean3d = make_float3(0.0f);
        mat3x3_triu dL_dcov3d = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float dL_dopacity = 0.0f;
        float3 dL_dcolor_sum = make_float3(0.0f);
        bool any_view = false;

        if (valid) {
            const float3 mean3d = means[primitive_idx];
            const float4 raw_rotation = rotations[primitive_idx];
            float quaternion_norm_sq = 1.0f;
            const mat3x3 R = convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq);
            mat3x3 RSS;
            float3 variance;
            const mat3x3_triu cov3d = compute_cov3d(scales[primitive_idx], R, RSS, variance);
            const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * total_sh_bases;
            const float opacity_base = proper_antialiasing ? sigmoid(opacities[primitive_idx]) : 0.0f;

            for (uint v = 0; v < n_views; v++) {
                const uint pv = pv_to_c[v * n_primitives + primitive_idx];
                if (pv == 0xffffffffu) continue;  // not visible in this view
                const float2 dL_dmean2d = grad_mean2d[pv];
                const float3 dL_dconic = make_float3(grad_conic[pv], grad_conic[plane_stride + pv], grad_conic[2 * plane_stride + pv]);
                const float dL_dopacity_v = grad_opacity_pv[pv];
                const float3 dL_dcolor = grad_color[pv];
                // a Gaussian outside every tile list of this view has all-zero gradients, which add nothing
                if (dL_dmean2d.x == 0.0f && dL_dmean2d.y == 0.0f && dL_dconic.x == 0.0f && dL_dconic.y == 0.0f &&
                    dL_dconic.z == 0.0f && dL_dopacity_v == 0.0f && dL_dcolor.x == 0.0f && dL_dcolor.y == 0.0f && dL_dcolor.z == 0.0f)
                    continue;
                any_view = true;
                const Camera cam = load_camera(w2c, intrinsics, cam_position, v);

                // colour
                dL_dcolor_sum += dL_dcolor;
                dL_dmean3d += sh_backward_accumulate(coefficients_ptr, grad_sh_rest_primitive, mean3d, cam.position, dL_dcolor, active_sh_bases);

                // projection
                const float depth = cam.r3.x * mean3d.x + cam.r3.y * mean3d.y + cam.r3.z * mean3d.z + cam.r3.w;
                const float x = (cam.r1.x * mean3d.x + cam.r1.y * mean3d.y + cam.r1.z * mean3d.z + cam.r1.w) / depth;
                const float y = (cam.r2.x * mean3d.x + cam.r2.y * mean3d.y + cam.r2.z * mean3d.z + cam.r2.w) / depth;
                const float clip_left = (-0.15f * width - cam.cx) / cam.fx;
                const float clip_right = (1.15f * width - cam.cx) / cam.fx;
                const float clip_top = (-0.15f * height - cam.cy) / cam.fy;
                const float clip_bottom = (1.15f * height - cam.cy) / cam.fy;
                const float x_clipped = clamp(x, clip_left, clip_right);
                const float y_clipped = clamp(y, clip_top, clip_bottom);
                const float j11 = cam.fx / depth;
                const float j13 = -j11 * x_clipped;
                const float j22 = cam.fy / depth;
                const float j23 = -j22 * y_clipped;
                const float3 jw_r1 = make_float3(j11 * cam.r1.x + j13 * cam.r3.x, j11 * cam.r1.y + j13 * cam.r3.y, j11 * cam.r1.z + j13 * cam.r3.z);
                const float3 jw_r2 = make_float3(j22 * cam.r2.x + j23 * cam.r3.x, j22 * cam.r2.y + j23 * cam.r3.y, j22 * cam.r2.z + j23 * cam.r3.z);
                const float3 jwc_r1 = make_float3(
                    jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
                    jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
                    jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
                const float3 jwc_r2 = make_float3(
                    jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
                    jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
                    jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);

                // 2d covariance gradient
                const float a_raw = dot(jwc_r1, jw_r1), b = dot(jwc_r1, jw_r2), c_raw = dot(jwc_r2, jw_r2);
                const float kernel_size = proper_antialiasing ? config::dilation_proper_antialiasing : config::dilation;
                const float a = a_raw + kernel_size, c = c_raw + kernel_size;
                const float aa = a * a, bb = b * b, cc = c * c;
                const float ac = a * c, ab = a * b, bc = b * c;
                const float determinant = ac - bb;
                const float determinant_rcp_sq = 1.0f / (determinant * determinant);
                const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                    2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                    bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
                    2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

                // opacity (AA: chain through the compensation factor, detached from cov2d as in Faster-GS)
                if (proper_antialiasing) {
                    const float conv_factor = sqrtf(fmaxf((a_raw * c_raw - bb) / determinant, 0.0f));
                    dL_dopacity += dL_dopacity_v * conv_factor * opacity_base * (1.0f - opacity_base);
                } else {
                    dL_dopacity += dL_dopacity_v;
                }

                // 3d covariance gradient (summed over views, converted to scale / rotation once below)
                dL_dcov3d.m11 += jw_r1.x * jw_r1.x * dL_dcov2d.x + 2.0f * jw_r1.x * jw_r2.x * dL_dcov2d.y + jw_r2.x * jw_r2.x * dL_dcov2d.z;
                dL_dcov3d.m12 += jw_r1.x * jw_r1.y * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.y * dL_dcov2d.z;
                dL_dcov3d.m13 += jw_r1.x * jw_r1.z * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.z * dL_dcov2d.z;
                dL_dcov3d.m22 += jw_r1.y * jw_r1.y * dL_dcov2d.x + 2.0f * jw_r1.y * jw_r2.y * dL_dcov2d.y + jw_r2.y * jw_r2.y * dL_dcov2d.z;
                dL_dcov3d.m23 += jw_r1.y * jw_r1.z * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + jw_r2.y * jw_r2.z * dL_dcov2d.z;
                dL_dcov3d.m33 += jw_r1.z * jw_r1.z * dL_dcov2d.x + 2.0f * jw_r1.z * jw_r2.z * dL_dcov2d.y + jw_r2.z * jw_r2.z * dL_dcov2d.z;

                // gradient of J * W
                const float3 dL_djw_r1 = 2.0f * make_float3(
                    jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
                    jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
                    jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y);
                const float3 dL_djw_r2 = 2.0f * make_float3(
                    jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
                    jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
                    jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z);
                const float dL_dj11 = cam.r1.x * dL_djw_r1.x + cam.r1.y * dL_djw_r1.y + cam.r1.z * dL_djw_r1.z;
                const float dL_dj22 = cam.r2.x * dL_djw_r2.x + cam.r2.y * dL_djw_r2.y + cam.r2.z * dL_djw_r2.z;
                const float dL_dj13 = cam.r3.x * dL_djw_r1.x + cam.r3.y * dL_djw_r1.y + cam.r3.z * dL_djw_r1.z;
                const float dL_dj23 = cam.r3.x * dL_djw_r2.x + cam.r3.y * dL_djw_r2.y + cam.r3.z * dL_djw_r2.z;

                // mean3d camera space gradient from mean2d and from J (accounting for clipping)
                float3 dL_dmean3d_cam = make_float3(
                    j11 * dL_dmean2d.x,
                    j22 * dL_dmean2d.y,
                    -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y);
                const bool valid_x = x >= clip_left && x <= clip_right;
                const bool valid_y = y >= clip_top && y <= clip_bottom;
                if (valid_x) dL_dmean3d_cam.x -= j11 * dL_dj13 / depth;
                if (valid_y) dL_dmean3d_cam.y -= j22 * dL_dj23 / depth;
                const float factor_x = 1.0f + static_cast<float>(valid_x);
                const float factor_y = 1.0f + static_cast<float>(valid_y);
                dL_dmean3d_cam.z += (j11 * (factor_x * x_clipped * dL_dj13 - dL_dj11) + j22 * (factor_y * y_clipped * dL_dj23 - dL_dj22)) / depth;
                dL_dmean3d += make_float3(
                    cam.r1.x * dL_dmean3d_cam.x + cam.r2.x * dL_dmean3d_cam.y + cam.r3.x * dL_dmean3d_cam.z,
                    cam.r1.y * dL_dmean3d_cam.x + cam.r2.y * dL_dmean3d_cam.y + cam.r3.y * dL_dmean3d_cam.z,
                    cam.r1.z * dL_dmean3d_cam.x + cam.r2.z * dL_dmean3d_cam.y + cam.r3.z * dL_dmean3d_cam.z);
            }

            float3 dL_dscale = make_float3(0.0f);
            float4 dL_drotation = make_float4(0.0f);
            if (any_view) {
                const float3 dL_dvariance = make_float3(
                    R.m11 * R.m11 * dL_dcov3d.m11 + R.m21 * R.m21 * dL_dcov3d.m22 + R.m31 * R.m31 * dL_dcov3d.m33 +
                        2.0f * (R.m11 * R.m21 * dL_dcov3d.m12 + R.m11 * R.m31 * dL_dcov3d.m13 + R.m21 * R.m31 * dL_dcov3d.m23),
                    R.m12 * R.m12 * dL_dcov3d.m11 + R.m22 * R.m22 * dL_dcov3d.m22 + R.m32 * R.m32 * dL_dcov3d.m33 +
                        2.0f * (R.m12 * R.m22 * dL_dcov3d.m12 + R.m12 * R.m32 * dL_dcov3d.m13 + R.m22 * R.m32 * dL_dcov3d.m23),
                    R.m13 * R.m13 * dL_dcov3d.m11 + R.m23 * R.m23 * dL_dcov3d.m22 + R.m33 * R.m33 * dL_dcov3d.m33 +
                        2.0f * (R.m13 * R.m23 * dL_dcov3d.m12 + R.m13 * R.m33 * dL_dcov3d.m13 + R.m23 * R.m33 * dL_dcov3d.m23));
                dL_dscale = 2.0f * variance * dL_dvariance;
                const mat3x3 dL_dR = {
                    2.0f * (RSS.m11 * dL_dcov3d.m11 + RSS.m21 * dL_dcov3d.m12 + RSS.m31 * dL_dcov3d.m13),
                    2.0f * (RSS.m12 * dL_dcov3d.m11 + RSS.m22 * dL_dcov3d.m12 + RSS.m32 * dL_dcov3d.m13),
                    2.0f * (RSS.m13 * dL_dcov3d.m11 + RSS.m23 * dL_dcov3d.m12 + RSS.m33 * dL_dcov3d.m13),
                    2.0f * (RSS.m11 * dL_dcov3d.m12 + RSS.m21 * dL_dcov3d.m22 + RSS.m31 * dL_dcov3d.m23),
                    2.0f * (RSS.m12 * dL_dcov3d.m12 + RSS.m22 * dL_dcov3d.m22 + RSS.m32 * dL_dcov3d.m23),
                    2.0f * (RSS.m13 * dL_dcov3d.m12 + RSS.m23 * dL_dcov3d.m22 + RSS.m33 * dL_dcov3d.m23),
                    2.0f * (RSS.m11 * dL_dcov3d.m13 + RSS.m21 * dL_dcov3d.m23 + RSS.m31 * dL_dcov3d.m33),
                    2.0f * (RSS.m12 * dL_dcov3d.m13 + RSS.m22 * dL_dcov3d.m23 + RSS.m32 * dL_dcov3d.m33),
                    2.0f * (RSS.m13 * dL_dcov3d.m13 + RSS.m23 * dL_dcov3d.m23 + RSS.m33 * dL_dcov3d.m33)
                };
                dL_drotation = convert_quaternion_to_rotation_matrix_backward(raw_rotation, dL_dR);
            }
            grad_means[primitive_idx] = dL_dmean3d;
            grad_scales[primitive_idx] = dL_dscale;
            grad_rotations[primitive_idx] = dL_drotation;
            grad_opacities[primitive_idx] = dL_dopacity;
            grad_sh_coefficients_0[primitive_idx] = C0 * dL_dcolor_sum;
        }
        warp.sync();

        // warp-wide coalesced write of the rest coefficients (the warp's first primitive is a multiple of 32)
        const uint warp_primitive_offset = thread_idx - lane_idx;
        const uint n_floats_warp = min(warp_size, n_primitives - warp_primitive_offset) * 3 * total_sh_bases;
        const float* collected_warp = reinterpret_cast<const float*>(collected_grad_sh_coefficients_rest + warp.meta_group_rank() * warp_size * total_sh_bases);
        float* grad_warp = reinterpret_cast<float*>(grad_sh_coefficients_rest + warp_primitive_offset * total_sh_bases);
        const uint n_float4_warp = n_floats_warp / 4;
        for (uint i = lane_idx; i < n_float4_warp; i += warp_size) reinterpret_cast<float4*>(grad_warp)[i] = reinterpret_cast<const float4*>(collected_warp)[i];
        for (uint i = 4 * n_float4_warp + lane_idx; i < n_floats_warp; i += warp_size) grad_warp[i] = collected_warp[i];
    }

}
