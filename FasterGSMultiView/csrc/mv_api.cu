// Host side of the multi-view Faster-GS rasterizer (see mv_kernels.cuh).
//
// mode "batched": one pipeline for all V views (one preprocess, one depth sort, one tile sort, one blend).
// mode "loop":    one pipeline per view inside a single call; the backward still reduces over views in-kernel.
//
// Memory: only what the backward reads is kept after the forward (per-(view, Gaussian) mean2d, conic+opacity
// and colour, the sorted instance list, tile / pixel / bucket buffers). Sort keys, the spare halves of the
// double buffers, screen bounds, tile counts and cub workspaces are freed when the forward returns.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <vector>
#include <tuple>
#include "mv_kernels.cuh"

namespace faster_gs_mv {

    using namespace faster_gs::rasterization;
    namespace k = faster_gs_mv::kernels;

    static torch::Tensor alloc_bytes(size_t n, const torch::Device& device) {
        return torch::empty({static_cast<int64_t>(std::max<size_t>(n, 1))}, torch::dtype(torch::kUInt8).device(device));
    }
    template <typename T> static torch::Tensor alloc(size_t count, const torch::Device& device) {
        return alloc_bytes(count * sizeof(T), device);
    }
    template <typename T> static T* p(const torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

    struct Params {
        const float3* means; const float3* scales; const float4* rotations; const float* opacities;
        const float3* sh0; const float3* sh_rest; const float3* bg;
        uint n_primitives, active_sh_bases, total_sh_bases;
        uint width, height;
        float near_plane, far_plane;
        bool proper_antialiasing;
    };

    // Saved per pipeline (one per call in batched mode, one per view in loop mode)
    enum Saved { kInstances = 0, kRanges, kBucketsOffset, kMaxNProcessed, kFinalT, kNProcessed, kBucketTile, kBucketCT, kNSaved };

    // Compact per-(view, Gaussian) storage shared by all pipelines of a call. Visible pairs get slots
    // 0..n_visible-1 in the order the preprocess finds them; pv_to_c maps (view, Gaussian) to the slot.
    struct Compact {
        torch::Tensor mean2d, conic_opacity, color, n_touched, screen_bounds, local_view, depth_a, values_a;
        torch::Tensor pv_to_c, counters;
        uint capacity;
    };

    // Visible fraction of (view, Gaussian) pairs in the last call, plus a margin. Sizes the compact storage
    // of the next call; a call that finds more visible pairs than fit reruns with the exact count.
    static double g_visible_fraction = 1.0;
    static constexpr double kVisibleMargin = 1.15;

    template <typename KeyT>
    static void instances_and_blend(
        const Params& P, uint n_views, uint grid_x, uint grid_y, uint n_tiles_view, uint n_tiles, int end_bit,
        uint n_visible, uint n_instances,
        const uint* c_sorted, const uint* offsets, const Compact& C, float* image,
        std::vector<torch::Tensor>& saved, int64_t& n_buckets_out,
        const torch::Device& device, cudaStream_t stream)
    {
        torch::Tensor instance_keys_a = alloc<KeyT>(n_instances, device), instance_keys_b = alloc<KeyT>(n_instances, device);
        torch::Tensor instance_idx_a = alloc<uint>(n_instances, device), instance_idx_b = alloc<uint>(n_instances, device);
        cub::DoubleBuffer<KeyT> keys(p<KeyT>(instance_keys_a), p<KeyT>(instance_keys_b));
        cub::DoubleBuffer<uint> values(p<uint>(instance_idx_a), p<uint>(instance_idx_b));
        torch::Tensor& ranges = saved[kRanges];
        if (n_instances > 0) {
            k::create_instances_cu<KeyT><<<div_round_up(n_visible, (uint)config::block_size_create_instances), config::block_size_create_instances, 0, stream>>>(
                c_sorted, offsets, p<ushort4>(C.screen_bounds), p<float2>(C.mean2d), p<float4>(C.conic_opacity),
                p<uint>(C.local_view), keys.Current(), values.Current(), grid_x, n_tiles_view, n_visible);
            size_t ws = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, ws, keys, values, n_instances, 0, end_bit, stream);
            torch::Tensor workspace = alloc_bytes(ws, device);
            cub::DeviceRadixSort::SortPairs(p<char>(workspace), ws, keys, values, n_instances, 0, end_bit, stream);
            k::extract_instance_ranges_cu<KeyT><<<div_round_up(n_instances, (uint)config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, stream>>>(
                keys.Current(), p<uint2>(ranges), n_instances);
        }
        saved[kInstances] = values.selector == 0 ? instance_idx_a : instance_idx_b;

        torch::Tensor tile_n_buckets = alloc<uint>(n_tiles, device);
        torch::Tensor& buckets_offset = saved[kBucketsOffset];
        k::extract_bucket_counts_cu<<<div_round_up(n_tiles, (uint)config::block_size_extract_bucket_counts), config::block_size_extract_bucket_counts, 0, stream>>>(
            p<uint2>(ranges), p<uint>(tile_n_buckets), n_tiles);
        size_t ws = 0;
        cub::DeviceScan::InclusiveSum(nullptr, ws, p<uint>(tile_n_buckets), p<uint>(buckets_offset), n_tiles, stream);
        torch::Tensor workspace = alloc_bytes(ws, device);
        cub::DeviceScan::InclusiveSum(p<char>(workspace), ws, p<uint>(tile_n_buckets), p<uint>(buckets_offset), n_tiles, stream);
        // exact bucket count (one small readback) instead of the n_instances / 32 + n_tiles bound
        uint n_buckets = 0;
        cudaMemcpyAsync(&n_buckets, p<uint>(buckets_offset) + (n_tiles - 1), sizeof(uint), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        n_buckets_out = n_buckets;

        saved[kBucketTile] = alloc<uint>(n_buckets, device);
        saved[kBucketCT] = alloc<float4>(static_cast<size_t>(n_buckets) * config::block_size_blend, device);
        const dim3 grid(grid_x, grid_y, n_views);
        const dim3 block(config::tile_width, config::tile_height, 1);
        k::blend_cu<<<grid, block, 0, stream>>>(
            p<uint2>(ranges), p<uint>(buckets_offset), p<uint>(saved[kInstances]),
            p<ushort4>(C.screen_bounds), p<float2>(C.mean2d), p<float4>(C.conic_opacity), p<float3>(C.color), P.bg, image,
            p<float>(saved[kFinalT]), p<uint>(saved[kMaxNProcessed]), p<uint>(saved[kNProcessed]),
            p<uint>(saved[kBucketTile]), p<float4>(saved[kBucketCT]),
            P.width, P.height, grid_x, n_tiles_view);
    }

    // One pipeline: views [view_offset, view_offset + n_views) of the call.
    static void pipeline_forward(
        const Params& P, uint n_views, uint view_offset,
        const float4* w2c, const float4* intrinsics, const float3* cam_position, float* image,
        Compact& C, uint2& h_counters, std::vector<torch::Tensor>& saved, int64_t& n_buckets,
        const torch::Device& device, cudaStream_t stream)
    {
        const uint N = P.n_primitives;
        const uint grid_x = div_round_up(P.width, (uint)config::tile_width);
        const uint grid_y = div_round_up(P.height, (uint)config::tile_height);
        const uint n_tiles_view = grid_x * grid_y;
        const uint n_tiles = n_views * n_tiles_view;
        const uint n_pixels = P.width * P.height;
        const int end_bit = std::max(1, extract_end_bit(n_tiles - 1));

        saved.assign(kNSaved, torch::Tensor());
        saved[kRanges] = alloc<uint2>(n_tiles, device);
        cudaMemsetAsync(saved[kRanges].data_ptr(), 0, sizeof(uint2) * n_tiles, stream);
        saved[kBucketsOffset] = alloc<uint>(n_tiles, device);
        saved[kMaxNProcessed] = alloc<uint>(n_tiles, device);
        saved[kFinalT] = alloc<float>(static_cast<size_t>(n_views) * n_pixels, device);
        saved[kNProcessed] = alloc<uint>(static_cast<size_t>(n_views) * n_pixels, device);

        k::preprocess_cu<<<div_round_up(N, (uint)config::block_size_preprocess), config::block_size_preprocess, 0, stream>>>(
            P.means, P.scales, P.rotations, P.opacities, P.sh0, P.sh_rest,
            w2c, intrinsics, cam_position,
            p<uint>(C.depth_a), p<uint>(C.values_a), p<uint>(C.n_touched), p<ushort4>(C.screen_bounds),
            p<float2>(C.mean2d), p<float4>(C.conic_opacity), p<float3>(C.color), p<uint>(C.local_view),
            p<uint>(C.pv_to_c), p<uint2>(C.counters), C.capacity,
            N, n_views, view_offset, grid_x, grid_y, P.active_sh_bases, P.total_sh_bases,
            static_cast<float>(P.width), static_cast<float>(P.height), P.near_plane, P.far_plane, P.proper_antialiasing);

        const uint2 before = h_counters;
        cudaMemcpyAsync(&h_counters, C.counters.data_ptr(), sizeof(uint2), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (h_counters.x > C.capacity) return;  // overflow, the caller reruns with more storage
        const uint base = before.x;
        const uint n_visible = h_counters.x - before.x;
        const uint n_instances = h_counters.y - before.y;

        // depth sort of this pipeline's slots [base, base + n_visible)
        torch::Tensor depth_b = alloc<uint>(n_visible, device), values_b = alloc<uint>(n_visible, device);
        cub::DoubleBuffer<uint> depth_keys(p<uint>(C.depth_a) + base, p<uint>(depth_b));
        cub::DoubleBuffer<uint> values(p<uint>(C.values_a) + base, p<uint>(values_b));
        torch::Tensor offsets = alloc<uint>(n_visible, device);
        if (n_visible > 0) {
            size_t ws_sort = 0, ws_scan = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, ws_sort, depth_keys, values, n_visible, 0, 32, stream);
            cub::DeviceScan::ExclusiveSum(nullptr, ws_scan, p<uint>(offsets), p<uint>(offsets), n_visible, stream);
            size_t ws = std::max(ws_sort, ws_scan);
            torch::Tensor workspace = alloc_bytes(ws, device);
            cub::DeviceRadixSort::SortPairs(p<char>(workspace), ws, depth_keys, values, n_visible, 0, 32, stream);
            k::apply_depth_ordering_cu<<<div_round_up(n_visible, (uint)config::block_size_apply_depth_ordering), config::block_size_apply_depth_ordering, 0, stream>>>(
                values.Current(), p<uint>(C.n_touched), p<uint>(offsets), n_visible);
            cub::DeviceScan::ExclusiveSum(p<char>(workspace), ws, p<uint>(offsets), p<uint>(offsets), n_visible, stream);
        }

        if (end_bit <= 16) {
            instances_and_blend<ushort>(P, n_views, grid_x, grid_y, n_tiles_view, n_tiles, end_bit, n_visible, n_instances,
                values.Current(), p<uint>(offsets), C, image, saved, n_buckets, device, stream);
        } else {
            instances_and_blend<uint>(P, n_views, grid_x, grid_y, n_tiles_view, n_tiles, end_bit, n_visible, n_instances,
                values.Current(), p<uint>(offsets), C, image, saved, n_buckets, device, stream);
        }
    }

    static Params make_params(
        const torch::Tensor& means, const torch::Tensor& scales, const torch::Tensor& rotations,
        const torch::Tensor& opacities, const torch::Tensor& sh0, const torch::Tensor& sh_rest,
        const torch::Tensor& bg, int active_sh_bases, int width, int height, float near_plane, float far_plane,
        bool proper_antialiasing)
    {
        Params P;
        P.means = reinterpret_cast<const float3*>(means.data_ptr<float>());
        P.scales = reinterpret_cast<const float3*>(scales.data_ptr<float>());
        P.rotations = reinterpret_cast<const float4*>(rotations.data_ptr<float>());
        P.opacities = opacities.data_ptr<float>();
        P.sh0 = sh0.defined() ? reinterpret_cast<const float3*>(sh0.data_ptr<float>()) : nullptr;
        P.sh_rest = reinterpret_cast<const float3*>(sh_rest.data_ptr<float>());
        P.bg = reinterpret_cast<const float3*>(bg.data_ptr<float>());
        P.n_primitives = means.size(0);
        P.active_sh_bases = active_sh_bases;
        P.total_sh_bases = sh_rest.size(1);
        P.width = width; P.height = height;
        P.near_plane = near_plane; P.far_plane = far_plane;
        P.proper_antialiasing = proper_antialiasing;
        return P;
    }

    // Returns image [V, 3, H, W], the tensors the backward needs and the bucket count of each pipeline.
    std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<int64_t>>
    forward(
        const torch::Tensor& means, const torch::Tensor& scales, const torch::Tensor& rotations,
        const torch::Tensor& opacities, const torch::Tensor& sh0, const torch::Tensor& sh_rest,
        const torch::Tensor& w2c, const torch::Tensor& intrinsics, const torch::Tensor& cam_position,
        const torch::Tensor& bg, int active_sh_bases, int width, int height, float near_plane, float far_plane,
        bool proper_antialiasing, bool batched)
    {
        const at::cuda::CUDAGuard device_guard(means.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const torch::Device device = means.device();
        const Params P = make_params(means, scales, rotations, opacities, sh0, sh_rest, bg, active_sh_bases,
                                     width, height, near_plane, far_plane, proper_antialiasing);
        const uint V = w2c.size(0), N = P.n_primitives;
        const size_t n_pixels = static_cast<size_t>(width) * height;
        const auto f_opts = torch::TensorOptions().dtype(torch::kFloat).device(device);
        torch::Tensor image = torch::empty({V, 3, height, width}, f_opts);
        const size_t NV = static_cast<size_t>(V) * N;
        const float4* w2c_p = reinterpret_cast<const float4*>(w2c.data_ptr<float>());
        const float4* intr_p = reinterpret_cast<const float4*>(intrinsics.data_ptr<float>());
        const float3* cam_p = reinterpret_cast<const float3*>(cam_position.data_ptr<float>());

        const uint n_pipelines = batched ? 1 : V;
        const uint views_per_pipeline = batched ? V : 1;
        Compact C;
        C.capacity = static_cast<uint>(std::min<double>(NV, std::ceil(g_visible_fraction * NV) + 1024));
        std::vector<torch::Tensor> pipelines_saved;
        std::vector<int64_t> out_n_buckets;
        uint2 h_counters;
        for (int attempt = 0; attempt < 2; attempt++) {
            C.mean2d = alloc<float2>(C.capacity, device);
            C.conic_opacity = alloc<float4>(C.capacity, device);
            C.color = alloc<float3>(C.capacity, device);
            C.n_touched = alloc<uint>(C.capacity, device);
            C.screen_bounds = alloc<ushort4>(C.capacity, device);
            C.local_view = alloc<uint>(C.capacity, device);
            C.depth_a = alloc<uint>(C.capacity, device);
            C.values_a = alloc<uint>(C.capacity, device);
            C.pv_to_c = alloc<uint>(NV, device);
            cudaMemsetAsync(C.pv_to_c.data_ptr(), 0xff, sizeof(uint) * NV, stream);
            C.counters = alloc<uint2>(1, device);
            cudaMemsetAsync(C.counters.data_ptr(), 0, sizeof(uint2), stream);
            pipelines_saved.clear();
            out_n_buckets.clear();
            h_counters = make_uint2(0, 0);
            for (uint i = 0; i < n_pipelines && h_counters.x <= C.capacity; i++) {
                std::vector<torch::Tensor> saved;
                int64_t n_buckets = 0;
                pipeline_forward(P, views_per_pipeline, i * views_per_pipeline,
                                 w2c_p + 4 * i * views_per_pipeline, intr_p + i * views_per_pipeline, cam_p + i * views_per_pipeline,
                                 image.data_ptr<float>() + i * views_per_pipeline * 3 * n_pixels,
                                 C, h_counters, saved, n_buckets, device, stream);
                pipelines_saved.insert(pipelines_saved.end(), saved.begin(), saved.end());
                out_n_buckets.push_back(n_buckets);
            }
            if (h_counters.x <= C.capacity) break;
            // more visible pairs than the storage holds: rerun with all of them (loop mode: an upper bound)
            C.capacity = batched ? h_counters.x : static_cast<uint>(NV);
        }
        const int64_t n_visible = h_counters.x;
        g_visible_fraction = std::min(1.0, kVisibleMargin * static_cast<double>(n_visible) / std::max<size_t>(1, NV));
        // the visible slots (views into the storage, which has at most the margin to spare)
        std::vector<torch::Tensor> out_saved = {
            C.mean2d.narrow(0, 0, std::max<int64_t>(1, n_visible * sizeof(float2))),
            C.conic_opacity.narrow(0, 0, std::max<int64_t>(1, n_visible * sizeof(float4))),
            C.color.narrow(0, 0, std::max<int64_t>(1, n_visible * sizeof(float3))),
            C.pv_to_c,
            torch::tensor({n_visible}, torch::kInt64),
        };
        out_saved.insert(out_saved.end(), pipelines_saved.begin(), pipelines_saved.end());
        return {image, out_saved, out_n_buckets};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward(
        const torch::Tensor& grad_image, const torch::Tensor& image,
        const torch::Tensor& means, const torch::Tensor& scales, const torch::Tensor& rotations,
        const torch::Tensor& opacities, const torch::Tensor& sh_rest,
        const torch::Tensor& w2c, const torch::Tensor& intrinsics, const torch::Tensor& cam_position,
        const torch::Tensor& bg, const std::vector<torch::Tensor>& saved, const std::vector<int64_t>& n_buckets,
        int active_sh_bases, int width, int height, bool proper_antialiasing, bool batched)
    {
        const at::cuda::CUDAGuard device_guard(means.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const torch::Device device = means.device();
        const Params P = make_params(means, scales, rotations, opacities, torch::Tensor(), sh_rest, bg, active_sh_bases,
                                     width, height, 0.0f, 0.0f, proper_antialiasing);
        const uint V = w2c.size(0), N = P.n_primitives;
        const size_t n_pixels = static_cast<size_t>(width) * height;
        const auto f_opts = torch::TensorOptions().dtype(torch::kFloat).device(device);
        const torch::Tensor grad_image_c = grad_image.contiguous();

        const torch::Tensor& mean2d = saved[0];
        const torch::Tensor& conic_opacity = saved[1];
        const torch::Tensor& color = saved[2];
        const torch::Tensor& pv_to_c = saved[3];
        const size_t n_visible = static_cast<size_t>(saved[4].item<int64_t>());
        constexpr int kNShared = 5;

        // per visible (view, Gaussian) gradients: mean2d (2), conic (3 planes), opacity (1), colour (3)
        torch::Tensor helpers = torch::zeros({static_cast<int64_t>(std::max<size_t>(1, 9 * n_visible))}, f_opts);
        float* h = helpers.data_ptr<float>();
        float2* g_mean2d = reinterpret_cast<float2*>(h);
        float* g_conic = h + 2 * n_visible;
        float* g_opacity = h + 5 * n_visible;
        float3* g_color = reinterpret_cast<float3*>(h + 6 * n_visible);

        const uint grid_x = div_round_up((uint)width, (uint)config::tile_width);
        const uint grid_y = div_round_up((uint)height, (uint)config::tile_height);
        const uint n_tiles_view = grid_x * grid_y;
        const uint n_pipelines = batched ? 1 : V;
        const uint views_per_pipeline = batched ? V : 1;
        for (uint i = 0; i < n_pipelines; i++) {
            if (n_buckets[i] == 0) continue;
            const torch::Tensor* s = saved.data() + kNShared + i * kNSaved;
            const size_t img_off = static_cast<size_t>(i) * views_per_pipeline * 3 * n_pixels;
            k::blend_backward_cu<<<n_buckets[i], 32, 0, stream>>>(
                p<uint2>(s[kRanges]), p<uint>(s[kBucketsOffset]), p<uint>(s[kInstances]),
                p<float2>(mean2d), p<float4>(conic_opacity), p<float3>(color), P.bg,
                grad_image_c.data_ptr<float>() + img_off, image.data_ptr<float>() + img_off,
                p<float>(s[kFinalT]), p<uint>(s[kMaxNProcessed]), p<uint>(s[kNProcessed]),
                p<uint>(s[kBucketTile]), p<float4>(s[kBucketCT]),
                g_mean2d, g_conic, g_opacity, g_color,
                static_cast<uint>(n_visible), width, height, grid_x, n_tiles_view, views_per_pipeline * n_tiles_view,
                proper_antialiasing);
        }

        torch::Tensor grad_means = torch::empty({N, 3}, f_opts);
        torch::Tensor grad_scales = torch::empty({N, 3}, f_opts);
        torch::Tensor grad_rotations = torch::empty({N, 4}, f_opts);
        torch::Tensor grad_opacities = torch::empty({N, 1}, f_opts);
        torch::Tensor grad_sh0 = torch::empty({N, 1, 3}, f_opts);
        torch::Tensor grad_sh_rest = torch::empty({N, (int64_t)P.total_sh_bases, 3}, f_opts);
        const uint block = config::block_size_preprocess_backward;
        k::preprocess_backward_cu<<<div_round_up(N, block), block, block * P.total_sh_bases * sizeof(float3), stream>>>(
            P.means, P.scales, P.rotations, P.opacities, P.sh_rest,
            reinterpret_cast<const float4*>(w2c.data_ptr<float>()),
            reinterpret_cast<const float4*>(intrinsics.data_ptr<float>()),
            reinterpret_cast<const float3*>(cam_position.data_ptr<float>()),
            p<uint>(pv_to_c), g_mean2d, g_conic, g_opacity, g_color, static_cast<uint>(n_visible),
            reinterpret_cast<float3*>(grad_means.data_ptr<float>()),
            reinterpret_cast<float3*>(grad_scales.data_ptr<float>()),
            reinterpret_cast<float4*>(grad_rotations.data_ptr<float>()),
            grad_opacities.data_ptr<float>(),
            reinterpret_cast<float3*>(grad_sh0.data_ptr<float>()),
            reinterpret_cast<float3*>(grad_sh_rest.data_ptr<float>()),
            N, V, P.active_sh_bases, P.total_sh_bases,
            static_cast<float>(width), static_cast<float>(height), proper_antialiasing);
        return {grad_means, grad_scales, grad_rotations, grad_opacities, grad_sh0, grad_sh_rest};
    }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &faster_gs_mv::forward);
    m.def("backward", &faster_gs_mv::backward);
}
