#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

// for CUB_WRAPPER
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Intersect.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

__forceinline__ __device__ float fast_sqrt_f32(float x) {
    float y;
    asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ bool segment_intersect_ellipse(float a, float b, float c, float d, float l, float r) {
    float delta = b * b - 4.0f * a * c;
    float t1 = (l - d) * (2.0f * a) + b;
    float t2 = (r - d) * (2.0f * a) + b;
    return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

__forceinline__ __device__ bool block_intersect_ellipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power) {
    float a, b, c, dx, dy;
    float w = 2.0f * power;

    if (center.x * 2.0f < pix_min.x + pix_max.x) {
        dx = center.x - pix_min.x;
    } else {
        dx = center.x - pix_max.x;
    }
    a = conic.z;
    b = -2.0f * conic.y * dx;
    c = conic.x * dx * dx - w;

    if (segment_intersect_ellipse(a, b, c, center.y, pix_min.y, pix_max.y)) {
        return true;
    }

    if (center.y * 2.0f < pix_min.y + pix_max.y) {
        dy = center.y - pix_min.y;
    } else {
        dy = center.y - pix_max.y;
    }
    a = conic.x;
    b = -2.0f * conic.y * dy;
    c = conic.z * dy * dy - w;

    return segment_intersect_ellipse(a, b, c, center.x, pix_min.x, pix_max.x);
}

__forceinline__ __device__ bool block_contains_center(int2 pix_min, int2 pix_max, float2 center) {
    return center.x >= pix_min.x && center.x <= pix_max.x &&
           center.y >= pix_min.y && center.y <= pix_max.y;
}

__forceinline__ __device__ void getRect(const float2 p, float radius, int width, int height,
                                      int2& rect_min, int2& rect_max) {
    rect_min = {
        min(width, max(0, (int)(p.x - radius))),
        min(height, max(0, (int)(p.y - radius)))
    };
    rect_max = {
        min(width, max(0, (int)(p.x + radius) + 1)),
        min(height, max(0, (int)(p.y + radius) + 1))
    };
}

template <typename scalar_t>
__global__ void intersect_tile_kernel(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const scalar_t *__restrict__ conics,             // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ opacities,          // [C, N] or [nnz]
    const scalar_t *__restrict__ depths,             // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    // Early culling
    const bool first_pass = cum_tiles_per_gauss == nullptr;
    scalar_t depth = depths[idx];
    if (depth <= 0.2f) {
        if (first_pass) tiles_per_gauss[idx] = 0;
        return;
    }

    if (opacities[idx] <= 1.0f / 255.f) {
        if (first_pass) tiles_per_gauss[idx] = 0;
        return;
    }

    const scalar_t radius = radii[idx];
    if (radius <= 0.f) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    // Convert to tile space and compute conic
    float2 center = {static_cast<scalar_t>(mean2d.x / static_cast<scalar_t>(tile_size)),
                    static_cast<scalar_t>(mean2d.y / static_cast<scalar_t>(tile_size))};
    scalar_t tile_radius = static_cast<scalar_t>(radius / static_cast<scalar_t>(tile_size));

    // Compute conic parameters
    const float3 conic = {conics[3 * idx], conics[3 * idx + 1], conics[3 * idx + 2]};
    float power = 4.0f; // Power tuning parameter

    // Get tile bounds
    int2 tile_min, tile_max;
    getRect(center, tile_radius, tile_width, tile_height, tile_min, tile_max);

    // Single tile optimization
    const bool single_tile = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y) == 1;
    const int64_t depth_id = __float_as_int(static_cast<float>(depth));
    if (single_tile) {
        int2 pix_min = {tile_min.x * (int)tile_size, tile_min.y * (int)tile_size};
        int2 pix_max = {pix_min.x + (int)tile_size - 1, pix_min.y + (int)tile_size - 1};

        float2 mean2d_float = {static_cast<float>(mean2d.x), static_cast<float>(mean2d.y)};
        if (block_contains_center(pix_min, pix_max, mean2d_float) ||
            block_intersect_ellipse(pix_min, pix_max, mean2d_float, conic, power)) {

            if (first_pass) {
                tiles_per_gauss[idx] = 1;
            } else {
                int64_t cid = packed ? camera_ids[idx] : idx / N;
                int64_t tile_id = tile_min.y * tile_width + tile_min.x;

                int64_t key = (cid << (32 + tile_n_bits)) | (tile_id << 32) | depth_id;
                int64_t offset = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];

                isect_ids[offset] = key;
                flatten_ids[offset] = idx;
            }
        } else if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    // Multi-tile processing
    int32_t tile_count = 0;
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int2 pix_min = {j * (int)tile_size, i * (int)tile_size};
            int2 pix_max = {pix_min.x + (int)tile_size - 1, pix_min.y + (int)tile_size - 1};

            float2 mean2d_float = {static_cast<float>(mean2d.x), static_cast<float>(mean2d.y)};
            if (block_contains_center(pix_min, pix_max, mean2d_float) ||
                block_intersect_ellipse(pix_min, pix_max, mean2d_float, conic, power)) {

                if (first_pass) {
                    tile_count++;
                } else {
                    int64_t cid = packed ? camera_ids[idx] : idx / N;
                    int64_t tile_id = i * tile_width + j;

                    int64_t key = (cid << (32 + tile_n_bits)) | (tile_id << 32) | depth_id;
                    int64_t offset = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
                    offset += tile_count;

                    isect_ids[offset] = key;
                    flatten_ids[offset] = idx;
                    tile_count++;
                }
            }
        }
    }

    if (first_pass) {
        tiles_per_gauss[idx] = tile_count;
    }
}

void launch_intersect_tile_kernel(
    // inputs
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N] or [nnz]
    const at::Tensor conics,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [C, N] or [nnz]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [C, N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [C, N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;
    int64_t n_elements;
    if (packed) {
        nnz = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    } else {
        N = means2d.size(1); // number of gaussians per camera
        n_elements = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_kernel",
        [&]() {
            intersect_tile_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids.has_value()
                        ? camera_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    gaussian_ids.has_value()
                        ? gaussian_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    means2d.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    conics.data_ptr<scalar_t>(),
                    opacities.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    tiles_per_gauss.has_value()
                        ? tiles_per_gauss.value().data_ptr<int32_t>()
                        : nullptr,
                    isect_ids.has_value()
                        ? isect_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    flatten_ids.has_value()
                        ? flatten_ids.value().data_ptr<int32_t>()
                        : nullptr
                );
        }
    );
}

__global__ void intersect_offset_kernel(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t C,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid,
        // tile_id) pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t cid_prev = isect_id_prev >> tile_n_bits;
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

void launch_intersect_offset_kernel(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [C, tile_height, tile_width]
) {
    int64_t n_elements = isect_ids.size(0); // total number of intersections
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        offsets.fill_(0);
        return;
    }

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    intersect_offset_kernel<<<
        grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        n_elements,
        isect_ids.data_ptr<int64_t>(),
        C,
        n_tiles,
        tile_n_bits,
        offsets.data_ptr<int32_t>()
    );
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t tile_n_bits,
    const uint32_t cam_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>()
    );
    CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        0,
        32 + tile_n_bits + cam_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch (d_values.selector) {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }

    // Double buffer is better than naive radix sort, in terms of mem usage.
    // CUB_WRAPPER(
    //     cub::DeviceRadixSort::SortPairs,
    //     isect_ids,
    //     isect_ids_sorted,
    //     flatten_ids,
    //     flatten_ids_sorted,
    //     n_isects,
    //     0,
    //     32 + tile_n_bits + cam_n_bits,
    //     stream
    // );
}

} // namespace gsplat
