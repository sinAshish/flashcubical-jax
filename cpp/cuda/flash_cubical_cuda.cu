#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// Device memory allocation for constant zero-persistence table (256 entries for 2D)
__constant__ unsigned char d_LUT2D[256];

// CUDA prepass kernel for 2D images
__global__ void prepass_2d_cuda_kernel(
    const double* births,
    int* parent,
    double* deaths,
    bool* has_zp,
    int n, int m, int v
) {
    int curr = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr >= v) return;

    double curr_birth = births[curr];
    int r = curr / m;
    int c = curr % m;

    unsigned char n_mask = 0;

    // Up (r > 0)
    if (r > 0) {
        int nbr = curr - m;
        double birth_up = births[nbr];
        if (birth_up < curr_birth || (birth_up == curr_birth && nbr < curr)) {
            n_mask |= 1;
        }
    }
    // Down (r < n - 1)
    if (r < n - 1) {
        int nbr = curr + m;
        double birth_down = births[nbr];
        if (birth_down < curr_birth || (birth_down == curr_birth && nbr < curr)) {
            n_mask |= 2;
        }
    }
    // Left (c > 0)
    if (c > 0) {
        int nbr = curr - 1;
        double birth_left = births[nbr];
        if (birth_left < curr_birth || (birth_left == curr_birth && nbr < curr)) {
            n_mask |= 4;
        }
    }
    // Right (c < m - 1)
    if (c < m - 1) {
        int nbr = curr + 1;
        double birth_right = births[nbr];
        if (birth_right < curr_birth || (birth_right == curr_birth && nbr < curr)) {
            n_mask |= 8;
        }
    }

    // Query LUT from __constant__ memory
    unsigned char zp_val = d_LUT2D[n_mask];

    int best_older = curr;
    if (zp_val == 0) best_older = curr - m;
    else if (zp_val == 1) best_older = curr + m;
    else if (zp_val == 2) best_older = curr - 1;
    else if (zp_val == 3) best_older = curr + 1;

    bool is_zero_persistence = (zp_val != 255) && (n_mask > 0);

    parent[curr] = is_zero_persistence ? best_older : curr;
    deaths[curr] = is_zero_persistence ? curr_birth : INFINITY;
    has_zp[curr] = is_zero_persistence;
}

__global__ void pointer_jump_cuda_kernel(
    const int* in_parent,
    int* out_parent,
    int v
) {
    int curr = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr >= v) return;
    int p = in_parent[curr];
    out_parent[curr] = in_parent[p];
}

struct SaddleEdge {
    double val;
    int u;
    int v;
};

static void resolve_saddles_host(
    const double* h_births,
    const int* h_parent,
    const bool* h_has_zp,
    double* h_deaths,
    const std::vector<SaddleEdge>& saddles,
    int v
) {
    std::vector<int> order(saddles.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return saddles[a].val < saddles[b].val;
    });

    std::vector<int> uf_parent(v);
    std::iota(uf_parent.begin(), uf_parent.end(), 0);

    for (int idx : order) {
        int ru = saddles[idx].u;
        while (uf_parent[ru] != ru) {
            uf_parent[ru] = uf_parent[uf_parent[ru]];
            ru = uf_parent[ru];
        }
        int rv = saddles[idx].v;
        while (uf_parent[rv] != rv) {
            uf_parent[rv] = uf_parent[uf_parent[rv]];
            rv = uf_parent[rv];
        }
        if (ru != rv) {
            double bu = h_births[ru];
            double bv = h_births[rv];
            int old_r, yng_r;
            if (bu < bv || (bu == bv && ru < rv)) {
                old_r = ru;
                yng_r = rv;
            } else {
                old_r = rv;
                yng_r = ru;
            }
            uf_parent[yng_r] = old_r;
            h_deaths[yng_r] = saddles[idx].val;
        }
    }

    for (int i = 0; i < v; ++i) {
        if (h_has_zp[i]) {
            h_deaths[i] = h_births[i];
        }
    }
}

// Host binding called from Python (2D)
torch::Tensor compute_h0_cuda(torch::Tensor births, int n, int m) {
    TORCH_CHECK(births.is_cuda(), "births must be a CUDA tensor");
    births = births.contiguous();
    const int v = n * m;

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(births.device());
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(births.device());
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(births.device());

    auto parent = torch::arange(v, options_int);
    auto next_parent = torch::empty({v}, options_int);
    auto deaths = torch::full({v}, INFINITY, options_double);
    auto has_zp = torch::zeros({v}, options_bool);

    // Initialize full 4-bit LUT configuration for constant memory (all masks 1..15)
    std::vector<unsigned char> h_LUT2D(256, 255);
    for (int mask = 1; mask < 16; ++mask) {
        if (mask & 1) h_LUT2D[mask] = 0;      // Up
        else if (mask & 2) h_LUT2D[mask] = 1; // Down
        else if (mask & 4) h_LUT2D[mask] = 2; // Left
        else if (mask & 8) h_LUT2D[mask] = 3; // Right
    }

    cudaMemcpyToSymbol(d_LUT2D, h_LUT2D.data(), 256 * sizeof(unsigned char));

    int threads = 512;
    int blocks = (v + threads - 1) / threads;

    prepass_2d_cuda_kernel<<<blocks, threads>>>(
        births.data_ptr<double>(),
        parent.data_ptr<int>(),
        deaths.data_ptr<double>(),
        has_zp.data_ptr<bool>(),
        n, m, v
    );

    for (int step = 0; step < 21; ++step) {
        pointer_jump_cuda_kernel<<<blocks, threads>>>(
            parent.data_ptr<int>(),
            next_parent.data_ptr<int>(),
            v
        );
        std::swap(parent, next_parent);
    }

    auto h_births_t = births.cpu();
    auto h_parent_t = parent.cpu();
    auto h_has_zp_t = has_zp.cpu();
    auto h_deaths_t = torch::full({v}, INFINITY, torch::TensorOptions().dtype(torch::kFloat64));

    const double* h_births = h_births_t.data_ptr<double>();
    const int* h_parent = h_parent_t.data_ptr<int>();
    const bool* h_has_zp = h_has_zp_t.data_ptr<bool>();
    double* h_deaths = h_deaths_t.data_ptr<double>();

    std::vector<SaddleEdge> saddles;
    saddles.reserve(v);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < m; ++c) {
            int u = r * m + c;
            if (c + 1 < m) {
                int v_nbr = u + 1;
                int ru = h_parent[u], rv = h_parent[v_nbr];
                if (ru != rv) {
                    saddles.push_back({std::max(h_births[u], h_births[v_nbr]), ru, rv});
                }
            }
            if (r + 1 < n) {
                int v_nbr = u + m;
                int ru = h_parent[u], rv = h_parent[v_nbr];
                if (ru != rv) {
                    saddles.push_back({std::max(h_births[u], h_births[v_nbr]), ru, rv});
                }
            }
        }
    }

    resolve_saddles_host(h_births, h_parent, h_has_zp, h_deaths, saddles, v);
    return h_deaths_t.to(births.device());
}

__global__ void prepass_3d_cuda_kernel(
    const double* births,
    int* parent,
    double* deaths,
    bool* has_zp,
    int d, int n, int m, int v
) {
    int curr = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr >= v) return;

    double curr_birth = births[curr];
    int z = curr / (n * m);
    int r = (curr / m) % n;
    int c = curr % m;

    int best_older = curr;
    double min_older_birth = curr_birth;
    bool found_older = false;

#define CHECK_NEIGHBOR(nbr) \
    do { \
        double birth_nbr = births[(nbr)]; \
        if (birth_nbr < curr_birth || (birth_nbr == curr_birth && (nbr) < curr)) { \
            if (!found_older || birth_nbr < min_older_birth || (birth_nbr == min_older_birth && (nbr) < best_older)) { \
                best_older = (nbr); \
                min_older_birth = birth_nbr; \
                found_older = true; \
            } \
        } \
    } while (0)

    if (z > 0) CHECK_NEIGHBOR(curr - n * m);
    if (z < d - 1) CHECK_NEIGHBOR(curr + n * m);
    if (r > 0) CHECK_NEIGHBOR(curr - m);
    if (r < n - 1) CHECK_NEIGHBOR(curr + m);
    if (c > 0) CHECK_NEIGHBOR(curr - 1);
    if (c < m - 1) CHECK_NEIGHBOR(curr + 1);
#undef CHECK_NEIGHBOR

    parent[curr] = found_older ? best_older : curr;
    deaths[curr] = found_older ? curr_birth : INFINITY;
    has_zp[curr] = found_older;
}

torch::Tensor compute_h0_cuda_3d(torch::Tensor births, int d, int n, int m) {
    TORCH_CHECK(births.is_cuda(), "births must be a CUDA tensor");
    births = births.contiguous();
    const int v = d * n * m;

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(births.device());
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(births.device());
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(births.device());

    auto parent = torch::arange(v, options_int);
    auto next_parent = torch::empty({v}, options_int);
    auto deaths = torch::full({v}, INFINITY, options_double);
    auto has_zp = torch::zeros({v}, options_bool);

    int threads = 512;
    int blocks = (v + threads - 1) / threads;

    prepass_3d_cuda_kernel<<<blocks, threads>>>(
        births.data_ptr<double>(),
        parent.data_ptr<int>(),
        deaths.data_ptr<double>(),
        has_zp.data_ptr<bool>(),
        d, n, m, v
    );

    for (int step = 0; step < 21; ++step) {
        pointer_jump_cuda_kernel<<<blocks, threads>>>(
            parent.data_ptr<int>(),
            next_parent.data_ptr<int>(),
            v
        );
        std::swap(parent, next_parent);
    }

    auto h_births_t = births.cpu();
    auto h_parent_t = parent.cpu();
    auto h_has_zp_t = has_zp.cpu();
    auto h_deaths_t = torch::full({v}, INFINITY, torch::TensorOptions().dtype(torch::kFloat64));

    const double* h_births = h_births_t.data_ptr<double>();
    const int* h_parent = h_parent_t.data_ptr<int>();
    const bool* h_has_zp = h_has_zp_t.data_ptr<bool>();
    double* h_deaths = h_deaths_t.data_ptr<double>();

    std::vector<SaddleEdge> saddles;
    saddles.reserve(v);
    const int plane = n * m;
    for (int z = 0; z < d; ++z) {
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < m; ++c) {
                int u = z * plane + r * m + c;
                if (c + 1 < m) {
                    int v_nbr = u + 1;
                    int ru = h_parent[u], rv = h_parent[v_nbr];
                    if (ru != rv) {
                        saddles.push_back({std::max(h_births[u], h_births[v_nbr]), ru, rv});
                    }
                }
                if (r + 1 < n) {
                    int v_nbr = u + m;
                    int ru = h_parent[u], rv = h_parent[v_nbr];
                    if (ru != rv) {
                        saddles.push_back({std::max(h_births[u], h_births[v_nbr]), ru, rv});
                    }
                }
                if (z + 1 < d) {
                    int v_nbr = u + plane;
                    int ru = h_parent[u], rv = h_parent[v_nbr];
                    if (ru != rv) {
                        saddles.push_back({std::max(h_births[u], h_births[v_nbr]), ru, rv});
                    }
                }
            }
        }
    }

    resolve_saddles_host(h_births, h_parent, h_has_zp, h_deaths, saddles, v);
    return h_deaths_t.to(births.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_h0_cuda_3d", &compute_h0_cuda_3d, "Compute H0 persistent homology on GPU using CUDA 3D");
    m.def("compute_h0_cuda", &compute_h0_cuda, "Compute H0 persistent homology on GPU using CUDA");
}
