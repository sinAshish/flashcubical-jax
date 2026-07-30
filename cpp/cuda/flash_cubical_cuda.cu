#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
    deaths[curr] = is_zero_persistence ? curr_birth : ::atof("inf");
    has_zp[curr] = is_zero_persistence;
}

// Host binding called from Python
torch::Tensor compute_h0_cuda(torch::Tensor births, int n, int m) {
    const int v = n * m;
    
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(births.device());
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(births.device());
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(births.device());
    
    auto parent = torch::arange(v, options_int);
    auto deaths = torch::full({v}, ::atof("inf"), options_double);
    auto has_zp = torch::zeros({v}, options_bool);
    
    // Initialize standard LUT configuration for constant memory
    std::vector<unsigned char> h_LUT2D(256, 255);
    h_LUT2D[1] = 0; // Up is best
    h_LUT2D[2] = 1; // Down is best
    h_LUT2D[4] = 2; // Left is best
    h_LUT2D[8] = 3; // Right is best
    
    // Copy lookup table to GPU Constant Memory
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
    
    return deaths;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_h0_cuda", &compute_h0_cuda, "Compute H0 persistent homology on GPU using CUDA");
}
