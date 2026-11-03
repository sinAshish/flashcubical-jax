#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "compute_2d_all.hpp"
#include "algorithms/smart_h0_h2_3d.hpp"
#include "zero_lookup/runtime/mask_build.hpp"
#include "zero_lookup/runtime/zero_table.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;

static std::uint64_t sortable_double_key(double x) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    if ((bits & (1ull << 63)) != 0) {
        return ~bits;
    }
    return bits ^ (1ull << 63);
}

static void radix_sort_vertices_by_value(std::vector<std::uint32_t>& active,
                                         const double* values) {
    const std::size_t n = active.size();
    std::vector<std::uint32_t> tmp_vertices(n);
    std::vector<std::uint64_t> keys(n);
    std::vector<std::uint64_t> tmp_keys(n);

    for (std::size_t i = 0; i < n; ++i) {
        keys[i] = sortable_double_key(values[active[i]]);
    }

    std::array<std::uint32_t, 65536> counts{};
    for (unsigned shift = 0; shift < 64; shift += 16) {
        counts.fill(0);
        for (std::uint64_t key : keys) {
            ++counts[static_cast<std::uint16_t>(key >> shift)];
        }

        std::uint32_t running = 0;
        for (std::uint32_t& count : counts) {
            const std::uint32_t bucket_size = count;
            count = running;
            running += bucket_size;
        }

        for (std::size_t i = 0; i < n; ++i) {
            const std::uint16_t bucket =
                static_cast<std::uint16_t>(keys[i] >> shift);
            const std::uint32_t out = counts[bucket]++;
            tmp_vertices[out] = active[i];
            tmp_keys[out] = keys[i];
        }

        active.swap(tmp_vertices);
        keys.swap(tmp_keys);
    }
}

static py::array_t<double> _resolve_saddles(
    py::array_t<double, py::array::c_style | py::array::forcecast> births_arr,
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> parent_arr,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask_arr,
    std::int32_t plane,
    std::int32_t cols)
{
    const auto b_buf = births_arr.request();
    const auto p_buf = parent_arr.request();
    const auto m_buf = mask_arr.request();

    const std::int32_t v = static_cast<std::int32_t>(p_buf.size);
    const double* values = static_cast<const double*>(b_buf.ptr);
    const std::int32_t* parent_ptr = static_cast<const std::int32_t*>(p_buf.ptr);
    const std::uint8_t* mask_ptr = static_cast<const std::uint8_t*>(m_buf.ptr);

    std::vector<std::uint32_t> active;
    active.reserve(static_cast<std::size_t>(v) / 2 + 1);
    for (std::int32_t y = 0; y < v; ++y) {
        if (mask_ptr[y] != 0) {
            active.push_back(static_cast<std::uint32_t>(y));
        }
    }

    if (active.size() > 4096) {
        radix_sort_vertices_by_value(active, values);
    } else {
        std::sort(active.begin(), active.end(), [&](std::uint32_t a, std::uint32_t b) {
            return values[a] < values[b] || (values[a] == values[b] && a < b);
        });
    }

    std::int32_t offsets[6];
    if (plane == 0) {
        offsets[0] = -cols;
        offsets[1] = -1;
        offsets[2] = 1;
        offsets[3] = cols;
        offsets[4] = 0;
        offsets[5] = 0;
    } else {
        offsets[0] = -plane;
        offsets[1] = -cols;
        offsets[2] = -1;
        offsets[3] = 1;
        offsets[4] = cols;
        offsets[5] = plane;
    }

    std::vector<std::int32_t> uf_parent(parent_ptr, parent_ptr + v);
    std::vector<std::array<double, 3>> pairs;
    pairs.reserve(static_cast<std::size_t>(v) / 4 + 1);

    for (std::uint32_t y : active) {
        std::uint8_t mask = mask_ptr[y];
        const double val = values[y];

        std::int32_t ru = uf_parent[y];
        while (uf_parent[ru] != ru) {
            uf_parent[ru] = uf_parent[uf_parent[ru]];
            ru = uf_parent[ru];
        }

        unsigned bit = 0;
        while (mask != 0) {
            if ((mask & 1u) != 0) {
                const std::int32_t z = static_cast<std::int32_t>(y) + offsets[bit];
                std::int32_t rv = uf_parent[z];
                while (uf_parent[rv] != rv) {
                    uf_parent[rv] = uf_parent[uf_parent[rv]];
                    rv = uf_parent[rv];
                }
                if (ru != rv) {
                    const double bu = values[ru];
                    const double bv = values[rv];
                    std::int32_t old_r, yng_r;
                    if (bu < bv || (bu == bv && ru < rv)) {
                        old_r = ru;
                        yng_r = rv;
                    } else {
                        old_r = rv;
                        yng_r = ru;
                    }
                    uf_parent[yng_r] = old_r;
                    ru = old_r;
                    const double b_yng = values[yng_r];
                    if (val != b_yng) {
                        pairs.push_back({b_yng, val, 0.0});
                    }
                }
            }
            mask >>= 1u;
            ++bit;
        }
    }

    const double inf = std::numeric_limits<double>::infinity();
    for (std::int32_t r = 0; r < v; ++r) {
        if (uf_parent[r] == r && values[r] != inf) {
            pairs.push_back({values[r], inf, 0.0});
        }
    }

    const std::size_t n_pairs = pairs.size();
    py::array_t<double> out({n_pairs, std::size_t{3}});
    auto res = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < n_pairs; ++i) {
        res(i, 0) = pairs[i][0];
        res(i, 1) = pairs[i][1];
        res(i, 2) = pairs[i][2];
    }
    return out;
}

static void _initialize_lookup_tables(const std::string& path2d,
                                      const std::string& path3d) {
    zero_lookup::load_zero_tables(std::filesystem::path(path2d),
                                  std::filesystem::path(path3d));
}

static py::array_t<double> _compute(
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    bool h1)
{
    const auto buf = x.request();
    const int ndim = static_cast<int>(buf.ndim);

    if (ndim != 2 && ndim != 3) {
        throw py::value_error(
            "cubicalp: input must be 2D or 3D, got " +
            std::to_string(ndim) + "D");
    }
    for (int i = 0; i < ndim; ++i) {
        if (buf.shape[i] <= 0) {
            throw py::value_error(
                "cubicalp: image dimensions must be positive");
        }
    }
    if (ndim == 2 && !h1) {
        throw py::value_error(
            "cubicalp: h1=False is only valid for 3D inputs; "
            "2D inputs always compute H0 and H1");
    }

    const double* values = static_cast<const double*>(buf.ptr);
    std::vector<std::array<double, 3>> pairs;

    if (ndim == 2) {
        zero_lookup::warm_zero_table2d();

        const std::size_t rows = static_cast<std::size_t>(buf.shape[0]);
        const std::size_t cols = static_cast<std::size_t>(buf.shape[1]);
        pairs = cubicalp_native::compute_2d_all(values, rows, cols);

    } else {
        zero_lookup::warm_zero_table3d();

        const std::size_t depth = static_cast<std::size_t>(buf.shape[0]);
        const std::size_t rows  = static_cast<std::size_t>(buf.shape[1]);
        const std::size_t cols  = static_cast<std::size_t>(buf.shape[2]);

        zero_lookup::MaskBuild3D masks =
            zero_lookup::build_masks3d(values, depth, rows, cols, nullptr);

        smart_h0_h2_3d::FullResult result = smart_h0_h2_3d::compute(
            values, depth, rows, cols,
            /*include_apparent_zero_pairs=*/false,
            /*value_codes=*/nullptr,
            /*compute_h1=*/h1,
            /*profile_h1=*/false,
            masks.masks.data());

        pairs.reserve(
            result.h0.pairs.size() +
            result.h1.pairs.size() +
            result.h2.pairs.size());

        for (const auto& p : result.h0.pairs) {
            pairs.push_back({p.birth, p.death, 0.0});
        }
        if (result.h1_computed) {
            for (const auto& p : result.h1.pairs) {
                pairs.push_back({p.birth, p.death, 1.0});
            }
        }
        for (const auto& p : result.h2.pairs) {
            pairs.push_back({p.birth, p.death, 2.0});
        }
    }

    const std::size_t n = pairs.size();
    py::array_t<double> out({n, std::size_t{3}});
    auto r = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < n; ++i) {
        r(i, 0) = pairs[i][0];
        r(i, 1) = pairs[i][1];
        r(i, 2) = pairs[i][2];
    }
    return out;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Cubical persistent homology – C++ backend";
    m.def("initialize_lookup_tables", &_initialize_lookup_tables,
          py::arg("path2d"), py::arg("path3d"),
          "Load packaged zero lookup tables from explicit file paths.");
    m.def("compute", &_compute,
          py::arg("x"), py::arg("h1"),
          "Compute cubical persistence pairs.\n\n"
          "Parameters\n"
          "----------\n"
          "x : ndarray, float64, C-contiguous, 2D or 3D\n"
          "h1 : bool – compute H1 (for 3D; always True for 2D)\n\n"
          "Returns\n"
          "-------\n"
          "ndarray of shape (n, 3), columns: birth, death, dim\n"
          "Infinite death values (+inf) represent essential classes.");
    m.def("resolve_saddles", &_resolve_saddles,
          py::arg("births"), py::arg("parent"), py::arg("saddle_mask"),
          py::arg("plane"), py::arg("cols"),
          "Resolve H0 saddle merges from GPU-computed pointer-jumped roots and neighbor masks.");
}
