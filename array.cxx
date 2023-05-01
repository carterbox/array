#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <omp.h>
#include <string.h>

namespace nb = nanobind;

NB_MODULE(custom, m) {
  m.def("copy",
        [](nb::ndarray<float, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>
               input,
           nb::ndarray<float, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>
               output) {
          size_t num_threads = 16;
          size_t chunk_size = input.size() / num_threads;
          size_t remainder = input.size() % num_threads;
          // for (size_t i = 0; i < input.size(); ++i) {
          //   output(i) = input(i);
          // }
#pragma omp parallel for
          for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = start + chunk_size;
            if (i < remainder){
              start = start + i - 1;
              end = end + i;
            }
            memcpy(
              output.data() + start,
              input.data() + start,
              (end - start) * sizeof(float)
            );
          }
        });

  m.def("copy1",
        [](nb::ndarray<float, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>
               input) {
          size_t num_threads = 16;
          size_t chunk_size = input.size() / num_threads;
          size_t remainder = input.size() % num_threads;
          // for (size_t i = 0; i < input.size(); ++i) {
          //   output(i) = input(i);
          // }
          float * copied_data = (float*)malloc(input.size() * sizeof(float));

#pragma omp parallel for
          for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = start + chunk_size;
            if (i < remainder){
              start = start + i - 1;
              end = end + i;
            }
            memcpy(
              copied_data + start,
              input.data() + start,
              (end - start) * sizeof(float)
            );
          }
        size_t * shape = (size_t*)malloc(input.ndim() * sizeof(size_t));
        for (size_t n = 0; n < input.ndim(); ++n){
          shape[n] = input.shape(n);
        }
        return nb::ndarray<nb::numpy, float>(copied_data, input.ndim(), shape);
        });

  m.def("copyd",
        [](nb::ndarray<double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>
               input,
           nb::ndarray<double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>
               output) {
          size_t num_threads = 16;
          size_t chunk_size = input.size() / num_threads;
          size_t remainder = input.size() % num_threads;
          // for (size_t i = 0; i < input.size(); ++i) {
          //   output(i) = input(i);
          // }
#pragma omp parallel for
          for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = start + chunk_size;
            if (i < remainder){
              start = start + i - 1;
              end = end + i;
            }
            memcpy(
              output.data() + start,
              input.data() + start,
              (end - start) * sizeof(double)
            );
          }
        });

  m.def("copy1d",
        [](nb::ndarray<double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>
               input) {
          size_t num_threads = 16;
          size_t chunk_size = input.size() / num_threads;
          size_t remainder = input.size() % num_threads;
          // for (size_t i = 0; i < input.size(); ++i) {
          //   output(i) = input(i);
          // }
          double * copied_data = (double*)malloc(input.size() * sizeof(double));

#pragma omp parallel for
          for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = start + chunk_size;
            if (i < remainder){
              start = start + i - 1;
              end = end + i;
            }
            memcpy(
              copied_data + start,
              input.data() + start,
              (end - start) * sizeof(double)
            );
          }
        size_t * shape = (size_t*)malloc(input.ndim() * sizeof(size_t));
        for (size_t n = 0; n < input.ndim(); ++n){
          shape[n] = input.shape(n);
        }
        return nb::ndarray<nb::numpy, double>(copied_data, input.ndim(), shape);
        });

  m.def("updated",
        [](
          nb::ndarray<double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu> m,
          nb::ndarray<double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu> a,
          nb::ndarray<double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu> x,
          nb::ndarray<double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu> b
        ) {
          size_t num_threads = 16;
          size_t chunk_size = a.size() / num_threads;
          size_t remainder = a.size() % num_threads;

#pragma omp parallel for
          for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = start + chunk_size;
            if (i < remainder){
              start = start + i - 1;
              end = end + i;
            }
            for (size_t j = start; j < end; ++j){
              b(j) = m(j) + a(j) * x(j);
            }
          }
        });

  m.def("updatef",
        [](
          nb::ndarray<float, nb::shape<nb::any>, nb::c_contig, nb::device::cpu> m,
          nb::ndarray<float, nb::shape<nb::any>, nb::c_contig, nb::device::cpu> a,
          nb::ndarray<float, nb::shape<nb::any>, nb::c_contig, nb::device::cpu> x,
          nb::ndarray<float, nb::shape<nb::any>, nb::c_contig, nb::device::cpu> b
        ) {
          size_t num_threads = 16;
          size_t chunk_size = a.size() / num_threads;
          size_t remainder = a.size() % num_threads;

#pragma omp parallel for
          for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = start + chunk_size;
            if (i < remainder){
              start = start + i - 1;
              end = end + i;
            }
            for (size_t j = start; j < end; ++j){
              b(j) = m(j) + a(j) * x(j);
            }
          }
        });
}
