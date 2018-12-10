#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cblas.h>
#include <lapacke.h>

#include "util.h"

#include "MatrixView.hpp"
#include "Range.hpp"

#include "tasks.hpp"

using namespace mblas;


struct DgetrfArgs {
    int m, n;
    double* A;
    int lda;
    int* ipiv;

    cudaStream_t stream;
};

void dgetrf_func(int device_id, void* p) {
    DgetrfArgs* args = static_cast<DgetrfArgs*>(p);
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, args->m, args->n, args->A, args->lda, args->ipiv);
}

struct DlaswpArgs {
    int m, n;
    double* A;
    int lda;

    int k1, k2;
    int* ipiv;
};

void dlaswp_func(int device_id, void* p) {
    DlaswpArgs* args = static_cast<DlaswpArgs*>(p);
    LAPACKE_dlaswp(LAPACK_COL_MAJOR, args->n, args->A, args->lda,
                   args->k1, args->k2, args->ipiv, 1);
}

struct AllocArgs {
    int m, jb, ncols;
    double** panel;
    double** aslice;
};

void allocate_temp_func(int device_id, void* p) {
    AllocArgs* args = static_cast<AllocArgs*>(p);

    CUDA_CALL(cudaSetDevice(device_id));
    CUDA_CALL(cudaMalloc(args->panel, sizeof(double) * args->m * args->jb));
    CUDA_CALL(cudaMalloc(args->aslice, sizeof(double) * args->m * args->ncols));
}

struct FreeArgs {
    double** panel;
    double** aslice;
};

void free_temp_func(int device_id, void* p) {
    FreeArgs* args = static_cast<FreeArgs*>(p);
    CUDA_CALL(cudaSetDevice(device_id));
    CUDA_CALL(cudaFree(*args->panel));
    CUDA_CALL(cudaFree(*args->aslice));
}

struct CopyRectArgs {
    double** A_slice;
    double* A;
    int lda;

    int row_begin, row_end;
    int col_begin, col_end;

    cudaStream_t stream;
};

void copy_rect_func(int device_id, void* p) {
    CopyRectArgs* args = static_cast<CopyRectArgs*>(p);

    int nrows = args->row_end - args->row_begin;
    int ncols = args->col_end - args->col_begin;

    double* A_start = args->A + (args->row_begin + args->col_begin * args->lda);

    CUDA_CALL(cudaSetDevice(device_id));
    CUDA_CALL(cudaMemcpy2DAsync(*args->A_slice, nrows * sizeof(double),
                                A_start,        args->lda * sizeof(double),
                                nrows * sizeof(double), ncols,
                                cudaMemcpyDefault, args->stream));
}

struct CopyBackArgs {
    double* A;
    int lda;
    double** A_slice;

    int row_begin, row_end;
    int col_begin, col_end;

    cudaStream_t stream;
};

void copy_back_func(int device_id, void* p) {
    CopyBackArgs* args = static_cast<CopyBackArgs*>(p);

    int nrows = args->row_end - args->row_begin;
    int ncols = args->col_end - args->col_begin;

    double* A_start = args->A + (args->row_begin + args->col_begin * args->lda);

    CUDA_CALL(cudaSetDevice(device_id));
    CUDA_CALL(cudaMemcpy2DAsync(A_start,        args->lda * sizeof(double),
                                *args->A_slice, nrows * sizeof(double),
                                nrows * sizeof(double), ncols,
                                cudaMemcpyDefault, args->stream));
}

struct CuDlaswpArgs {
    int m, n;
    double** A;
    int lda;

    int k1, k2;
    int* ipiv;
    cublasHandle_t handle;
};

void cudlaswp_func(int device_id, void* p) {
    CuDlaswpArgs* args = static_cast<CuDlaswpArgs*>(p);

    CUDA_CALL(cudaSetDevice(device_id));
    CUBLAS_CALL(cublasDlaswp(args->handle, args->m, args->n, *args->A, args->lda,
                 args->k1, args->k2, args->ipiv, 1));
}

struct CuDtrsmArgs {
    int m, n;
    double** A;
    int lda;
    double** B;
    int ldb;
    cublasHandle_t handle;
};

void cudtrsm_func(int device_id, void* p) {
    CuDtrsmArgs* args = static_cast<CuDtrsmArgs*>(p);

    double ONE = 1.0;

    CUDA_CALL(cudaSetDevice(device_id));
    CUBLAS_CALL(cublasDtrsm(args->handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                            args->m, args->n,
                            &ONE,
                            *args->A, args->lda,
                            *args->B, args->ldb));
}

struct CuDgemmArgs {
    int jb;
    int m, n, k;
    double** A;
    int lda;
    double** B;
    int ldb;
    double** C;
    int ldc;
    cublasHandle_t handle;
};

void cudgemm_func(int device_id, void* p) {
    CuDgemmArgs* args = static_cast<CuDgemmArgs*>(p);

    double ONE = 1.0;
    double MINUS_ONE = -1.0;

    double* a = *args->A;
    double* b = *args->B;
    double* c = *args->C;

    CUDA_CALL(cudaSetDevice(device_id));    
    CUBLAS_CALL(cublasDgemm(args->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            args->m, args->n, args->k,
                            &MINUS_ONE,
                            &a[args->jb], args->lda,
                            b, args->ldb,
                            &ONE,
                            &c[args->jb], args->ldc));
}

struct CuSynchronizeArgs { int dummy; };

void cusynchronize(int device_id, void* p) {
    CUDA_CALL(cudaSetDevice(device_id));    
    CUDA_CALL(cudaDeviceSynchronize());
}

int dgetrf(int ngpus, cudaStream_t* streams, cublasHandle_t* handles, int m, int n,
           double* a, int lda, int* ipiv) {

    constexpr int nb = 1024;

    GpuTaskScheduler scheduler(ngpus, streams);

    int next_iteration_event = NULL_EVENT;

    double** panel   = (double**)malloc(sizeof(double*) * ngpus);
    double** aslices = (double**)malloc(sizeof(double*) * ngpus);

    std::vector<Range> ranges = partition_min(0, n, ngpus, nb);
    for (int gpu = 0; gpu < ngpus; gpu++) {
        Range range = ranges[gpu];

        if (range.size()) {
            CUDA_CALL(cudaSetDevice(gpu));
            CUDA_CALL(cudaMalloc(&panel[gpu], sizeof(double) * m * nb));
            CUDA_CALL(cudaMalloc(&aslices[gpu], sizeof(double) * m * range.size()));
        }
    }

    for (int j = 0; j < n; j += nb) {
        int jb = std::min(nb, n-j);

        DgetrfArgs dgetrf_args{m-j, jb, &a[j + j*lda], lda, &ipiv[j]};
        int dgetrf_task = scheduler.enqueue_task("dgetrf", CPU_DEVICE, next_iteration_event,
                                                 dgetrf_func, &dgetrf_args);

        next_iteration_event = scheduler.aggregate_event({dgetrf_task, next_iteration_event});

        DlaswpArgs swap_left_args{m-j, j, &a[j], lda, 1, jb, &ipiv[j]};
        int swap_left_task = scheduler.enqueue_task("dlaswp", CPU_DEVICE, next_iteration_event,
                                                    dlaswp_func, &swap_left_args);
        std::vector<int> done_events;
        done_events.reserve(ngpus+1);
        done_events.push_back(swap_left_task);

        std::vector<Range> ranges = partition_min(j+jb, n, ngpus, jb);

        for (int gpu = 0; gpu < ngpus; gpu++) {
            if (ranges[gpu].size()) {
                auto subranges = partition_tiles(ranges[gpu].begin, ranges[gpu].end, 4*nb);
                for (auto range : subranges) {
                    //
                    // Copy over the part of A this GPU is working on
                    //
                    CopyRectArgs copy_aslice_args{&aslices[gpu], a, lda, j, m, range.begin, range.end, streams[gpu]};
                    scheduler.enqueue_task("copy A slice", gpu, next_iteration_event, copy_rect_func, &copy_aslice_args);
                    
                    //
                    // Copy over the panel updated by the last 'dgetrf' call
                    // 
                    CopyRectArgs copy_panel_args{&panel[gpu], a, lda, j, m, j, j+jb, streams[gpu]};
                    scheduler.enqueue_task("copy panel", gpu, dgetrf_task, copy_rect_func, &copy_panel_args);

                    //
                    // Apply row swaps to A right of the current panel
                    // 
                    CuDlaswpArgs cudlaswp_args{m-j, range.size(), &aslices[gpu], m-j, 1, jb, &ipiv[j], handles[gpu]};
                    scheduler.enqueue_task("cudlaswp", gpu, dgetrf_task, cudlaswp_func, &cudlaswp_args);


                    //
                    // Update the upper part of my A slice
                    //
                    CuDtrsmArgs cudtrsm_args{jb, range.size(), &panel[gpu], m-j, &aslices[gpu], m-j, handles[gpu]};
                    scheduler.enqueue_task("cudtrsm", gpu, NULL_EVENT, cudtrsm_func, &cudtrsm_args);

                    //
                    // Update the rest of my A slice
                    //
                    CuDgemmArgs cudgemm_args{jb, m-(j+jb), range.size(), jb,
                                             &panel[gpu], m-j, 
                                             &aslices[gpu], m-j,
                                             &aslices[gpu], m-j,
                                             handles[gpu]};
                    scheduler.enqueue_task("cudgemm", gpu, NULL_EVENT, cudgemm_func, &cudgemm_args);

                    //
                    // Copy back my updated slice
                    //
                    CopyBackArgs copy_back_args{a, lda, &aslices[gpu], j, m, range.begin, range.end};
                    scheduler.enqueue_task("copy back", gpu, NULL_EVENT, copy_back_func, &copy_back_args);

                    CuSynchronizeArgs sync_args{};
                    int done_event = scheduler.enqueue_task("synchronize", gpu, NULL_EVENT, cusynchronize, &sync_args);
                    done_events.push_back(done_event);
                }
            }

        }

        next_iteration_event = scheduler.aggregate_event(done_events);
    }

    scheduler.run();

    for (int gpu = 0; gpu < ngpus; gpu++) {
        Range range = ranges[gpu];

        if (range.size()) {
            CUDA_CALL(cudaSetDevice(gpu));
            CUDA_CALL(cudaFree(panel[gpu]));
            CUDA_CALL(cudaFree(aslices[gpu]));
        }
    }

    for (int j = 0; j < n; j += nb) {
        int jb = std::min(nb, n-j);

        //
        // Update the pivoted rows to global row indices
        // 
        for (int i = 0; i < jb; i++)
            ipiv[j + i] += j;
    }
#ifdef PROFILE_TASKS
    for (size_t i = 0; i < scheduler.durations.size(); i++) {
        Duration d = scheduler.durations[i];
        printf("%d\t%s\t%f\t%f\t%f\n", d.device_id, d.name, d.start_time, d.stop_time, d.stop_time-d.start_time);
    }
#endif
    return 0;
}


int main(int argc, char** argv) {
    constexpr int TEST_SIZE = 10000;

    int m = 8192;
    if (argc >= 2)
        m = std::stoi(argv[1]);

    int n = m;
    int lda = m;

    int ngpus;
    CUDA_CALL(cudaGetDeviceCount(&ngpus));

    cudaStream_t streams[ngpus];
    cublasHandle_t handles[ngpus];

    for (int gpu = 0; gpu < ngpus; gpu++) {
        CUDA_CALL(cudaSetDevice(gpu));
        CUDA_CALL(cudaStreamCreate(&streams[gpu]));
        cublasCreate(&handles[gpu]);
        cublasSetStream(handles[gpu], streams[gpu]);
    }
    CUDA_CALL(cudaSetDevice(0));


    double* A;
    double* Acopy = NULL;
    double* b;

    size_t data_size = sizeof(double) * lda * n;
    CUDA_CALL(cudaMallocManaged(&A, data_size));
    CUDA_CALL(cudaMallocManaged(&b, sizeof(double) * m));


    // Generate random coefficient matrix and result vector
    //
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 2343ULL));
    CURAND_CALL(curandGenerateUniformDouble(gen, A, lda*n));
    CURAND_CALL(curandGenerateUniformDouble(gen, b, m));
    CURAND_CALL(curandDestroyGenerator(gen));

    // Copy A for later reference
    //
    if (m < TEST_SIZE)
    {
        Acopy = (double*)malloc(data_size);
        CUDA_CALL(cudaMemcpy(Acopy, A, data_size, cudaMemcpyDefault));
    }

    // Run LU factorization
    //
    int* ipiv = (int*)malloc(sizeof(int) * m);
    

    double startTime = get_real_time();
    dgetrf(ngpus, streams, handles, m, n, A, m, ipiv);
    double endTime = get_real_time();
    
    printf("ngpus=%d,matrix_size=%d,time_s=%f\n", ngpus, n, endTime-startTime);

    // Extract the solution vector
    //
    if (m < TEST_SIZE)
    {
        double* x = (double*)malloc(sizeof(double) * n);
        CUDA_CALL(cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDefault));

        double* Ahost = (double*)malloc(data_size);
        CUDA_CALL(cudaMemcpy(Ahost, A, data_size, cudaMemcpyDefault));
        LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, 1, Ahost, lda, ipiv, x, n);

        // Check the solution
        //
        double* b2 = (double*)malloc(sizeof(double) * n);
        cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, 1.0, Acopy, lda,
                    x, 1, 0.0, b2, 1);

        for (int i = 0; i < n; i++) {
            double x = b[i];
            double y = b2[i];
            if (!(std::abs(x-y) < 0.00001)) {
                printf("Result vectors are not equal:\n");
                printf("b[%d] = %f | b2[%d] = %f\n", i, b[i], i, b2[i]);
                return 0;
            }
        }
    }
    return 0;
}

