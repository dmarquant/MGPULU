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

#include "tasks.hpp"

using namespace mblas;

constexpr int CPU_DEVICE = -1;
constexpr int NULL_EVENT = 0;

void sync_all(int ngpus)
{
    for (int gpu = 0; gpu < ngpus; gpu++)
        CUDA_CALL(cudaSetDevice(gpu));
}


struct DgetrfArgs {
    MatrixView A;
    int* ipiv;
    int j;
};

void dgetrf_func(int device_id, void* p) {
    DgetrfArgs* args = static_cast<DgetrfArgs*>(p);

    MatrixView* A = &args->A;
    int* ipiv = args->ipiv;
    int j = args->j;

    LAPACKE_dgetrf(LAPACK_COL_MAJOR, A->nrows, A->ncols, A->data, A->stride, ipiv);
    
    // Adjust indices in 'ipiv' to global indices
    for (int i = 0; i < A->ncols; i++)
        ipiv[i] += j;

}

struct DlaswpArgs {
    MatrixView A;
    int* ipiv;
    int k1, k2;
};

struct CuDlaswpArgs {
    cublasHandle_t cublas;
    MatrixView A;
    int* ipiv;
    int k1, k2;
};

void dlaswp_func(int device_id, void* p) {
    DlaswpArgs* args = static_cast<DlaswpArgs*>(p);
    
    LAPACKE_dlaswp(LAPACK_COL_MAJOR, args->A.ncols, args->A.data, args->A.stride,
                   args->k1, args->k2, args->ipiv, 1);    
}

void cu_dlaswp_func(int device_id, void* p) {
    CuDlaswpArgs* args = static_cast<CuDlaswpArgs*>(p);

    CUDA_CALL(cudaSetDevice(device_id));
    mblas::dlaswp(args->cublas, args->A, args->k1, args->k2, args->ipiv, 1);
}

struct CuDtrsmArgs {
    cublasHandle_t cublas;
    MatrixView A, B;
};

void cu_dtrsm_func(int device_id, void* p) {
    CuDtrsmArgs* args = static_cast<CuDtrsmArgs*>(p);

    CUDA_CALL(cudaSetDevice(device_id));
    mblas::dtrsm(args->cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, 
                 1.0, args->A, args->B);
}

struct CuDgemmArgs {
    cublasHandle_t cublas;
    MatrixView A, B, C;
};

void cu_dgemm_func(int device_id, void* p) {
    CuDgemmArgs* args = static_cast<CuDgemmArgs*>(p);

    CUDA_CALL(cudaSetDevice(device_id));    
    mblas::dgemm(args->cublas, -1.0, args->A, args->B, 1.0, args->C);
}

void cu_synchronize(int device_id, void* p) {
    CUDA_CALL(cudaSetDevice(device_id));    
    CUDA_CALL(cudaDeviceSynchronize());
}

/// Multi GPU implementation of LU decomposition. The input matrix is split horizontally
/// among the GPUs. During each iteration the slices are copied local to the GPUs.
///
int m_dgetrf(int ngpus, cudaStream_t* streams, cublasHandle_t* handles, int m, int n, double* a, int lda, int* ipiv)
{
    constexpr int nb = 512;
    MatrixView A(a, m, n, lda);

    GpuTaskScheduler scheduler(ngpus);
    int next_iteration_event = NULL_EVENT;

    for (int j = 0; j < n; j += nb)
    {
        int jb = std::min(nb, n-j);

        auto panel = A.subview(j, j, m-j, jb);
        DgetrfArgs dgetrf_args{panel, &ipiv[j], j};
        int dgetrf_task = scheduler.enqueue_task("dgetrf", CPU_DEVICE, next_iteration_event, dgetrf_func, &dgetrf_args);

        auto leftA = A.subview( 0,    0, m, j);
        auto rightA = A.subview(0, j+jb, m, n-j-jb);

        DlaswpArgs left_args{leftA,    ipiv, j+1, j+jb};
        CuDlaswpArgs right_args{handles[0], rightA, ipiv, j+1, j+jb};

        scheduler.enqueue_task("dlswp", CPU_DEVICE, dgetrf_task, dlaswp_func, &left_args);
        //int swp_left_task = scheduler.enqueue_task(0, dgetrf_task, cu_dlaswp_func, &left_args);
        int swp_right_task = scheduler.enqueue_task("cudlswp", 0, dgetrf_task, cu_dlaswp_func, &right_args);

        if (n - (j+jb) > 0) {
            auto tileA = A.subview(j, j, jb, jb);
            auto U = A.subview(j, j+jb, jb, n - (j+jb));
            auto L = A.subview(j+jb, j, m - (j+jb), jb);
            auto restA = A.subview(j+jb, j+jb, m - (j+jb), n - (j+jb));

            CuDtrsmArgs dtrsm_args{handles[0], tileA, U};
            int dtrsm_task = scheduler.enqueue_task("cudtrsm", 0, swp_right_task, cu_dtrsm_func, &dtrsm_args);


            CuDgemmArgs dgemm_args{handles[0], L, U, restA};
            int dgemm_task = scheduler.enqueue_task("cudgemm", 0, dtrsm_task, cu_dgemm_func, &dgemm_args);            

            int dummy = 0;
            next_iteration_event = scheduler.enqueue_task("synchronize GPU ", 0, dgemm_task, cu_synchronize, &dummy);
        }

    }
    scheduler.run();    
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
    double* Acopy;
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
#ifdef TEST_CPU
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, A, m, ipiv);
#else
    m_dgetrf(ngpus, streams, handles, m, n, A, m, ipiv);
#endif
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

