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
    MatrixView* A;
    int* ipiv;
    int j;
};

void dgetrf_func(int device_id, void* p) {
    DgetrfArgs* args = static_cast<DgetrfArgs*>(p);

    MatrixView* A = args->A;
    int* ipiv = args->ipiv;
    int j = args->j;

    LAPACKE_dgetrf(LAPACK_COL_MAJOR, A->nrows, A->ncols, A->data, A->stride, ipiv);
    
    // Adjust indices in 'ipiv' to global indices
    for (int i = 0; i < A->ncols; i++)
        ipiv[i] += j;

}

struct DlaswpArgs {
    MatrixView* A;
    int* ipiv;
    int k1, k2;
};

struct CuDlaswpArgs {
    cublasHandle_t cublas;
    MatrixView* A;
    int* ipiv;
    int k1, k2;
};

void dlaswp_func(int device_id, void* p) {
    DlaswpArgs* args = static_cast<DlaswpArgs*>(p);
    
    LAPACKE_dlaswp(LAPACK_COL_MAJOR, args->A->ncols, args->A->data, args->A->stride,
                   args->k1, args->k2, args->ipiv, 1);    
}

void cu_dlaswp_func(int device_id, void* p) {
    CuDlaswpArgs* args = static_cast<CuDlaswpArgs*>(p);

    CUDA_CALL(cudaSetDevice(device_id));
    mblas::dlaswp(args->cublas, *args->A, args->k1, args->k2, args->ipiv, 1);
}

struct CuDtrsmArgs {
    cublasHandle_t cublas;
    MatrixView* A;
    MatrixView* B;
};

void cu_dtrsm_func(int device_id, void* p) {
    CuDtrsmArgs* args = static_cast<CuDtrsmArgs*>(p);

    CUDA_CALL(cudaSetDevice(device_id));
    mblas::dtrsm(args->cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, 
                 1.0, *args->A, *args->B);
}

struct CuDgemmArgs {
    cublasHandle_t cublas;
    MatrixView* A;
    MatrixView* B;
    MatrixView* C;
};

void cu_dgemm_func(int device_id, void* p) {
    CuDgemmArgs* args = static_cast<CuDgemmArgs*>(p);

    CUDA_CALL(cudaSetDevice(device_id));    
    mblas::dgemm(args->cublas, -1.0, *args->A, *args->B, 1.0, *args->C);
}

void cu_synchronize(int device_id, void* p) {
    CUDA_CALL(cudaSetDevice(device_id));    
    CUDA_CALL(cudaDeviceSynchronize());
}

struct Copy_U_A22_Args {
    // Current iteration of LU 
    int j;
    int jb;

    // The whole matrix
    MatrixView* A;

    // The column range of A to work on
    int col_begin;
    int col_end;

    // Blocks U and A22 (only a subset of the columns)
    // This is locally allocated memory
    MatrixView* U_A22;

    // Block U (only a view to memory U_A22)
    MatrixView* U;

    // Block A22 (only a view to memory U_A22);
    MatrixView* A22;

    cudaStream_t stream;
};

void copy_u_a22_func(int device, void* p) {
    Copy_U_A22_Args* args = static_cast<Copy_U_A22_Args*>(p);
    int j = args->j;
    int jb = args->jb;
    MatrixView* A = args->A;

    // Copy U and A22 to a buffer allocated on the device
    CUDA_CALL(cudaSetDevice(device));
    *args->U_A22 = A->localcopy(j, args->col_begin, A->nrows-j,
                                args->col_end - args->col_begin, args->stream);

    // Create views to the blocks U and A22
    *args->U   = args->U_A22->subview( 0, 0, jb, args->U_A22->ncols);
    *args->A22 = args->U_A22->subview(jb, 0, args->U_A22->nrows - jb, args->U_A22->ncols);
}

struct Copy_A11_L_Args {
    // Current iteration of LU 
    int j;
    int jb;

    // The whole matrix
    MatrixView* A;

    // Blocks A11 and L, the allocated memory local to the GPU
    MatrixView* A11_L;

    // Views to A11 and L (only views to A11_L)
    MatrixView* A11;
    MatrixView* L;

    cudaStream_t stream;
};

void copy_a11_l_func(int device, void* p) {
    Copy_A11_L_Args* args = static_cast<Copy_A11_L_Args*>(p);
    int j = args->j;
    int jb = args->jb;
    MatrixView* A = args->A;

    // Copy A11 and L to a local buffer 
    CUDA_CALL(cudaSetDevice(device));
    *args->A11_L = A->localcopy(j, j, A->nrows-j, jb, args->stream);

    *args->A11 = args->A11_L->subview(0, 0, jb, jb);
    *args->L   = args->A11_L->subview(jb, 0, args->A11_L->nrows, jb);
}

struct Copy_Back_Args {
    // Current iteration of LU 
    int j;
    int jb;

    // The whole matrix
    MatrixView* A;

    // The column range of A to work on
    int col_begin;
    int col_end;

    // Blocks U and A22 (only a subset of the columns)
    // This is locally allocated memory
    MatrixView* U_A22;

    cudaStream_t stream;
};

void copy_back_func(int device, void* p) {
    Copy_Back_Args* args = static_cast<Copy_Back_Args*>(p);
    int j = args->j;
    int jb = args->jb;
    MatrixView* A = args->A;

    CUDA_CALL(cudaSetDevice(device));
    A->copyback(j, args->col_begin, args->U_A22, args->stream);
}

int m_dgetrf(int ngpus, cudaStream_t* streams, cublasHandle_t* handles, int m, int n, double* a, int lda, int* ipiv)
{
    // Block sized used in the computation
    constexpr int nb = 512;

    MatrixView A(a, m, n, lda);

    GpuTaskScheduler scheduler(ngpus);

    // Event to wait on before starting the next iteration
    int next_iteration_event = NULL_EVENT;

    for (int j = 0; j < n; j += nb) {

        // The actual block size for this iteration (may be smaller at the edge)
        int jb = std::min(nb, n-j);

        // Do partial LU decomposition
        MatrixView* panel = new MatrixView(A.subview(j, j, m-j, jb));
        DgetrfArgs dgetrf_args{panel, &ipiv[j], j};
        int dgetrf_task = scheduler.enqueue_task("dgetrf", CPU_DEVICE, next_iteration_event, dgetrf_func, &dgetrf_args);

        // Apply row swaps left of the panel
        MatrixView* leftA = new MatrixView(A.subview(0, 0, m, j));
        DlaswpArgs left_args{leftA, ipiv, j+1, j+jb};
        int swap_left_task = scheduler.enqueue_task("dlaswp", CPU_DEVICE, dgetrf_task, dlaswp_func, &left_args);

        std::vector<MatrixView*> A11_L(ngpus), A11(ngpus), L(ngpus), U_A22(ngpus), U(ngpus), A22(ngpus);
        std::vector<int> copy_u_a22_tasks(ngpus), copy_a11_l_tasks(ngpus), copy_back_tasks(ngpus), sync_events(ngpus);
        std::vector<Range> colranges = partition(j+jb, A.ncols, ngpus);
        for (int gpu = 0; gpu < ngpus; gpu++) {
            U_A22[gpu] = new MatrixView;
            U[gpu]     = new MatrixView;
            A22[gpu]   = new MatrixView;

            Copy_U_A22_Args copy_u_a22_args{j, jb, &A, 
                                            colranges[gpu].begin, colranges[gpu].end, 
                                            &U_A22[gpu], &U[gpu], &A22[gpu], streams[gpu]};
            copy_u_a22_tasks[gpu] = scheduler.enqueue_task("copy_U_A22", gpu, next_iteration_event, copy_u_a22_func, &copy_u_a22_args);

            A11_L[gpu] = new MatrixView;
            A11[gpu]   = new MatrixView;
            L[gpu]     = new MatrixView;
            CopyA11_LArgs copy_a11_l_args{j, jb, &A, &A11_L[gpu], &A11[gpu], &L[gpu], streams[gpu]};
            copy_a11_l_tasks[gpu] = scheduler.enqueue_task("copy_A11_L", gpu, dgetrf_task, copy_a11_l_func, &copy_a11_l_args);

            int wait_to_swap = scheduler.aggregate_event({dgetrf_task, copy_u_a22_tasks[gpu]});
            CuDlaswpArgs swap_right_args{handles[gpu], &U_A22[gpu], &ipiv, j+1, j+jb};
            int cudlaswp_task = scheduler.enqueue_task("cudlaswp", gpu, wait_to_swap, cudlaswp_func, &swap_right_args);

            DtrsmArgs dtrsm_args{handles[gpu], &A11[gpu], &U[gpu]};
            int cudtrsm_task = scheduler.enqueue_task("cudtrsm", gpu, cudlaswp_task, cudtrsm_func, &dtrsm_args);

            DgemmArgs dgemm_args{handles[gpu], &L[gpu], &U[gpu], &A22[gpu]};
            int cudgemm_task = scheduler.enqueue_task("cudgemm", gpu, cudtrsm_task, cudgemm_func, &dgemm_args);

            Copy_Back_Args copy_back_args{j, jb, &A, colranges[gpu].begin, colranges[gpu].end, &U_A22[gpu]};
            copy_back_tasks[gpu] = scheduler.enqueue_task("copy_back", gpu, ..., copy_back_func, &copy_back_args);

            int dummy = 0;
            sync_events[gpu] = scheduler.enqueue_task("synchronize GPU ", 0, copy_back_task[gpu], 
                                                          cu_synchronize, &dummy);
        }
    
        sync_events.push_back(swap_left_task);

        next_iteration_event = scheduler.aggregate_event(sync_events);
    }

    scheduler.run();
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

