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

using namespace mblas;

int dgetrf(cublasHandle_t cublas, int m, int n, double* a, int lda, int* ipiv)
{
    constexpr int nb = 512;

    MatrixView A(a, m, n, lda);

    for (int j = 0; j < n; j += nb)
    {
        int jb = std::min(nb, n-j);

        auto panel = A.subview(j, j, m-j, jb);

        LAPACKE_dgetrf(LAPACK_COL_MAJOR, panel.nrows, panel.ncols, panel.data, panel.stride, &ipiv[j]);

        auto leftA = A.subview(j, 0, m-j, j);
        auto rightA = A.subview(j, j+jb, m-j, n-j-jb);

        dlaswp(cublas, leftA, 1, jb, &ipiv[j], 1);
        dlaswp(cublas, rightA, 1, jb, &ipiv[j], 1);

        for (int i = 0; i < jb; i++)
            ipiv[j + i] += j;
        
        if (n - (j+jb) > 0) {
            auto tileA = A.subview(j, j, jb, jb);
            auto U = A.subview(j, j+jb, jb, n - (j+jb));
            auto L = A.subview(j+jb, j, m - (j+jb), jb);
            auto restA = A.subview(j+jb, j+jb, m - (j+jb), n - (j+jb));

            dtrsm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, 
                  1.0, tileA, U);
            
            dgemm(cublas, -1.0, L, U, 1.0, restA);

            cudaDeviceSynchronize();
        }
    }
    return 0;
}

void sync_all(int ngpus)
{
    for (int gpu = 0; gpu < ngpus; gpu++)
        CUDA_CALL(cudaSetDevice(gpu));
}

int m_dgetrf(int ngpus, cudaStream_t* streams, cublasHandle_t* handles, int m, int n, double* a, int lda, int* ipiv)
{
    constexpr int nb = 512;

    MatrixView A(a, m, n, lda);

    for (int j = 0; j < n; j += nb)
    {
        int jb = std::min(nb, n-j);

        auto panel = A.subview(j, j, m-j, jb);

        LAPACKE_dgetrf(LAPACK_COL_MAJOR, panel.nrows, panel.ncols, panel.data, panel.stride, &ipiv[j]);

        auto leftA = A.subview(j, 0, m-j, j);
        auto rightA = A.subview(j, j+jb, m-j, n-j-jb);

        MatrixView rightAcols[ngpus], leftAcols[ngpus], Lpanel[ngpus];
        auto leftAcolsTiles = partitionCols(leftA, ngpus);
        auto rightAcolsTiles = partitionCols(rightA, ngpus);
        for (int gpu = 0; gpu < ngpus; gpu++)
        {
            CUDA_CALL(cudaSetDevice(gpu));
            leftAcols[gpu] = leftA.localcopy(leftAcolsTiles[gpu], streams[gpu]);
            rightAcols[gpu] = rightA.localcopy(rightAcolsTiles[gpu], streams[gpu]);
            Lpanel[gpu] = panel.localcopy(0, 0, panel.nrows, panel.ncols, streams[gpu]);
        }

        m_dlaswp(ngpus, handles, leftAcols, 1, jb, &ipiv[j], 1);
        m_dlaswp(ngpus, handles, rightAcols, 1, jb, &ipiv[j], 1);

        for (int i = 0; i < jb; i++)
            ipiv[j + i] += j;
        
        if (n - (j+jb) > 0) {
            MatrixView As[ngpus], Ls[ngpus], restAs[ngpus], Us[ngpus];
            for (int gpu = 0; gpu < ngpus; gpu++)
            {
                As[gpu] = Lpanel[gpu].subview(0, 0, jb, jb);
                Ls[gpu] = Lpanel[gpu].subview(jb, 0, Lpanel[gpu].nrows-jb, Lpanel[gpu].ncols);
                Us[gpu] = rightAcols[gpu].subview(0, 0, jb, rightAcols[gpu].ncols);
                restAs[gpu] = rightAcols[gpu].subview(jb, 0, rightAcols[gpu].nrows-jb, rightAcols[gpu].ncols);
            }

            m_dtrsm(ngpus, handles, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, 
                    1.0, As, Us);
            
            m_dgemm(ngpus, handles, -1.0, Ls, Us, 1.0, restAs);
        }
        for (int gpu = 0; gpu < ngpus; gpu++)
        {
            leftA.copyback(leftAcolsTiles[gpu].row, leftAcolsTiles[gpu].col,
                           leftAcols[gpu], streams[gpu]);
            rightA.copyback(rightAcolsTiles[gpu].row, rightAcolsTiles[gpu].col,
                           rightAcols[gpu], streams[gpu]);
        }
        for (int gpu = 0; gpu < ngpus; gpu++)
        {
            leftAcols[gpu].free();
            rightAcols[gpu].free();
            Lpanel[gpu].free();
        }
        sync_all(ngpus);
    }
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
    if (ngpus == 1)
        dgetrf(handles[0], m, n, A, m, ipiv);
    else
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

