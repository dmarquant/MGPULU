#pragma once
#include <cstdio>
#include <cassert>
#include <cublas_v2.h>

#include "util.h"


namespace mblas
{

struct Tile { int row, col, nrows, ncols; };

/// A matrix in col major format lending memory.
/// 
class MatrixView
{
public:
    double* data;

    int nrows, ncols;

    int stride; /// Stride to next value in same row

    MatrixView() : data(NULL), nrows(0), ncols(0), stride(0) {}
    MatrixView(double* A, int m, int n, int lda) : data(A), nrows(m), ncols(n), stride(lda) {}

    void prefetch(cudaStream_t stream, int gpu) const
    {
        // Prefetch more (contiguous block)
        size_t size = nrows + (ncols-1) * stride;
        CUDA_CALL(cudaMemPrefetchAsync(data, size * sizeof(double), gpu, stream));
    }

    void setreadmostly(int ngpus) const
    {
        // Advise on more (contiguous block)
        //size_t size = nrows + (ncols-1) * stride;
        //for (int gpu = 0; gpu < ngpus; gpu++)
            //CUDA_CALL(cudaMemAdvise(data, size * sizeof(double), cudaMemAdviseSetReadMostly, gpu));
    }


    MatrixView subview(int row, int col, int nrows, int ncols)
    {
        return MatrixView{data + row + col * stride, nrows, ncols, stride};
    }
    MatrixView subview(Tile t)
    {
        return subview(t.row, t.col, t.nrows, t.ncols);
    }
    const MatrixView subview(int row, int col, int nrows, int ncols) const
    {
        return MatrixView{data + row + col * stride, nrows, ncols, stride};
    }
    const MatrixView subview(Tile t) const
    {
        return subview(t.row, t.col, t.nrows, t.ncols);
    }

    MatrixView localcopy(int row, int col, int nrows, int ncols, cudaStream_t stream)
    {
        MatrixView mv{NULL, nrows, ncols, nrows};
        if (nrows * ncols != 0)
        {
            CUDA_CALL(cudaMalloc(&mv.data, sizeof(double) * nrows * ncols));
            CUDA_CALL(cudaMemcpy2DAsync(mv.data, sizeof(double) * nrows, 
                                        this->data + row + col * this->stride, sizeof(double) * this->stride,
                                        nrows * sizeof(double), ncols, cudaMemcpyDefault,
                                        stream));
        }
        return mv;
    }

    MatrixView localcopy(int row, int col, int nrows, int ncols)
    {
        MatrixView mv{NULL, nrows, ncols, nrows};
        if (nrows * ncols != 0)
        {
            CUDA_CALL(cudaMalloc(&mv.data, sizeof(double) * nrows * ncols));
            CUDA_CALL(cudaMemcpy2D(mv.data, sizeof(double) * nrows, 
                                   this->data + row + col * this->stride, sizeof(double) * this->stride,
                                    nrows * sizeof(double), ncols, cudaMemcpyDefault));
        }
        return mv;
    }

    MatrixView localcopy(Tile t, cudaStream_t stream)
    {
        return localcopy(t.row, t.col, t.nrows, t.ncols, stream);
    }

    void copyback(int row, int col, MatrixView& mv, cudaStream_t stream)
    {
        CUDA_CALL(cudaMemcpy2DAsync(this->data + row + col * this->stride, sizeof(double) * this->stride,
                                    mv.data, sizeof(double) * mv.nrows, 
                                    mv.nrows * sizeof(double), mv.ncols, cudaMemcpyDefault,
                                    stream));
    }

    void free()
    {
        CUDA_CALL(cudaFree(data));
    }
};

std::vector<Tile> partitionCols(MatrixView& M, int ngpus) {
    int perGpu = M.ncols / ngpus;

    std::vector<Tile> tiles; tiles.reserve(ngpus);
    if (M.ncols == 0)
    {
        for (int gpu = 0; gpu < ngpus; gpu++)
            tiles.push_back({0, 0, 0, 0});
    }
    else
    {
        for (int col = 0; col < M.ncols; col += perGpu) {
            tiles.push_back({0, col, M.nrows, perGpu});
        }
        tiles.back().ncols = M.ncols - tiles.back().col;
    }
    return tiles;
}

std::vector<Tile> partitionCols(MatrixView& M) {
    const int TILE_SIZE = 1024 * 2;

    std::vector<Tile> tiles;
    tiles.reserve(M.ncols / TILE_SIZE + 1);

    for (int col = 0; col < M.ncols; col += TILE_SIZE) {
        tiles.push_back({0, col, M.nrows, std::min(TILE_SIZE, M.ncols-col)});
    }
    return tiles;
};

void dgemm(cublasHandle_t handle, double alpha, const MatrixView& A, const MatrixView& B, double beta, MatrixView& C)
{
    assert(A.nrows == C.nrows);
    assert(A.ncols == B.nrows);
    assert(B.ncols == C.ncols);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.nrows, B.ncols, A.ncols, &alpha,
                A.data, A.stride, B.data, B.stride, &beta, C.data, C.stride);
}

void m_dgemm(int ngpus, cublasHandle_t* handles, double alpha, const MatrixView* As, const MatrixView* Bs, double beta, MatrixView* Cs)
{
    for (int gpu = 0; gpu < ngpus; gpu++)
    {
        CUDA_CALL(cudaSetDevice(gpu));
        dgemm(handles[gpu], alpha, As[gpu], Bs[gpu], beta, Cs[gpu]);
    }
    CUDA_CALL(cudaSetDevice(0));
}

void dtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag,
           double alpha, const MatrixView& A, MatrixView& B)
{
    assert(A.nrows == A.ncols);
    assert(A.nrows == B.nrows);

    cublasDtrsm(handle, side, uplo, CUBLAS_OP_N, diag, B.nrows, B.ncols, &alpha, A.data, A.stride,
                B.data, B.stride);
}

void m_dtrsm(int ngpus, cublasHandle_t* handles, cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag,
             double alpha, const MatrixView* As, MatrixView* Bs)
{
    for (int gpu = 0; gpu < ngpus; gpu++)
    {
        CUDA_CALL(cudaSetDevice(gpu));
        dtrsm(handles[gpu], side, uplo, diag, alpha, As[gpu], Bs[gpu]);
    }
    CUDA_CALL(cudaSetDevice(0));
}

int dlaswp(cublasHandle_t handle, MatrixView& A, int k1, int k2, const int* ipiv, int incx)
{
    assert(incx == 1);

    if (A.ncols == 0) return 0;

    for (int i = k1-1; i < k2; i++)
    {
        cublasDswap(handle, A.ncols, &A.data[i], A.stride, &A.data[ipiv[i]-1], A.stride);
    }
    return 0;
}

int dlaswp(cublasHandle_t handle, int off, MatrixView& A, int k1, int k2, const int* ipiv, int incx)
{
    assert(incx == 1);

    if (A.ncols == 0) return 0;

    for (int i = k1-1; i < k2; i++)
    {
        cublasDswap(handle, A.ncols, &A.data[i-off], A.stride, &A.data[ipiv[i]-1-off], A.stride);
        CUDA_CALL(cudaDeviceSynchronize());
    }
    return 0;
}

int m_dlaswp(int ngpus, cublasHandle_t* handles, MatrixView* As, int k1, int k2, const int* ipiv, int incx)
{
    assert(incx == 1);

    for (int gpu = 0; gpu < ngpus; gpu++)
    {
        CUDA_CALL(cudaSetDevice(gpu));
        dlaswp(handles[gpu], As[gpu], k1, k2, ipiv, incx);
    }
    CUDA_CALL(cudaSetDevice(0));
    return 0;
}

}
