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
};

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

void m_dgemm(int ngpus, cudaStream_t* streams, cublasHandle_t* handles, double alpha, const MatrixView& A, const MatrixView& B, double beta, MatrixView& C)
{
    assert(A.nrows == C.nrows);
    assert(A.ncols == B.nrows);
    assert(B.ncols == C.ncols);

    A.setreadmostly(ngpus);
    B.setreadmostly(ngpus);

    int gpu = 0;
    std::vector<Tile> Ctiles = partitionCols(C);
    for (auto tile : Ctiles) {
        auto Ctile = C.subview(tile);
        
        auto Atile = A.subview(tile.row, 0, tile.nrows, A.ncols);
        auto Btile = B.subview(0, tile.col, B.nrows, tile.ncols);

        Ctile.prefetch(streams[gpu], gpu);
        Atile.prefetch(streams[gpu], gpu);
        Btile.prefetch(streams[gpu], gpu);

        CUDA_CALL(cudaSetDevice(gpu));
        dgemm(handles[gpu], alpha, Atile, Btile, beta, Ctile);

        gpu = (gpu+1) % ngpus;
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

void m_dtrsm(int ngpus, cudaStream_t* streams, cublasHandle_t* handles, cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag,
             double alpha, const MatrixView& A, MatrixView& B)
{
    assert(A.nrows == A.ncols);
    assert(A.nrows == B.nrows);

    A.setreadmostly(ngpus);

    int gpu = 0;
    auto Btiles = partitionCols(B);
    for (auto tile : Btiles)
    {
        auto Btile = B.subview(tile);
        A.prefetch(streams[gpu], gpu);
        Btile.prefetch(streams[gpu], gpu);

        CUDA_CALL(cudaSetDevice(gpu));
        dtrsm(handles[gpu], side, uplo, diag, alpha, A, Btile);

        gpu = (gpu+1)%ngpus;
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

int m_dlaswp(int ngpus, cudaStream_t* streams, cublasHandle_t* handles, MatrixView& A, int k1, int k2, const int* ipiv, int incx)
{
    assert(incx == 1);

    if (A.ncols == 0) return 0;

    int gpu = 0;
    auto Atiles = partitionCols(A);
    for (auto tile : Atiles)
    {
        auto Atile = A.subview(tile);
        Atile.prefetch(streams[gpu], gpu);

        CUDA_CALL(cudaSetDevice(gpu));
        dlaswp(handles[gpu], Atile, k1, k2, ipiv, incx);
        gpu = (gpu+1)%ngpus;
    }
    return 0;
}

}
