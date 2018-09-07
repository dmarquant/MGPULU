#pragma once
#include <cassert>
#include <cublas_v2.h>


namespace mblas
{

/// A matrix in col major format lending memory.
/// 
class MatrixView
{
public:
    double* data;

    int nrows, ncols;

    int stride; /// Stride to next value in same row

    MatrixView(double* A, int m, int n, int lda) : data(A), nrows(m), ncols(n), stride(lda) {}


    MatrixView subview(int row, int col, int nrows, int ncols)
    {
        return MatrixView{data + row + col * stride, nrows, ncols, stride};
    }
};

void dgemm(cublasHandle_t handle, double alpha, const MatrixView& A, const MatrixView& B, double beta, MatrixView& C)
{
    assert(A.nrows == C.nrows);
    assert(A.ncols == B.nrows);
    assert(B.ncols == C.ncols);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.nrows, B.ncols, A.ncols, &alpha,
                A.data, A.stride, B.data, B.stride, &beta, C.data, C.stride);
}

void dtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag,
           double alpha, const MatrixView& A, MatrixView& B)
{
    assert(A.nrows == A.ncols);
    assert(A.nrows == B.nrows);

    cublasDtrsm(handle, side, uplo, CUBLAS_OP_N, diag, B.nrows, B.ncols, &alpha, A.data, A.stride,
                B.data, B.stride);
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

}
