#include "../ferry/ferry.h"
#include "cublas_v2.h"
#include "util.h"
cublasHandle_t CublasHandle[ferry::MAX_GPUS];

cublasStatus_t _ferry_cublasCreate(cublasHandle_t* handle)
{
    ferry::State state;
    ferry::save_state(&state);
    
    cublasCreate(handle);
    const int ngpus = ferry::num_gpus();
    for (int gpu = 0; gpu < ngpus; gpu++)
    {
        ferry::set_gpu(gpu);
        cublasCreate(&CublasHandle[gpu]);
    }
    ferry::restore_state(&state);
}
cublasStatus_t _ferry_cublasDlaswp(cublasHandle_t handle, int m, int n, double* A, int lda, int k1, int k2, int* ipiv, int incx)
{
    ferry::State state;
    ferry::save_state(&state);
    ferry::ArrayView2D<double> Aview{A, m, 1, n, lda};
    
    const int ngpus = ferry::num_gpus();
    std::vector<ferry::GPURange> n_ranges = ferry::make_ranges(0, n, ngpus);
    
    ferry::Region2D ARegions[ngpus];
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ARegions[gpu].makeUnion(0, m, n_range.begin, n_range.end);
    }
    
    ferry::Array2D<double> ABuffers[ngpus];
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        ABuffers[gpu].allocate(ARegions[gpu]);
        ferry::copy(ABuffers[gpu], Aview.subview(ARegions[gpu]));
    }
    for (ferry::GPURange n_range : n_ranges)
    {
        const int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        cublasDlaswp(CublasHandle[gpu], m, n_range.size, ABuffers[gpu].at(0-ARegions[gpu].xmin, n_range.begin-ARegions[gpu].ymin), ABuffers[gpu].colstride, k1, k2, ipiv, incx);
    }
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        ferry::copy(Aview.subview(ARegions[gpu]),ABuffers[gpu]);
    }
    ferry::restore_state(&state);
}
cublasStatus_t _ferry_cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, double const* alpha, double const* A, int lda, double* B, int ldb)
{
    ferry::State state;
    ferry::save_state(&state);
    ferry::ArrayView2D<double const> Aview{A, m, 1, m, lda};
    ferry::ArrayView2D<double> Bview{B, m, 1, n, ldb};
    
    const int ngpus = ferry::num_gpus();
    std::vector<ferry::GPURange> n_ranges = ferry::make_ranges(0, n, ngpus);
    
    ferry::Region2D ARegions[ngpus];
    ferry::Region2D BRegions[ngpus];
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ARegions[gpu].makeUnion(0, m, 0, m);
        BRegions[gpu].makeUnion(0, m, n_range.begin, n_range.end);
    }
    
    ferry::Array2D<double> ABuffers[ngpus];
    ferry::Array2D<double> BBuffers[ngpus];
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        ABuffers[gpu].allocate(ARegions[gpu]);
        ferry::copy(ABuffers[gpu], Aview.subview(ARegions[gpu]));
        BBuffers[gpu].allocate(BRegions[gpu]);
        ferry::copy(BBuffers[gpu], Bview.subview(BRegions[gpu]));
    }
    for (ferry::GPURange n_range : n_ranges)
    {
        const int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        cublasDtrsm(CublasHandle[gpu], side, uplo, trans, diag, m, n_range.size, alpha, ABuffers[gpu].at(0-ARegions[gpu].xmin, 0-ARegions[gpu].ymin), ABuffers[gpu].colstride, BBuffers[gpu].at(0-BRegions[gpu].xmin, n_range.begin-BRegions[gpu].ymin), BBuffers[gpu].colstride);
    }
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        ferry::copy(Bview.subview(BRegions[gpu]),BBuffers[gpu]);
    }
    ferry::restore_state(&state);
}
cublasStatus_t _ferry_cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, double const* alpha, double const* A, int lda, double const* B, int ldb, double const* beta, double* C, int ldc)
{
    ferry::State state;
    ferry::save_state(&state);
    ferry::ArrayView2D<double const> Aview{A, m, 1, k, lda};
    ferry::ArrayView2D<double const> Bview{B, k, 1, n, ldb};
    ferry::ArrayView2D<double> Cview{C, m, 1, n, ldc};
    
    const int ngpus = ferry::num_gpus();
    std::vector<ferry::GPURange> n_ranges = ferry::make_ranges(0, n, ngpus);
    
    ferry::Region2D ARegions[ngpus];
    ferry::Region2D BRegions[ngpus];
    ferry::Region2D CRegions[ngpus];
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ARegions[gpu].makeUnion(0, m, 0, k);
        BRegions[gpu].makeUnion(0, k, n_range.begin, n_range.end);
        CRegions[gpu].makeUnion(0, m, n_range.begin, n_range.end);
    }
    
    ferry::Array2D<double> ABuffers[ngpus];
    ferry::Array2D<double> BBuffers[ngpus];
    ferry::Array2D<double> CBuffers[ngpus];
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        ABuffers[gpu].allocate(ARegions[gpu]);
        ferry::copy(ABuffers[gpu], Aview.subview(ARegions[gpu]));
        BBuffers[gpu].allocate(BRegions[gpu]);
        ferry::copy(BBuffers[gpu], Bview.subview(BRegions[gpu]));
        CBuffers[gpu].allocate(CRegions[gpu]);
        ferry::copy(CBuffers[gpu], Cview.subview(CRegions[gpu]));
    }
    for (ferry::GPURange n_range : n_ranges)
    {
        const int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        cublasDgemm(CublasHandle[gpu], transa, transb, m, n_range.size, k, alpha, ABuffers[gpu].at(0-ARegions[gpu].xmin, 0-ARegions[gpu].ymin), ABuffers[gpu].colstride, BBuffers[gpu].at(0-BRegions[gpu].xmin, n_range.begin-BRegions[gpu].ymin), BBuffers[gpu].colstride, beta, CBuffers[gpu].at(0-CRegions[gpu].xmin, n_range.begin-CRegions[gpu].ymin), CBuffers[gpu].colstride);
    }
    for (ferry::GPURange n_range : n_ranges)
    {
        int gpu = n_range.gpu;
        ferry::set_gpu(gpu);
        ferry::copy(Cview.subview(CRegions[gpu]),CBuffers[gpu]);
    }
    ferry::restore_state(&state);
}
#undef cublasCreate
#define cublasCreate _ferry_cublasCreate
#undef cublasDlaswp
#define cublasDlaswp _ferry_cublasDlaswp
#undef cublasDtrsm
#define cublasDtrsm _ferry_cublasDtrsm
#undef cublasDgemm
#define cublasDgemm _ferry_cublasDgemm
