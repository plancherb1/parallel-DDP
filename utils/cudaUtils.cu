/*****************************************************************
 * Utils for Cuda code
 * 1 CUDA ERROR CHECKING CODE
 *
 * Adapted from:
 * https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 * https://stackoverflow.com/questions/22399794/qr-decomposition-to-solve-linear-systems-in-cuda
 *
 * error checking usage for library functions:
 *    gpuErrchk(cudaMalloc((void**)&a_d, size*sizeof(int)));
 *    cusolveErrchk(<cusolvefunctioncall>)
 *    cublasErrchk(<cubalsfunctioncall>)
 * error checking usage for custom kernels:
 *    kernel<<<1,1>>>(a);
 *    gpuErrchk(cudaPeekAtLastError());
 *    gpuErrchk(cudaDeviceSynchronize());
 *
 * 2 Matrix printing code
 *
 * 3 Templated External Memory Wrapper
 *****************************************************************/
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

/*** 1 CUDA ERROR CHECKING CODE 1 ***/
    __host__ 
    void gpuAssert(cudaError_t code, const char *file, const int line, bool abort=true){
        if (code != cudaSuccess){
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort){cudaDeviceReset(); exit(code);}
        }
    }

    __host__ __device__ 
    static const char *cublasGetErrorEnum(cublasStatus_t error){
        switch (error){
            case CUBLAS_STATUS_SUCCESS:             return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:     return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:        return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:       return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:       return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:       return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:    return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:      return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:       return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:       return "CUBLAS_STATUS_LICENSE_ERROR";
            default:                                return "<unknown>";
        }
    }

    __host__ __device__ 
    void cublasAssert(cublasStatus_t err, const char *file, const int line){
        if(CUBLAS_STATUS_SUCCESS != err) {
            #ifdef  __CUDA_ARCH__
                printf("CUBLAS error in file '%s', line %d\n error %d: %s\n terminating!\n",file,line,err,cublasGetErrorEnum(err));
            #else
                fprintf(stderr, "CUBLAS error in file '%s', line %d\n error %d: %s\n terminating!\n",file,line,err,cublasGetErrorEnum(err));
                cudaDeviceReset();
            #endif
            assert(0);
        }
    }

    __host__ 
    static const char *cusolverGetErrorEnum(cusolverStatus_t error){
        switch (error){
            case CUSOLVER_STATUS_SUCCESS:                   return "CUSOLVER_SUCCESS";
            case CUSOLVER_STATUS_NOT_INITIALIZED:           return "CUSOLVER_STATUS_NOT_INITIALIZED";
            case CUSOLVER_STATUS_ALLOC_FAILED:              return "CUSOLVER_STATUS_ALLOC_FAILED";
            case CUSOLVER_STATUS_INVALID_VALUE:             return "CUSOLVER_STATUS_INVALID_VALUE";
            case CUSOLVER_STATUS_ARCH_MISMATCH:             return "CUSOLVER_STATUS_ARCH_MISMATCH";
            case CUSOLVER_STATUS_EXECUTION_FAILED:          return "CUSOLVER_STATUS_EXECUTION_FAILED";
            case CUSOLVER_STATUS_INTERNAL_ERROR:            return "CUSOLVER_STATUS_INTERNAL_ERROR";
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            default:                                        return "<unknown>";
        }
    }

    __host__ void cusolverAssert(cusolverStatus_t err, const char *file, const int line){
        if(CUSOLVER_STATUS_SUCCESS != err) {
            fprintf(stderr, "CUSOLVE error in file '%s', line %d\n error %d: %s\n terminating!\n",file,line,err,cusolverGetErrorEnum(err));
            cudaDeviceReset(); assert(0);
        }
    }

    __host__ 
    static const char *cusparseGetErrorEnum(cusparseStatus_t error){
        switch (error){
            case CUSPARSE_STATUS_SUCCESS:                   return "CUSPARSE_SUCCESS";
            case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
            case CUSPARSE_STATUS_ALLOC_FAILED:              return "CUSPARSE_STATUS_ALLOC_FAILED";
            case CUSPARSE_STATUS_INVALID_VALUE:             return "CUSPARSE_STATUS_INVALID_VALUE";
            case CUSPARSE_STATUS_ARCH_MISMATCH:             return "CUSPARSE_STATUS_ARCH_MISMATCH";
            case CUSPARSE_STATUS_EXECUTION_FAILED:          return "CUSPARSE_STATUS_EXECUTION_FAILED";
            case CUSPARSE_STATUS_INTERNAL_ERROR:            return "CUSPARSE_STATUS_INTERNAL_ERROR";
            case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            default:                                        return "<unknown>";
        }
    }

    __host__ void cusparseAssert(cusparseStatus_t err, const char *file, const int line){
        if(CUSPARSE_STATUS_SUCCESS != err) {
            fprintf(stderr, "CUSPARSE error in file '%s', line %d\n error %d: %s\n terminating!\n",file,line,err,cusparseGetErrorEnum(err));
            cudaDeviceReset(); assert(0);
        }
    }
  
/*** 1 CUDA ERROR CHECKING CODE 1 ***/