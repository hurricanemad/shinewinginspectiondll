#include "pch.h"
#include "book.h"

texture<int, 2> texDiffImage;
__device__ float fA = 1062.4f;
__device__ float fB = -0.2723f;
__device__ float fC = 10.0f;
__device__ float fR = 260.0f;
const float fPi = 3.1415926535897932384626433f;

__constant__ float cfGaussParam[3];
__constant__ float cfLaplaceParamX[5];
__constant__ float cfLaplaceParamY[5];
__constant__ float cfCEGaussParam[nCEKernelSz];
__constant__ float cfGaussParam55[25];
__constant__ float cfGaussParam33[9];
__constant__ float fMeanIHb[1];
__constant__ float fMeanY[1];
__constant__ int nMeanIHb[1];
__constant__ int nMeanY[1];
__constant__ float cfREpsilon[1];
__constant__ float cfGEpsilon[1];
__constant__ float cfBEpsilon[1];

__device__ const int nKernelSz5 = 5;
__device__ const int nKernelRSz5 = nKernelSz5 / 2;

__device__ const int nKernelSz3 = 3;
__device__ const int nKernelRSz3 = nKernelSz3 / 2;

int nBackupS = 100;

__global__ void AddImage(float* pfImagePtr1, float* pfImagePtr2, float* pfImage3, float fRatio) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	pfImage3[nOffset] = pfImagePtr1[nOffset] + fRatio * pfImagePtr2[nOffset];
}

__global__ void AddImageThresh(float* pBKImagePtr, float* pfDetailImagePtr, float* pfEnhanceImagePtr, float* pfResultImagePtr, float fThresh,float fRatio) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	if(fabs(pfEnhanceImagePtr[nOffset]) > fThresh)
		pfResultImagePtr[nOffset] = pBKImagePtr[nOffset] + fRatio * pfEnhanceImagePtr[nOffset];
	else
		pfResultImagePtr[nOffset] = pBKImagePtr[nOffset] + pfDetailImagePtr[nOffset];
}

__global__ void AddImageThresh(float* pBKImagePtr, float* pfDetailImagePtr, float* pfEnhanceImagePtr, float* pfResultImagePtr, float fThresh) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	if (fabs(pfEnhanceImagePtr[nOffset]) > fThresh)
		pfResultImagePtr[nOffset] = pBKImagePtr[nOffset] + pfEnhanceImagePtr[nOffset];
	else
		pfResultImagePtr[nOffset] = pBKImagePtr[nOffset] + pfDetailImagePtr[nOffset];
}

__global__ void DiffImage(unsigned char* pucImagePtr1, unsigned char* pucImagePtr2, int* pnImage3) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;

	int nOffset = nY * (blockDim.x * gridDim.x) + nX;

	pnImage3[nOffset] = pucImagePtr1[nOffset] - pucImagePtr2[nOffset];

}

__global__ void DiffImage(float* pfImagePtr1, float* pfImagePtr2, float* pfImage3) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;

	int nOffset = nY * (blockDim.x * gridDim.x) + nX;

	pfImage3[nOffset] = pfImagePtr1[nOffset] - pfImagePtr2[nOffset];
}


__global__ void MeanImage(float* pfFilterPtr, int nImageW, int nImageH, int nKernelSzW, int nKernelSzH) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;

	int nOffset = nY * (blockDim.x * gridDim.x) + nX;

	int r, c;

	int nTempV;
	int nHalfKW = nKernelSzW / 2, nHalfKH = nKernelSzH / 2;
	int nR, nC;
	pfFilterPtr[nOffset] = 0.0f;
	for (r = 0; r < nKernelSzH; r++) {
		for (c = 0; c < nKernelSzW; c++) {
			nR = r + nY - nHalfKH;
			nC = c + nX - nHalfKW;
			if (nR >= 0 && nR < nImageH && nC >= 0 && nC < nImageW) {
				nTempV = tex2D(texDiffImage, nC, nR);
				pfFilterPtr[nOffset] += nTempV;
			}

		}
	}
	pfFilterPtr[nOffset] = fabs(pfFilterPtr[nOffset] / (nKernelSzW * nKernelSzH));
}

//__global__ void MeanImage(float* pfFilterPtr, float* pfSrcImagePtr, int nImageW, int nImageH, int nKernelSzW, int nKernelSzH) {
//	int nX = threadIdx.x + blockIdx.x * blockDim.x;
//	int nY = threadIdx.y + blockIdx.y * blockDim.y;
//
//	int nOffset = nY * (blockDim.x * gridDim.x) + nX;
//
//	int r, c;
//
//	int nTempV;
//	int nHalfKW = nKernelSzW / 2, nHalfKH = nKernelSzH / 2;
//	int nR, nC;
//	pfFilterPtr[nOffset] = 0.0f;
//	for (r = 0; r < nKernelSzH; r++) {
//		for (c = 0; c < nKernelSzW; c++) {
//			nR = r + nY - nHalfKH;
//			nC = c + nX - nHalfKW;
//			if (nR >= 0 && nR < nImageH && nC >= 0 && nC < nImageW) {
//				nTempV = tex2D(texDiffImage, nC, nR);
//				pfFilterPtr[nOffset] += nTempV;
//			}
//
//		}
//	}
//	pfFilterPtr[nOffset] = fabs(pfFilterPtr[nOffset] / (nKernelSzW * nKernelSzH));
//}

__global__ void MeanImage(float* pfSrcImagePtr, float* pfFilterPtr, int nImageW, int nImageH) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	__shared__ float fTempImageBuffer[(nBlockDimX + nMultiple * nMeanKernelRadius) * (nBlockDimY + nMultiple * nMeanKernelRadius)];

	int nBufferWidth = blockDim.x + __mul24(nMultiple, nMeanKernelRadius);
	if (nX < nMeanKernelRadius || nX > nBlockDimX - 1 - nMeanKernelRadius ||
		nY < nMeanKernelRadius || nY > nBlockDimY - 1 - nMeanKernelRadius) {
		//upper left
		if (nY - nMeanKernelRadius < 0 || nX - nMeanKernelRadius < 0) {
			fTempImageBuffer[threadIdx.x + __mul24(threadIdx.y, nBufferWidth)] = pfSrcImagePtr[nOffset];
		}
		else {
			fTempImageBuffer[threadIdx.x + __mul24(threadIdx.y, nBufferWidth)] = pfSrcImagePtr[nOffset - nMeanKernelRadius - __mul24(nMeanKernelRadius, nImageW)];
		}
		//upper right
		if (nY - nMeanKernelRadius < 0 || nX + nMeanKernelRadius > nImageW - 1) {
			fTempImageBuffer[threadIdx.x + __mul24(2, nMeanKernelRadius) + __mul24(threadIdx.y, nBufferWidth)] = pfSrcImagePtr[nOffset];
		}
		else {
			fTempImageBuffer[threadIdx.x + __mul24(2, nMeanKernelRadius) + __mul24(threadIdx.y, nBufferWidth)] = pfSrcImagePtr[nOffset + nMeanKernelRadius - __mul24(nMeanKernelRadius, nImageW)];
		}
		//lower left
		if (nY + nMeanKernelRadius > nImageH - 1 || nX - nMeanKernelRadius < 0) {
			fTempImageBuffer[threadIdx.x + __mul24(threadIdx.y + __mul24(2, nMeanKernelRadius), nBufferWidth)] = pfSrcImagePtr[nOffset];
		}
		else {
			fTempImageBuffer[threadIdx.x + __mul24(threadIdx.y + __mul24(2, nMeanKernelRadius), nBufferWidth)] = pfSrcImagePtr[nOffset - nMeanKernelRadius + __mul24(nMeanKernelRadius, nImageW)];
		}
		//lower right
		if (nY + nMeanKernelRadius > nImageH - 1 || nX + nMeanKernelRadius > nImageW - 1) {
			fTempImageBuffer[threadIdx.x + __mul24(2, nMeanKernelRadius) + __mul24(threadIdx.y + __mul24(2, nMeanKernelRadius), nBufferWidth)] = pfSrcImagePtr[nOffset];
		}
		else {
			fTempImageBuffer[threadIdx.x + __mul24(2, nMeanKernelRadius) + __mul24(threadIdx.y + __mul24(2, nMeanKernelRadius), nBufferWidth)] = pfSrcImagePtr[nOffset + nMeanKernelRadius + __mul24(nMeanKernelRadius, nImageW)];
		}

		fTempImageBuffer[threadIdx.x + nMeanKernelRadius + __mul24(threadIdx.y + nMeanKernelRadius, nBufferWidth)] = pfSrcImagePtr[nOffset];
	}
	else {
		fTempImageBuffer[threadIdx.x + nMeanKernelRadius + __mul24(threadIdx.y + nMeanKernelRadius, nBufferWidth)] = pfSrcImagePtr[nOffset];
	}

	__syncthreads();

	int n, r, c;

	pfFilterPtr[nOffset] = 0.0f;

	for (n = 0; n < nMeanKernelSz * nMeanKernelSz; n++) {
		r = n / nMeanKernelSz;
		c = n % nMeanKernelSz;

		pfFilterPtr[nOffset] += fTempImageBuffer[threadIdx.x + c + (threadIdx.y + r) * nBufferWidth];
	}

	pfFilterPtr[nOffset] = fabs(pfFilterPtr[nOffset] / (nMeanKernelSz * nMeanKernelSz));
}

__global__ void FilterImage(unsigned char* pucSrcImagePtr, int* pnDiffImagePtr, float* pfFilterImagePtr, unsigned char* pucResultImgPtr, float* pfResultImagePtr, int nImageW, int nImageH, float fRRR) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;

	int nOffset = nY * (blockDim.x * gridDim.x) + nX;

	float fTA = fA * fRRR;
	float fRatio;

	float fV;
	if (pfFilterImagePtr[nOffset] > 128.0) {
		fV = 128.0f;
	}
	else {
		fV = pfFilterImagePtr[nOffset];
	}

	if (fV < 1e-6) {
		fRatio = (exp(pow(2.0 / fTA, fB)) + (2.0) * fRRR * 5.0) / fR;
	}
	else if (fV <= 2.0f) {
		fRatio = (exp(pow(2.0 / fTA, fB)) + (2.0 - fV) * fRRR * 5.0) / fR;
	}
	else {
		fRatio = exp(pow(fV / fTA, fB)) / fR;
	}

	pucResultImgPtr[nOffset] = (unsigned char)(pucSrcImagePtr[nOffset] - fRatio * pnDiffImagePtr[nOffset] + 0.5f);
	pfResultImagePtr[nOffset] = pucSrcImagePtr[nOffset] - fRatio * pnDiffImagePtr[nOffset] + 0.5f;
}

__global__ void FilterImage(float* pfSrcImagePtr, float* pfDiffImagePtr, float* pfFilterImagePtr, float* pucResultImgPtr, float* pfResultImagePtr, int nImageW, int nImageH, float fRRR) {
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
	int nY = threadIdx.y + blockIdx.y * blockDim.y;

	int nOffset = nY * (blockDim.x * gridDim.x) + nX;

	float fTA = fA * fRRR;
	float fRatio;

	float fV;
	if (pfFilterImagePtr[nOffset] > 128.0) {
		fV = 128.0f;
	}
	else {
		fV = pfFilterImagePtr[nOffset];
	}

	if (fV < 1e-6) {
		fRatio = (exp(pow(2.0 / fTA, fB)) + (2.0) * fRRR * 5.0) / fR;
	}
	else if (fV <= 2.0f) {
		fRatio = (exp(pow(2.0 / fTA, fB)) + (2.0 - fV) * fRRR * 5.0) / fR;
	}
	else {
		fRatio = exp(pow(fV / fTA, fB)) / fR;
	}

	pucResultImgPtr[nOffset] = pfSrcImagePtr[nOffset] - fRatio * pfDiffImagePtr[nOffset];
	pfResultImagePtr[nOffset] = pfSrcImagePtr[nOffset] - fRatio * pfDiffImagePtr[nOffset];
}

void MANRFilter(unsigned char* pucSrcImagePtr, unsigned char* pucResultImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, double dRatio) {
	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageW / 16, nImageH / 16);

	//ofstream fout(".\Log.txt", ios_base::out|ios_base::ate);

	unsigned char* pucDeviceSrcImgPtr;
	unsigned char* pucDeviceResultImgPtr;
	int* pnDeviceDiffImgPtr;
	float* pfDeviceFilterDiffImgPtr;
	unsigned char* pucDeviceResultImagePtr;
	float* pfDeviceResultImagePtr;

	//fout << __LINE__ << endl;

	HANDLE_ERROR(cudaMalloc((void**)(&pucDeviceSrcImgPtr), sizeof(unsigned char) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pucDeviceResultImgPtr), sizeof(unsigned char) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pnDeviceDiffImgPtr), sizeof(int) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pfDeviceFilterDiffImgPtr), sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pucDeviceResultImagePtr), sizeof(unsigned char) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pfDeviceResultImagePtr), sizeof(float) * nImageW * nImageH));

	//fout << __LINE__ << endl;

	HANDLE_ERROR(cudaMemcpy(pucDeviceSrcImgPtr, pucSrcImagePtr, sizeof(unsigned char) * nImageW * nImageH, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pucDeviceResultImgPtr, pucResultImagePtr, sizeof(unsigned char) * nImageW * nImageH, cudaMemcpyHostToDevice));
	//fout << __LINE__ << endl;

	DiffImage << <d3Block, d3Thread >> > (pucDeviceSrcImgPtr, pucDeviceResultImgPtr, pnDeviceDiffImgPtr);
	cudaChannelFormatDesc cfdTemp = cudaCreateChannelDesc<int>();
	//fout << __LINE__ << endl;
	HANDLE_ERROR(cudaBindTexture2D(NULL, &texDiffImage, pnDeviceDiffImgPtr, &cfdTemp, nImageW, nImageH, sizeof(int) * nImageW));

	//fout << __LINE__ << endl;
	MeanImage << <d3Block, d3Thread >> > (pfDeviceFilterDiffImgPtr, nImageW, nImageH, 5, 5);
	FilterImage << < d3Block, d3Thread >> > (pucDeviceSrcImgPtr, pnDeviceDiffImgPtr, pfDeviceFilterDiffImgPtr, pucDeviceResultImagePtr, pfDeviceResultImagePtr, nImageW, nImageH, 1.0f);

	//fout << __LINE__ << endl;
	//unsigned char* pucTempResultImagePtr;
	//pucTempResultImagePtr = (unsigned char*)(malloc(sizeof(unsigned char) * nImageW * nImageH));
	HANDLE_ERROR(cudaMemcpy(pucResultImagePtr, pucDeviceResultImagePtr, sizeof(unsigned char) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfResultImagePtr, pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	//fout << __LINE__ << endl;

	//free(pucTempResultImagePtr);
	HANDLE_ERROR(cudaUnbindTexture(&texDiffImage));
	HANDLE_ERROR(cudaFree(pfDeviceResultImagePtr));
	HANDLE_ERROR(cudaFree(pucDeviceSrcImgPtr));
	HANDLE_ERROR(cudaFree(pucDeviceResultImgPtr));
	HANDLE_ERROR(cudaFree(pnDeviceDiffImgPtr));
	HANDLE_ERROR(cudaFree(pfDeviceFilterDiffImgPtr));
	HANDLE_ERROR(cudaFree(pucDeviceResultImagePtr));
	//fout << __LINE__ << endl;
}

void MANRFilter(float* pfSrcImagePtr, float* pfResultImagePtr, float* pfDstImagePtr, int nImageW, int nImageH, double dRatio) {
	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageW / 16, nImageH / 16);

	//ofstream fout(".\Log.txt", ios_base::out|ios_base::ate);

	float* pfDeviceSrcImgPtr;
	float* pfDeviceResultImgPtr;
	float* pfDeviceDiffImgPtr;
	float* pfDeviceFilterDiffImgPtr;
	float* pfDeviceResultImagePtrTemp;
	float* pfDeviceResultImagePtr;

	//fout << __LINE__ << endl;

	HANDLE_ERROR(cudaMalloc((void**)(&pfDeviceSrcImgPtr), sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pfDeviceResultImgPtr), sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pfDeviceDiffImgPtr), sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pfDeviceFilterDiffImgPtr), sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pfDeviceResultImagePtrTemp), sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc((void**)(&pfDeviceResultImagePtr), sizeof(float) * nImageW * nImageH));

	//fout << __LINE__ << endl;

	HANDLE_ERROR(cudaMemcpy(pfDeviceSrcImgPtr, pfSrcImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceResultImgPtr, pfResultImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));
	//fout << __LINE__ << endl;

	DiffImage << <d3Block, d3Thread >> > (pfDeviceSrcImgPtr, pfDeviceResultImgPtr, pfDeviceDiffImgPtr);
	//cudaChannelFormatDesc cfdTemp = cudaCreateChannelDesc<int>();
	//fout << __LINE__ << endl;
	//HANDLE_ERROR(cudaBindTexture2D(NULL, &texDiffImage, pnDeviceDiffImgPtr, &cfdTemp, nImageW, nImageH, sizeof(int) * nImageW));

	//fout << __LINE__ << endl;
	MeanImage << <d3Block, d3Thread >> > (pfDeviceDiffImgPtr, pfDeviceFilterDiffImgPtr, nImageW, nImageH);
	FilterImage << < d3Block, d3Thread >> > (pfDeviceSrcImgPtr, pfDeviceDiffImgPtr, pfDeviceFilterDiffImgPtr, pfDeviceResultImagePtrTemp, pfDeviceResultImagePtr, nImageW, nImageH, 1.0f);

	//fout << __LINE__ << endl;
	//unsigned char* pucTempResultImagePtr;
	//pucTempResultImagePtr = (unsigned char*)(malloc(sizeof(unsigned char) * nImageW * nImageH));
	HANDLE_ERROR(cudaMemcpy(pfDstImagePtr, pfDeviceResultImagePtrTemp, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfResultImagePtr, pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	//fout << __LINE__ << endl;

	//free(pucTempResultImagePtr);
	//HANDLE_ERROR(cudaUnbindTexture(&texDiffImage));
	HANDLE_ERROR(cudaFree(pfDeviceResultImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceSrcImgPtr));
	HANDLE_ERROR(cudaFree(pfDeviceResultImgPtr));
	HANDLE_ERROR(cudaFree(pfDeviceDiffImgPtr));
	HANDLE_ERROR(cudaFree(pfDeviceFilterDiffImgPtr));
	HANDLE_ERROR(cudaFree(pfDeviceResultImagePtrTemp));
	//fout << __LINE__ << endl;
}

void GenGauss1D(float* pfKernel, int nKernelW, float fSigma) {
	int n;

	int nHalfKernelW = nKernelW / 2;
	float fSum = 0.0f;
	for (n = 0; n < nKernelW; n++) {
		pfKernel[n] = (1.0f / (sqrt(2 * fPi) * fSigma) * exp(-((n - nHalfKernelW) * (n - nHalfKernelW)) / (2 * fSigma * fSigma)));
		fSum += pfKernel[n];
	}

	for (n = 0; n < nKernelW; n++) {
		pfKernel[n] /= fSum;
	}


}

void GenGauss1DLaplacian(float* pfKernel, int nKernelW, float fSigma) {
	int n;

	//int nHalfKernelW = nKernelW / 2;
	//float fSum = 0.0f;
	//for (n = 0; n < nKernelW; n++) {
	//	pfKernel[n] = (1.0f / (sqrt(2 * fPi) * fSigma) * exp(-((n - nHalfKernelW) * (n - nHalfKernelW)) / (2 * fSigma * fSigma)));
	//	fSum += pfKernel[n];
	//}

	//for (n = 0; n < nKernelW; n++) {
	//	pfKernel[n] /= fSum;
	//}

	for (n = 0; n < nKernelW; n++) {
		pfKernel[n] = 0.33333333333f;

	}
}

void GenLaplaceKernel1D5Sz(float* pfKernelX, float* pfKernelY) {

	pfKernelX[0] = 1.0f;
	pfKernelX[1] = 0.0f;
	pfKernelX[2] = -2.0f;
	pfKernelX[3] = 0.0f;
	pfKernelX[4] = 1.0f;

	pfKernelY[0] = 1.0f;
	pfKernelY[1] = 4.0f;
	pfKernelY[2] = 6.0f;
	pfKernelY[3] = 4.0f;
	pfKernelY[4] = 1.0f;
}


__global__ void ConvolutionRow(float* pfImagePtr, float* pfConvoImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nShift = __mul24(threadIdx.y, (nBlockDimX + nKernelRSz * 2));
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[nBlockDimY * (nBlockDimX + 2 * nKernelRSz)];

	if (threadIdx.x < nKernelRSz || threadIdx.x >= nBlockDimX - nKernelRSz) {
		if ((nX - nKernelRSz) < 0) {
			fConvData[threadIdx.x + nShift] = 0.0f;
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - nKernelRSz];
		}

		if ((nX + nKernelRSz) > nImageW - 1) {
			fConvData[threadIdx.x + 2 * nKernelRSz + nShift] = 0.0f;
		}
		else {
			fConvData[threadIdx.x + 2 * nKernelRSz + nShift] = pfImagePtr[nOffset + nKernelRSz];
		}
		fConvData[threadIdx.x + nKernelRSz + nShift] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + nKernelRSz + nShift] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;
	int nConX = nKernelRSz + threadIdx.x;

	for (int i = -nKernelRSz; i <= nKernelRSz; i++) {
		fSum += fConvData[nConX + i + nShift] * cfGaussParam[nKernelRSz + i];
	}

	pfConvoImagePtr[nOffset] = fSum;
}

__global__ void ConvolutionCol(float* pfImagePtr, float* pfConvoImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nShift = __mul24(threadIdx.y, nBlockDimX);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[nBlockDimX * (nBlockDimY + 2 * nKernelRSz)];

	if (threadIdx.y < nKernelRSz || threadIdx.y >= nBlockDimY - nKernelRSz) {
		if ((nY - nKernelRSz) < 0) {
			fConvData[threadIdx.x + nShift] = 0.0f;
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - __mul24(nImageW, nKernelRSz)];
		}

		const int nShift1 = nShift + __mul24(2 * nKernelRSz, nBlockDimX);
		const int nShift2 = nShift + __mul24(nKernelRSz, nBlockDimX);
		if ((nY + nKernelRSz) > nImageH - 1) {
			fConvData[threadIdx.x + nShift1] = 0.0f;
		}
		else {
			fConvData[threadIdx.x + nShift1] = pfImagePtr[nOffset + __mul24(nImageW, nKernelRSz)];
		}
		fConvData[threadIdx.x + nShift2] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + nBlockDimX + nShift] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;
	//int nConX = nKernelRSz + threadIdx.x;

	for (int i = 0; i <= nKernelRSz * 2; i++) {
		fSum += fConvData[threadIdx.x + (threadIdx.y + i) * nBlockDimX] * cfGaussParam[i];
	}

	pfConvoImagePtr[nOffset] = fSum;
}

__global__ void Convolution2D3(float* pfImagePtr, float* pfConvoImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nBlockWidth = nBlockDimX + __mul24(nKernelRSz3, 2);
	const int nBlockHeight = nBlockDimY + __mul24(nKernelRSz3, 2);
	const int nShift = __mul24(threadIdx.y, nBlockWidth);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[(nBlockDimY + 2 * nKernelRSz3) * (nBlockDimX + 2 * nKernelRSz3)];

	if (threadIdx.x < nKernelRSz3 || threadIdx.x >= nBlockDimX - nKernelRSz3 ||
		threadIdx.y < nKernelRSz3 || threadIdx.y >= nBlockDimY - nKernelRSz3) {
		if ((nX - nKernelRSz3) < 0 || (nY - nKernelRSz3) < 0) {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - __mul24(nKernelRSz3, nImageW) - nKernelRSz3];
		}

		if ((nX + nKernelRSz3) > nImageW - 1 || (nY - nKernelRSz3) < 0) {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz3) + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz3) + nShift] = pfImagePtr[nOffset - __mul24(nKernelRSz3, nImageW) + nKernelRSz3];
		}

		if ((nX - nKernelRSz3) < 0 || (nY + nKernelRSz3) > nImageH - 1) {
			fConvData[threadIdx.x + nShift + __mul24(__mul24(2, nKernelRSz3), nBlockWidth)] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift + __mul24(__mul24(2, nKernelRSz3), nBlockWidth)] = pfImagePtr[nOffset + __mul24(nKernelRSz3, nImageW) - nKernelRSz3];
		}

		if ((nX + nKernelRSz3) > nImageW - 1 || (nY + nKernelRSz3) > nImageH - 1) {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz3) + nShift + __mul24(__mul24(2, nKernelRSz3), nBlockWidth)] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz3) + nShift + __mul24(__mul24(2, nKernelRSz3), nBlockWidth)] = pfImagePtr[nOffset + __mul24(nKernelRSz3, nImageW) + nKernelRSz3];
		}
		fConvData[threadIdx.x + nKernelRSz3 + nShift + __mul24(nKernelRSz3, nBlockWidth)] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + nKernelRSz3 + nShift + __mul24(nKernelRSz3, nBlockWidth)] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;

	int nKernelSZ = nKernelSz3 * nKernelSz3;
	int r, c;

	for (int i = 0; i < nKernelSZ; i++) {
		r = i / nKernelSz3;
		c = i % nKernelSz3;

		fSum += fConvData[threadIdx.x + c + __mul24(threadIdx.y + r, nBlockWidth)] * cfGaussParam33[i];
	}

	pfConvoImagePtr[nOffset] = fSum;
}

__global__ void Convolution2D5(float* pfImagePtr, float* pfConvoImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nBlockWidth = nBlockDimX + __mul24(nKernelRSz5, 2);
	const int nBlockHeight = nBlockDimY + __mul24(nKernelRSz5, 2);
	const int nShift = __mul24(threadIdx.y, nBlockWidth);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[(nBlockDimY + 2 * nKernelRSz5) * (nBlockDimX + 2 * nKernelRSz5)];

	if (threadIdx.x < nKernelRSz5 || threadIdx.x >= nBlockDimX - nKernelRSz5 ||
		threadIdx.y < nKernelRSz5 || threadIdx.y >= nBlockDimY - nKernelRSz5) {
		if ((nX - nKernelRSz5) < 0 || (nY - nKernelRSz5) < 0) {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - __mul24(nKernelRSz5, nImageW) - nKernelRSz5];
		}

		if ((nX + nKernelRSz5) > nImageW - 1 || (nY - nKernelRSz5) < 0) {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift] = pfImagePtr[nOffset - __mul24(nKernelRSz5, nImageW) + nKernelRSz5];
		}

		if ((nX - nKernelRSz5) < 0 || (nY + nKernelRSz5) > nImageH - 1) {
			fConvData[threadIdx.x + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pfImagePtr[nOffset + __mul24(nKernelRSz5, nImageW) - nKernelRSz5];
		}

		if ((nX + nKernelRSz5) > nImageW - 1 || (nY + nKernelRSz5) > nImageH - 1) {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pfImagePtr[nOffset + __mul24(nKernelRSz5, nImageW) + nKernelRSz5];
		}
		fConvData[threadIdx.x + nKernelRSz5 + nShift + __mul24(nKernelRSz5, nBlockWidth)] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + nKernelRSz5 + nShift + __mul24(nKernelRSz5, nBlockWidth)] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;

	int nKernelSZ = nKernelSz5 * nKernelSz5;
	int r, c;

	for (int i = 0; i < nKernelSZ; i++) {
		r = i / nKernelSz5;
		c = i % nKernelSz5;

		fSum += fConvData[threadIdx.x + c + __mul24(threadIdx.y + r, nBlockWidth)] * cfGaussParam55[i];
	}

	pfConvoImagePtr[nOffset] = fSum;
}

__global__ void LaplacianFilterRow1(float* pfImagePtr, float* pfLFImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nShift = __mul24(threadIdx.y, (nBlockDimX + nLaplacianKRSz * 2));
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[nBlockDimY * (nBlockDimX + 2 * nLaplacianKRSz)];

	if (threadIdx.x < nLaplacianKRSz || threadIdx.x >= nBlockDimX - nLaplacianKRSz) {
		if ((nX - nLaplacianKRSz) < 0) {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - nLaplacianKRSz];
		}

		if ((nX + nLaplacianKRSz) > nImageW - 1) {
			fConvData[threadIdx.x + 2 * nLaplacianKRSz + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + 2 * nLaplacianKRSz + nShift] = pfImagePtr[nOffset + nLaplacianKRSz];
		}

		fConvData[threadIdx.x + nLaplacianKRSz + nShift] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + nLaplacianKRSz + nShift] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;
	int nConX = nLaplacianKRSz + threadIdx.x;

	for (int i = -nLaplacianKRSz; i <= nLaplacianKRSz; i++) {
		fSum += fConvData[nConX + i + nShift] * cfLaplaceParamX[nLaplacianKRSz + i];
	}

	pfLFImagePtr[nOffset] = fSum;
}

__global__ void LaplacianFilterRow2(float* pfImagePtr, float* pfLFImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nShift = __mul24(threadIdx.y, (nBlockDimX + nLaplacianKRSz * 2));
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[nBlockDimY * (nBlockDimX + 2 * nLaplacianKRSz)];

	if (threadIdx.x < nLaplacianKRSz || threadIdx.x >= nBlockDimX - nLaplacianKRSz) {
		if ((nX - nLaplacianKRSz) < 0) {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - nLaplacianKRSz];
		}

		if ((nX + nLaplacianKRSz) > nImageW - 1) {
			fConvData[threadIdx.x + 2 * nLaplacianKRSz + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + 2 * nLaplacianKRSz + nShift] = pfImagePtr[nOffset + nLaplacianKRSz];
		}

		fConvData[threadIdx.x + nLaplacianKRSz + nShift] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + nLaplacianKRSz + nShift] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;
	int nConX = nLaplacianKRSz + threadIdx.x;

	for (int i = -nLaplacianKRSz; i <= nLaplacianKRSz; i++) {
		fSum += fConvData[nConX + i + nShift] * cfLaplaceParamY[nLaplacianKRSz + i];
	}

	pfLFImagePtr[nOffset] = fSum;
}

__global__ void LaplacianFilterCol1(float* pfImagePtr, float* pfLFImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nShift = __mul24(threadIdx.y, nBlockDimX);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[nBlockDimX * (nBlockDimY + 2 * nLaplacianKRSz)];

	if (threadIdx.y < nLaplacianKRSz || threadIdx.y >= nBlockDimY - nLaplacianKRSz) {
		if ((nY - nLaplacianKRSz) < 0) {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - __mul24(nImageW, nLaplacianKRSz)];
		}

		const int nShift1 = nShift + __mul24(2 * nLaplacianKRSz, nBlockDimX);
		const int nShift2 = nShift + __mul24(nLaplacianKRSz, nBlockDimX);
		if ((nY + nLaplacianKSz) > nImageH - 1) {
			fConvData[threadIdx.x + nShift1] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift1] = pfImagePtr[nOffset + __mul24(nImageW, nLaplacianKRSz)];
		}
		fConvData[threadIdx.x + nShift2] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + __mul24(nLaplacianKRSz, nBlockDimX) + nShift] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;
	//int nConX = nKernelRSz + threadIdx.x;

	for (int i = 0; i <= nLaplacianKRSz * 2; i++) {
		fSum += fConvData[threadIdx.x + (threadIdx.y + i) * nBlockDimX] * cfLaplaceParamY[i];
	}

	pfLFImagePtr[nOffset] = fSum;
}

__global__ void LaplacianFilterCol2(float* pfImagePtr, float* pfLFImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nShift = __mul24(threadIdx.y, nBlockDimX);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[nBlockDimX * (nBlockDimY + 2 * nLaplacianKRSz)];

	if (threadIdx.y < nLaplacianKRSz || threadIdx.y >= nBlockDimY - nLaplacianKRSz) {
		if ((nY - nLaplacianKRSz) < 0) {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - __mul24(nImageW, nLaplacianKRSz)];
		}

		const int nShift1 = nShift + __mul24(2 * nLaplacianKRSz, nBlockDimX);
		const int nShift2 = nShift + __mul24(nLaplacianKRSz, nBlockDimX);
		if ((nY + nLaplacianKSz) > nImageH - 1) {
			fConvData[threadIdx.x + nShift1] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift1] = pfImagePtr[nOffset + __mul24(nImageW, nLaplacianKRSz)];
		}
		fConvData[threadIdx.x + nShift2] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + __mul24(nLaplacianKRSz, nBlockDimX) + nShift] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;
	//int nConX = nKernelRSz + threadIdx.x;

	for (int i = 0; i <= nLaplacianKRSz * 2; i++) {
		fSum += fConvData[threadIdx.x + (threadIdx.y + i) * nBlockDimX] * cfLaplaceParamX[i];
	}

	pfLFImagePtr[nOffset] = fSum;
}


void GenGauss2D(float* pfKernel, int nKernelW, float fSigma) {
	float fSum = 0.0;
	int r, c;
	for (r = 0; r < nKernelW; r++) {
		for (c = 0; c < nKernelW; c++) {
			pfKernel[nKernelW * r + c] = (1.0 / (2 * fPi * fSigma * fSigma)) * exp(-((r - nKernelW / 2) * (r - nKernelW / 2) + (c - nKernelW / 2) * (c - nKernelW / 2)) / (2 * fSigma * fSigma));
			fSum += pfKernel[nKernelW * r + c];
		}
	}

	//double dWeighDist = 0.0;

	for (r = 0; r < nKernelW; r++) {
		for (c = 0; c < nKernelW; c++) {
			//pfKernel[nKernelW * r + c] = static_cast<int>(pfKernel[nKernelW * r + c] / fSum * 512 + 0.5);
			pfKernel[nKernelW * r + c] = pfKernel[nKernelW * r + c] / fSum;
		}
	}
}

void GenLaplaceKernel2D3Sz(float* pfKernel) {

	pfKernel[0] = 0.0f;	pfKernel[1] = 1.0f;	pfKernel[2] = 0.0f;
	pfKernel[3] = 1.0f;	pfKernel[4] = -4.0f;	pfKernel[5] = 1.0f;
	pfKernel[6] = 0.0f;	pfKernel[7] = 1.0f;	pfKernel[8] = 0.0f;
}

__global__ void multiple(float* pfSrcImage, float fV, float* pfDstImage) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfDstImage[nOffset] = pfSrcImage[nOffset] * fV;
}

void EnhanceImage(float* fImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, int nKernelW, int nEnhanceLevel, bool bLG) {
	float* pfDeviceImagePtr, * pfConvolutionImagePtr, * pfDeviceResultImagePtr, * pfLaplaceResultImagePtr,
		* pfLaplaceImagePtrX1, * pfLaplaceImagePtrX2, * pfLaplaceImagePtrY1, * pfLaplaceImagePtrY2;

	//cudaEvent_t eventStart, eventEnd;
	//cudaEventCreate(&eventStart, 0);
	//cudaEventCreate(&eventEnd, 0);

	//cudaEventRecord(eventStart);
	HANDLE_ERROR(cudaMalloc(&pfDeviceImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfConvolutionImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfLaplaceImagePtrX1, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfLaplaceImagePtrX2, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfLaplaceImagePtrY1, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfLaplaceImagePtrY2, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfLaplaceResultImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH));

	HANDLE_ERROR(cudaMemcpy(pfDeviceImagePtr, fImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));

	float* pfKernel = (float*)(malloc(sizeof(float) * nKernelW));
	float* pfLaplacianKernelX = (float*)(malloc(sizeof(float) * 5));
	float* pfLaplacianKernelY = (float*)(malloc(sizeof(float) * 5));

	GenGauss1DLaplacian(pfKernel, nKernelW, 200.0);
	GenLaplaceKernel1D5Sz(pfLaplacianKernelX, pfLaplacianKernelY);

	cudaMemcpyToSymbol(cfGaussParam, pfKernel, sizeof(float) * nKernelW);
	cudaMemcpyToSymbol(cfLaplaceParamX, pfLaplacianKernelX, sizeof(float) * 5);
	cudaMemcpyToSymbol(cfLaplaceParamY, pfLaplacianKernelY, sizeof(float) * 5);


	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageW / 16, nImageH / 16);


	if (bLG) {
		ConvolutionRow << <d3Block, d3Thread >> > (pfDeviceImagePtr, pfConvolutionImagePtr, nImageW, nImageH);
		ConvolutionCol << <d3Block, d3Thread >> > (pfConvolutionImagePtr, pfDeviceResultImagePtr, nImageW, nImageH);
		LaplacianFilterRow1 << <d3Block, d3Thread >> > (pfDeviceResultImagePtr, pfLaplaceImagePtrX1, nImageW, nImageH);
		LaplacianFilterCol1 << <d3Block, d3Thread >> > (pfLaplaceImagePtrX1, pfLaplaceImagePtrX2, nImageW, nImageH);
		LaplacianFilterRow2 << <d3Block, d3Thread >> > (pfDeviceResultImagePtr, pfLaplaceImagePtrY1, nImageW, nImageH);
		LaplacianFilterCol2 << <d3Block, d3Thread >> > (pfLaplaceImagePtrY1, pfLaplaceImagePtrY2, nImageW, nImageH);
		AddImage << <d3Block, d3Thread >> > (pfLaplaceImagePtrX2, pfLaplaceImagePtrY2, pfLaplaceResultImagePtr, 1.0);
		AddImage << <d3Block, d3Thread >> > (pfDeviceImagePtr, pfLaplaceResultImagePtr, pfDeviceResultImagePtr, -0.1f * nEnhanceLevel);
	}
	else {
		LaplacianFilterRow1 << <d3Block, d3Thread >> > (pfDeviceImagePtr, pfLaplaceImagePtrX1, nImageW, nImageH);
		LaplacianFilterCol1 << <d3Block, d3Thread >> > (pfLaplaceImagePtrX1, pfLaplaceImagePtrX2, nImageW, nImageH);
		LaplacianFilterRow2 << <d3Block, d3Thread >> > (pfDeviceResultImagePtr, pfLaplaceImagePtrY1, nImageW, nImageH);
		LaplacianFilterCol2 << <d3Block, d3Thread >> > (pfLaplaceImagePtrY1, pfLaplaceImagePtrY2, nImageW, nImageH);
		AddImage << <d3Block, d3Thread >> > (pfLaplaceImagePtrX2, pfLaplaceImagePtrY2, pfLaplaceResultImagePtr, 1.0);
		AddImage << <d3Block, d3Thread >> > (pfDeviceImagePtr, pfLaplaceResultImagePtr, pfDeviceResultImagePtr, -0.1f * nEnhanceLevel);
	}


	HANDLE_ERROR(cudaMemcpy(pfResultImagePtr, pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	//cudaEventRecord(eventEnd);
	//cudaEventSynchronize(eventEnd);
	//float fETime;
	//cudaEventElapsedTime(&fETime, eventStart, eventEnd);

	//cout << "Process time is :" << fETime << endl;

	free(pfKernel);
	free(pfLaplacianKernelX);
	free(pfLaplacianKernelY);
	HANDLE_ERROR(cudaFree(pfLaplaceResultImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceResultImagePtr));
	HANDLE_ERROR(cudaFree(pfLaplaceImagePtrX1));
	HANDLE_ERROR(cudaFree(pfLaplaceImagePtrX2));
	HANDLE_ERROR(cudaFree(pfLaplaceImagePtrY1));
	HANDLE_ERROR(cudaFree(pfLaplaceImagePtrY2));
	HANDLE_ERROR(cudaFree(pfConvolutionImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceImagePtr));
}

void EnhanceImageChange(float* fImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, float fEnhanceSigma, float fEnhanceP) {
	float* pfDeviceImagePtr, * pfDeviceResultImagePtr, * pfEnhanceImagePtr, * pfFilterEnhanceImagePtr;

	//cudaEvent_t eventStart, eventEnd;
	//cudaEventCreate(&eventStart, 0);
	//cudaEventCreate(&eventEnd, 0);

	//cudaEventRecord(eventStart);
	HANDLE_ERROR(cudaMalloc(&pfDeviceImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfEnhanceImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfFilterEnhanceImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH));

	HANDLE_ERROR(cudaMemcpy(pfDeviceImagePtr, fImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));

	float* pfKernel = (float*)(malloc(sizeof(float) * 5 * 5));
	float* pfLPKernel = (float*)(malloc(sizeof(float) * 3 * 3));

	GenGauss2D(pfKernel, 5, fEnhanceSigma);
	GenLaplaceKernel2D3Sz(pfLPKernel);

	HANDLE_ERROR(cudaMemcpyToSymbol(cfGaussParam55, pfKernel, sizeof(float) * 5 * 5));
	HANDLE_ERROR(cudaMemcpyToSymbol(cfGaussParam33, pfLPKernel, sizeof(float) * 3 * 3));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageW / 16, nImageH / 16);

	Convolution2D3 << <d3Block, d3Thread >> > (pfDeviceImagePtr, pfEnhanceImagePtr, nImageW, nImageH);
	Convolution2D5 << <d3Block, d3Thread >> > (pfEnhanceImagePtr, pfFilterEnhanceImagePtr, nImageW, nImageH);
	AddImage << <d3Block, d3Thread >> > (pfDeviceImagePtr, pfFilterEnhanceImagePtr, pfDeviceResultImagePtr, -0.006f* fEnhanceP);


	HANDLE_ERROR(cudaMemcpy(pfResultImagePtr, pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	//cudaEventRecord(eventEnd);
	//cudaEventSynchronize(eventEnd);
	//float fETime;
	//cudaEventElapsedTime(&fETime, eventStart, eventEnd);

	//cout << "Process time is :" << fETime << endl;

	free(pfKernel);
	free(pfLPKernel);

	HANDLE_ERROR(cudaFree(pfDeviceResultImagePtr));
	HANDLE_ERROR(cudaFree(pfEnhanceImagePtr));
	HANDLE_ERROR(cudaFree(pfFilterEnhanceImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceImagePtr));
}

void EnhanceImageChange1(float* pfBKImagePtr, float* fDetailImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, float fEnhanceSigma, float fEnhanceThresh, float fEnhanceP) {
	float* pfDeviceBKImagePtr, * pfDeviceResultImagePtr, *pfDeviceDetailImagePtr, * pfEnhanceImagePtr, * pfFilterEnhanceImagePtr;


	HANDLE_ERROR(cudaMalloc(&pfDeviceBKImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceDetailImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfEnhanceImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfFilterEnhanceImagePtr, sizeof(float) * nImageW * nImageH));

	HANDLE_ERROR(cudaMemcpy(pfDeviceBKImagePtr, pfBKImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceDetailImagePtr, fDetailImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));

	float* pfKernel = (float*)(malloc(sizeof(float) * 5 * 5));
	float* pfLPKernel = (float*)(malloc(sizeof(float) * 3 * 3));

	GenGauss2D(pfKernel, 5, fEnhanceSigma);
	GenLaplaceKernel2D3Sz(pfLPKernel);

	HANDLE_ERROR(cudaMemcpyToSymbol(cfGaussParam55, pfKernel, sizeof(float) * 5 * 5));
	HANDLE_ERROR(cudaMemcpyToSymbol(cfGaussParam33, pfLPKernel, sizeof(float) * 3 * 3));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageW / 16, nImageH / 16);

	Convolution2D3 << <d3Block, d3Thread >> > (pfDeviceDetailImagePtr, pfEnhanceImagePtr, nImageW, nImageH);
	Convolution2D5 << <d3Block, d3Thread >> > (pfEnhanceImagePtr, pfFilterEnhanceImagePtr, nImageW, nImageH);
	AddImageThresh << <d3Block, d3Thread >> > (pfDeviceBKImagePtr, pfDeviceDetailImagePtr, pfFilterEnhanceImagePtr, pfDeviceResultImagePtr, fEnhanceThresh, -fEnhanceP);


	HANDLE_ERROR(cudaMemcpy(pfResultImagePtr, pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));

	free(pfKernel);
	free(pfLPKernel);


	HANDLE_ERROR(cudaFree(pfDeviceBKImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceResultImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceDetailImagePtr));
	HANDLE_ERROR(cudaFree(pfEnhanceImagePtr));
	HANDLE_ERROR(cudaFree(pfFilterEnhanceImagePtr));
}

void EnhanceImageChange2(float* pfBKImagePtr, float* fDetailImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, float fEnhanceThresh, float fEnhanceP) {
	float* pfDeviceBKImagePtr, * pfDeviceDetailImagePtr, * pfDeviceResultImagePtr, * pfDeviceEnhanceImagePtr;

	//cudaEvent_t eventStart, eventEnd;
	//cudaEventCreate(&eventStart, 0);
	//cudaEventCreate(&eventEnd, 0);

	//cudaEventRecord(eventStart);
	HANDLE_ERROR(cudaMalloc(&pfDeviceBKImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceDetailImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceEnhanceImagePtr, sizeof(float) * nImageW * nImageH));

	HANDLE_ERROR(cudaMemcpy(pfDeviceBKImagePtr, pfBKImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceDetailImagePtr, fDetailImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));

	dim3 d3Block(nImageW/16, nImageH/16), d3Thread(16, 16);

	multiple << <d3Block, d3Thread >> > (pfDeviceDetailImagePtr, fEnhanceP, pfDeviceEnhanceImagePtr);
	AddImageThresh << <d3Block, d3Thread >> > (pfDeviceBKImagePtr, pfDeviceDetailImagePtr, pfDeviceEnhanceImagePtr, pfDeviceResultImagePtr, fEnhanceThresh);

	HANDLE_ERROR(cudaMemcpy(pfResultImagePtr, pfDeviceResultImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	//cudaEventRecord(eventEnd);
	//cudaEventSynchronize(eventEnd);
	//float fETime;
	//cudaEventElapsedTime(&fETime, eventStart, eventEnd);

	//cout << "Process time is :" << fETime << endl;

	HANDLE_ERROR(cudaFree(pfDeviceBKImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceResultImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceDetailImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceEnhanceImagePtr));

}

__global__ void Norm(float* pfInputImagePtr, float* pfNormImagePtr) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	pfNormImagePtr[nOffset] = pfInputImagePtr[nOffset] / 255.0;
}

__global__ void Multiple(float* pfInputImagePtr, float* pfNormImagePtr, float fM) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	pfNormImagePtr[nOffset] = fM * pfInputImagePtr[nOffset];
}

__global__ void CENhance(float* pfSrcImagePtr, float* pfGaussImagePtr, float* pfLEnhanceImagePtr, float* pfEnhanceImagePtr, float fP, int nImageW, int nImageH) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));
	float fE, fTempE;

	//if (pfSrcImagePtr[nOffset] <1.0f || pfGaussImagePtr[nOffset] < 1.0f) {
	//	fTempSrcV = 1.0f;
	//	fTempGauV = 1.0f;
	//}
	//else {
	//	fTempSrcV = pfSrcImagePtr[nOffset];
	//	fTempGauV = pfGaussImagePtr[nOffset];
	//}

	fTempE = pfGaussImagePtr[nOffset] / pfSrcImagePtr[nOffset];
	if (fTempE < 0.8f) {
		fTempE = 0.8f;
	}
	//if (fTempE > 10.0f) {
	//	fTempE = 10.0f;
	//}

	fE = powf(fTempE, fP);


	float fTempV = 255.0f * powf(pfLEnhanceImagePtr[nOffset], fE);
	if (fTempV > 254.5f)
		fTempV = 255.0f;
	if (fTempV < 0.5f)
		fTempV = 0.0f;
	pfEnhanceImagePtr[nOffset] = fTempV;

}

__global__ void LENhance(float* pInputImagePtr, float* pfEnhanceImagePtr, int nImageW, int nImageH) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));



	float fRV = 0.0f;

	float fGV = pInputImagePtr[nOffset];
	float fV = (pow(fGV, 0.75 * fRV + 0.25) + 0.4 * (1 - fRV) * (1 - fGV) + fGV * (1 - fRV)) / 2.0f;
	pfEnhanceImagePtr[nOffset] = fV;
}
__global__ void GENhance(float* pInputImagePtr, float* pfEnhanceImagePtr, float fGamma, int nImageW, int nImageH) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	float fGV = pInputImagePtr[nOffset];
	float fV = pow(fGV, fGamma);
	pfEnhanceImagePtr[nOffset] = fV;
}

__global__ void CEConvolutionRow(float* pfImagePtr, float* pfConvoImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nShift = __mul24(threadIdx.y, (nBlockDimX + nCEKernelRadius * 2));
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[nBlockDimY * (nBlockDimX + 2 * nCEKernelRadius)];

	if (threadIdx.x < nCEKernelRadius || threadIdx.x >= nBlockDimX - nCEKernelRadius) {
		if ((nX - nCEKernelRadius) < 0) {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - nCEKernelRadius];
		}

		if ((nX + nCEKernelRadius) > nImageW - 1) {
			fConvData[threadIdx.x + 2 * nCEKernelRadius + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + 2 * nCEKernelRadius + nShift] = pfImagePtr[nOffset + nCEKernelRadius];
		}
		fConvData[threadIdx.x + nCEKernelRadius + nShift] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + nCEKernelRadius + nShift] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;
	int nConX = nCEKernelRadius + threadIdx.x;

	for (int i = -nCEKernelRadius; i <= nCEKernelRadius; i++) {
		fSum += fConvData[nConX + i + nShift] * cfCEGaussParam[nCEKernelRadius + i];
	}

	pfConvoImagePtr[nOffset] = fSum;
}

__global__ void CEConvolutionCol(float* pfImagePtr, float* pfConvoImagePtr, int nImageW, int nImageH) {
	//int nBlockDimX = blockDim.x;
	//int nBlockDimY = blockDim.y;
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nShift = __mul24(threadIdx.y, nBlockDimX);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[nBlockDimX * (nBlockDimY + 2 * nCEKernelRadius)];

	if (threadIdx.y < nCEKernelRadius || threadIdx.y >= nBlockDimY - nCEKernelRadius) {
		if ((nY - nCEKernelRadius) < 0) {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfImagePtr[nOffset - __mul24(nImageW, nCEKernelRadius)];
		}

		const int nShift1 = nShift + __mul24(2 * nCEKernelRadius, nBlockDimX);
		const int nShift2 = nShift + __mul24(nCEKernelRadius, nBlockDimX);
		if ((nY + nCEKernelRadius) > nImageH - 1) {
			fConvData[threadIdx.x + nShift1] = pfImagePtr[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift1] = pfImagePtr[nOffset + __mul24(nImageW, nCEKernelRadius)];
		}
		fConvData[threadIdx.x + nShift2] = pfImagePtr[nOffset];
	}
	else {
		fConvData[threadIdx.x + nCEKernelRadius * nBlockDimX + nShift] = pfImagePtr[nOffset];
	}

	__syncthreads();

	//convolution
	float fSum = 0.0f;
	//int nConX = nKernelRSz + threadIdx.x;

	for (int i = 0; i <= nCEKernelRadius * 2; i++) {
		fSum += fConvData[threadIdx.x + (threadIdx.y + i) * nBlockDimX] * cfCEGaussParam[i];
	}

	pfConvoImagePtr[nOffset] = fSum;
}

void AINDANEEnhance(float* pfSrcImage, float* pfEnhanceImage, float fGamma, float fP, bool bCEnhance, int nImageW, int nImageH) {
	float* pfDeviceSrcImage, * pfDeviceNormSrcImage, * pfEnhanceImagePtr, * pfLENhanceResult, * pfGENhanceResult,
		* pfDeviceGaussImageRow, * pfDeviceGaussImageCol;

	HANDLE_ERROR(cudaMalloc(&pfDeviceSrcImage, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceNormSrcImage, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfEnhanceImagePtr, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfLENhanceResult, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfGENhanceResult, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGaussImageRow, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGaussImageCol, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMemcpy(pfDeviceSrcImage, pfSrcImage, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));

	float pfGaussKernel[nMeanKernelSz * nMeanKernelSz];

	GenGauss1D(pfGaussKernel, 5, 1.0f);

	HANDLE_ERROR(cudaMemcpyToSymbol(cfCEGaussParam, pfGaussKernel, sizeof(float) * nMeanKernelSz));

	dim3 d3Block(nImageW / 16, nImageH / 16), d3Thread(16, 16);

	Norm << <d3Block, d3Thread >> > (pfDeviceSrcImage, pfDeviceNormSrcImage);
	//LENhance << < d3Block, d3Thread >> > (pfDeviceNormSrcImage, pfLENhanceResult, nImageW, nImageH);
	GENhance << < d3Block, d3Thread >> > (pfDeviceNormSrcImage, pfLENhanceResult, fGamma, nImageW, nImageH);

	if (bCEnhance) {
		CEConvolutionRow << <d3Block, d3Thread >> > (pfDeviceSrcImage, pfDeviceGaussImageRow, nImageW, nImageH);
		CEConvolutionCol << <d3Block, d3Thread >> > (pfDeviceGaussImageRow, pfDeviceGaussImageCol, nImageW, nImageH);
		CENhance << <d3Block, d3Thread >> > (pfDeviceSrcImage, pfDeviceGaussImageCol, pfLENhanceResult, pfEnhanceImagePtr, fP, nImageW, nImageH);
		HANDLE_ERROR(cudaMemcpy(pfEnhanceImage, pfEnhanceImagePtr, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	}
	else {
		Multiple << <d3Block, d3Thread >> > (pfLENhanceResult, pfGENhanceResult, 255.0);
		HANDLE_ERROR(cudaMemcpy(pfEnhanceImage, pfGENhanceResult, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));
	}

	HANDLE_ERROR(cudaFree(pfDeviceSrcImage));
	HANDLE_ERROR(cudaFree(pfDeviceNormSrcImage));
	HANDLE_ERROR(cudaFree(pfEnhanceImagePtr));
	HANDLE_ERROR(cudaFree(pfLENhanceResult));
	HANDLE_ERROR(cudaFree(pfGENhanceResult));
	HANDLE_ERROR(cudaFree(pfDeviceGaussImageRow));
	HANDLE_ERROR(cudaFree(pfDeviceGaussImageCol));
}

__constant__ double pdConstCurveImagePts[32];
__constant__ double pdConstCurveImageParams[60];

__global__ void CurveImage(float* pfSrcImage, float* pfEnhanceImage, int nArraySz) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;


	float fDiffV;
	int nLineC;
	for (int n = 0; n < nArraySz; n++) {
		if (pfSrcImage[nOffset] < pdConstCurveImagePts[2 * n + 0]) {
			nLineC = n-1;
			break;
		}
	}

	fDiffV = pfSrcImage[nOffset] - pdConstCurveImagePts[2 * nLineC + 0];

	pfEnhanceImage[nOffset] = pdConstCurveImageParams[4 * nLineC + 0] + pdConstCurveImageParams[4 * nLineC + 1] * (fDiffV)+pdConstCurveImageParams[4 * nLineC + 2] * fDiffV * fDiffV + pdConstCurveImageParams[4 * nLineC + 3] * fDiffV * fDiffV * fDiffV;

	if (pfEnhanceImage[nOffset] < 1.0f) {
		pfEnhanceImage[nOffset] = 1.0f;
	}
	if (pfEnhanceImage[nOffset] > 255.0f) {
		pfEnhanceImage[nOffset] = 255.0f;
	}
}

void CurveImageEnhance(float* pfSrcImage, float* pfEnhanceImage, double* pdCurveImagePts, double* pdCurveImageParams, int nCurveImageArraySz, int nImageW, int nImageH) {
	float* pfDeviceSrcImage, * pfDeviceEnhanceImage;

	HANDLE_ERROR(cudaMalloc(&pfDeviceSrcImage, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceEnhanceImage, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMemcpy(pfDeviceSrcImage, pfSrcImage, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(pdConstCurveImagePts, pdCurveImagePts, sizeof(double) * nCurveImageArraySz * 2));
	HANDLE_ERROR(cudaMemcpyToSymbol(pdConstCurveImageParams, pdCurveImageParams, sizeof(double) * (nCurveImageArraySz - 1) * 4));

	dim3 d3Thread(16, 16), d3Block(nImageW / 16, nImageH / 16);
	CurveImage << < d3Block, d3Thread >> > (pfDeviceSrcImage, pfDeviceEnhanceImage, nCurveImageArraySz);

	HANDLE_ERROR(cudaMemcpy(pfEnhanceImage, pfDeviceEnhanceImage, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceSrcImage));
	HANDLE_ERROR(cudaFree(pfDeviceEnhanceImage));
	delete[] pdCurveImagePts;
	delete[] pdCurveImageParams;
}

__global__ void BoxFilter(float* pfSrcImagePtr, float* pfFilterPtr, int nImageW, int nImageH) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	__shared__ float fTempImageBuffer[(nBlockDimX + nMultiple * nBoxFKernelRadius) * (nBlockDimY + nMultiple * nBoxFKernelRadius)];

	int nBufferWidth = blockDim.x + __mul24(nMultiple, nBoxFKernelRadius);
	if (nX < nBoxFKernelRadius || nX > nBlockDimX - 1 - nBoxFKernelRadius ||
		nY < nBoxFKernelRadius || nY > nBlockDimY - 1 - nBoxFKernelRadius) {
		//upper left
		if (nY - nBoxFKernelRadius < 0 || nX - nBoxFKernelRadius < 0) {
			fTempImageBuffer[threadIdx.x + __mul24(threadIdx.y, nBufferWidth)] = /*pfSrcImagePtr[nOffset]*/0.0f;
		}
		else {
			fTempImageBuffer[threadIdx.x + __mul24(threadIdx.y, nBufferWidth)] = pfSrcImagePtr[nOffset - nBoxFKernelRadius - __mul24(nBoxFKernelRadius, nImageW)];
		}
		//upper right
		if (nY - nBoxFKernelRadius < 0 || nX + nBoxFKernelRadius > nImageW - 1) {
			fTempImageBuffer[threadIdx.x + __mul24(2, nBoxFKernelRadius) + __mul24(threadIdx.y, nBufferWidth)] = 0.0f;
		}
		else {
			fTempImageBuffer[threadIdx.x + __mul24(2, nBoxFKernelRadius) + __mul24(threadIdx.y, nBufferWidth)] = pfSrcImagePtr[nOffset + nBoxFKernelRadius - __mul24(nBoxFKernelRadius, nImageW)];
		}
		//lower left
		if (nY + nBoxFKernelRadius > nImageH - 1 || nX - nBoxFKernelRadius < 0) {
			fTempImageBuffer[threadIdx.x + __mul24(threadIdx.y + __mul24(2, nBoxFKernelRadius), nBufferWidth)] = 0.0f;
		}
		else {
			fTempImageBuffer[threadIdx.x + __mul24(threadIdx.y + __mul24(2, nBoxFKernelRadius), nBufferWidth)] = pfSrcImagePtr[nOffset - nBoxFKernelRadius + __mul24(nBoxFKernelRadius, nImageW)];
		}
		//lower right
		if (nY + nBoxFKernelRadius > nImageH - 1 || nX + nBoxFKernelRadius > nImageW - 1) {
			fTempImageBuffer[threadIdx.x + __mul24(2, nBoxFKernelRadius) + __mul24(threadIdx.y + __mul24(2, nBoxFKernelRadius), nBufferWidth)] = 0.0f;
		}
		else {
			fTempImageBuffer[threadIdx.x + __mul24(2, nBoxFKernelRadius) + __mul24(threadIdx.y + __mul24(2, nBoxFKernelRadius), nBufferWidth)] = pfSrcImagePtr[nOffset + nBoxFKernelRadius + __mul24(nBoxFKernelRadius, nImageW)];
		}

		fTempImageBuffer[threadIdx.x + nBoxFKernelRadius + __mul24(threadIdx.y + nBoxFKernelRadius, nBufferWidth)] = pfSrcImagePtr[nOffset];
	}
	else {
		fTempImageBuffer[threadIdx.x + nBoxFKernelRadius + __mul24(threadIdx.y + nBoxFKernelRadius, nBufferWidth)] = pfSrcImagePtr[nOffset];
	}

	__syncthreads();

	int n, r, c;

	pfFilterPtr[nOffset] = 0.0f;

	for (n = 0; n < nBoxFKernelSz * nBoxFKernelSz; n++) {
		r = n / nBoxFKernelSz;
		c = n % nBoxFKernelSz;

		pfFilterPtr[nOffset] += fTempImageBuffer[threadIdx.x + c + (threadIdx.y + r) * nBufferWidth];
	}

	pfFilterPtr[nOffset] = fabs(pfFilterPtr[nOffset] / (nBoxFKernelSz * nBoxFKernelSz));
}

__global__ void division(float* pfSrcImage, float* pfDstImage, float fDiv) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfDstImage[nOffset] = pfSrcImage[nOffset] / fDiv;
}

__global__ void division(float* pfSrcImage, float* pfParamImage, float* pfDstImage) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfDstImage[nOffset] = pfSrcImage[nOffset] / pfParamImage[nOffset];
}

__global__ void multiple(float* pfSrcImage, float* pfParamImage, float* pfDstImage) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfDstImage[nOffset] = pfSrcImage[nOffset] * pfParamImage[nOffset];
}




__global__ void substract(float* pfSrcImage, float* pfParamImage, float* pfDstImage) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfDstImage[nOffset] = pfSrcImage[nOffset] - pfParamImage[nOffset];

}

__global__ void substract(float* pfSrcImage, float* pfDstImage) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfSrcImage[nOffset] = pfSrcImage[nOffset] - pfDstImage[nOffset];
	if (pfSrcImage[nOffset] < 1e-4f)
		pfSrcImage[nOffset] = 1e-4f;
}

__global__ void add(float* pfSrcImage, float fV, float* pfDstImage) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfDstImage[nOffset] = pfSrcImage[nOffset] + fV;
}

__global__ void add(float* pfSrcImage, float* pfParamImage, float* pfDstImage) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfDstImage[nOffset] = pfSrcImage[nOffset] + pfParamImage[nOffset];
}

__global__ void MeanImage(float* pfMeanImage, float* pfSrcImage, int nMeanFN) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = __mul24(nY, __mul24(blockDim.x, gridDim.x)) + nX;

	pfMeanImage[nOffset] = (pfMeanImage[nOffset] * (nMeanFN)+pfSrcImage[nOffset]) / (float)(nMeanFN + 1);
}


void DarkMeanImage(float* pfMeanImage, float* pfSrcImage, int nImageW, int nImageH, int nMeanFN) {
	float* pfDeviceMeanImage, * pfDeviceSrcImage;

	HANDLE_ERROR(cudaMalloc(&pfDeviceMeanImage, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceSrcImage, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMemcpy(pfDeviceMeanImage, pfMeanImage, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceSrcImage, pfSrcImage, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16), d3Block(nImageW / 16, nImageH / 16);
	MeanImage << <d3Block, d3Thread >> > (pfDeviceMeanImage, pfDeviceSrcImage, nMeanFN);

	HANDLE_ERROR(cudaMemcpy(pfMeanImage, pfDeviceMeanImage, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceMeanImage));
	HANDLE_ERROR(cudaFree(pfDeviceSrcImage));
}

void DarkRefine(float* pfSrcImage, float* pfDarkImage, int nImageW, int nImageH) {
	float* pfDeviceSrcImage, * pfDeviceDarkImage;

	int nImageBufferSz = sizeof(float) * nImageW * nImageH;
	HANDLE_ERROR(cudaMalloc(&pfDeviceSrcImage, nImageBufferSz));
	HANDLE_ERROR(cudaMalloc(&pfDeviceDarkImage, nImageBufferSz));
	HANDLE_ERROR(cudaMemcpy(pfDeviceSrcImage, pfSrcImage, nImageBufferSz, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceDarkImage, pfDarkImage, nImageBufferSz, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16), d3Block(nImageW / 16, nImageH / 16);
	substract << <d3Block, d3Thread >> > (pfDeviceSrcImage, pfDeviceDarkImage);

	HANDLE_ERROR(cudaMemcpy(pfSrcImage, pfDeviceSrcImage, nImageBufferSz, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceSrcImage));
	HANDLE_ERROR(cudaFree(pfDeviceDarkImage));
}

void substractDB(float* pfOriginImage, float* pfBKImage, float* pfDetailImage, int nWidth, int nHeight) {
	float* pfDeviceOriginImage, * pfDeviceBKImage, * pfDeviceDetialImage;

	HANDLE_ERROR(cudaMalloc(&pfDeviceOriginImage, sizeof(float)* nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceBKImage, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceDetialImage, sizeof(float) * nWidth * nHeight));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nWidth / 16, nHeight / 16);

	HANDLE_ERROR(cudaMemcpy(pfDeviceOriginImage, pfOriginImage, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceBKImage, pfBKImage, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	substract << <d3Block, d3Thread >> > (pfDeviceOriginImage, pfDeviceBKImage, pfDeviceDetialImage);
	HANDLE_ERROR(cudaMemcpy(pfDetailImage, pfDeviceDetialImage, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceOriginImage));
	HANDLE_ERROR(cudaFree(pfDeviceBKImage));
	HANDLE_ERROR(cudaFree(pfDeviceDetialImage));
}

void GaussianFilter(float* pfMatI, float* pfGaussianF, float* pfResultImg, int nImageW, int nImageH, int nKernelSz, double dSigma) {
	float* pfDeviceMatI, * pfDeviceResultImage;

	HANDLE_ERROR(cudaMalloc(&pfDeviceMatI, sizeof(float) * nImageW * nImageH));

	HANDLE_ERROR(cudaMalloc(&pfDeviceResultImage, sizeof(float) * nImageW * nImageH));

	HANDLE_ERROR(cudaMemcpy(pfDeviceMatI, pfMatI, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));

	float* pfKernel = (float*)(malloc(sizeof(float) * 5 * 5));
	GenGauss2D(pfKernel, 5, dSigma);

	HANDLE_ERROR(cudaMemcpyToSymbol(cfGaussParam55, pfKernel, sizeof(float) * 5 * 5));

	dim3 d3Block(nImageW / 16, nImageH / 16), d3Thread(16, 16);
	Convolution2D5<<<d3Block , d3Thread >>>(pfDeviceMatI, pfDeviceResultImage, nImageW, nImageH);

	HANDLE_ERROR(cudaMemcpy(pfResultImg, pfDeviceResultImage, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));


	free(pfKernel);
	HANDLE_ERROR(cudaFree(pfDeviceMatI));

	HANDLE_ERROR(cudaFree(pfDeviceResultImage));

}

void GuidedFilter(float* pfMatI, float* pfMatP, float* pfResultImg, int nImageW, int nImageH, int nKernelSz, float fEpsilon) {
	float* pfDeviceMatI, * pfDeviceMatP, * pfDeviceResultImg/*, * pfOnesImage*//*, * pfMeanOneImage*//*,
		* pfMeanDeviceMatI*/, * pfMeanI/*, *pfMeanDeviceMatP*/, * pfMeanP,
		* pfMatII, * pfMatIP/*, *pfmatTempMeanIP*/, * pfmatMeanIP,
		* pfMeanIMeanP, * pfCovIP/*, *pfTempMeanII*/, * pfMeanII,
		* pfMeanIMeanI, * pfVarI, * pfDA, * pfA, * pfB, * pfAMeanI/*,
		*pfTempMeanA*/, * pfMeanA/*, *pfTempMeanB*/, * pfMeanB, * pfMeanAI;

		//cv::Mat matOnesImage = cv::Mat::ones(nImageH, nImageW, CV_32FC1);
	HANDLE_ERROR(cudaMalloc(&pfDeviceMatI, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceMatP, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDeviceResultImg, sizeof(float) * nImageW * nImageH));
	//HANDLE_ERROR(cudaMalloc(&pfOnesImage, sizeof(float) * nImageW * nImageH));
	//HANDLE_ERROR(cudaMalloc(&pfMeanOneImage, sizeof(float) * nImageW * nImageH));
	//HANDLE_ERROR(cudaMalloc(&pfMeanDeviceMatI, sizeof(float) * nImageW * nImageH));
	//HANDLE_ERROR(cudaMalloc(&pfMeanDeviceMatP, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMeanI, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMeanP, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMatII, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMatIP, sizeof(float) * nImageW * nImageH));
	//HANDLE_ERROR(cudaMalloc(&pfmatTempMeanIP, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfmatMeanIP, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMeanIMeanP, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfCovIP, sizeof(float) * nImageW * nImageH));
	//HANDLE_ERROR(cudaMalloc(&pfTempMeanII, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMeanII, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMeanIMeanI, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfVarI, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfA, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfB, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfDA, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfAMeanI, sizeof(float) * nImageW * nImageH));
	//HANDLE_ERROR(cudaMalloc(&pfTempMeanA, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMeanA, sizeof(float) * nImageW * nImageH));
	//HANDLE_ERROR(cudaMalloc(&pfTempMeanB, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMeanB, sizeof(float) * nImageW * nImageH));
	HANDLE_ERROR(cudaMalloc(&pfMeanAI, sizeof(float) * nImageW * nImageH));

	HANDLE_ERROR(cudaMemcpy(pfDeviceMatI, pfMatI, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceMatP, pfMatP, sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(pfOnesImage, matOnesImage.ptr<float>(0), sizeof(float) * nImageW * nImageH, cudaMemcpyHostToDevice));


	dim3 d3Thread(16, 16), d3Block(nImageW / 16, nImageH / 16);
	//BoxFilter << <d3Block, d3Thread >> > (pfOnesImage, pfMeanOneImage, nImageW, nImageH);
	BoxFilter << <d3Block, d3Thread >> > (pfDeviceMatI, pfMeanI, nImageW, nImageH);
	//division << <d3Block, d3Thread >> > (pfMeanDeviceMatI, pfMeanOneImage, pfMeanI);

	//HANDLE_ERROR(cudaFree(pfOnesImage));
	//HANDLE_ERROR(cudaFree(pfMeanDeviceMatI));

	BoxFilter << <d3Block, d3Thread >> > (pfDeviceMatP, pfMeanP, nImageW, nImageH);
	//division << <d3Block, d3Thread >> > (pfMeanDeviceMatP, pfMeanOneImage, pfMeanP);

	//HANDLE_ERROR(cudaFree(pfMeanDeviceMatP));

	multiple << <d3Block, d3Thread >> > (pfDeviceMatI, pfDeviceMatI, pfMatII);
	multiple << <d3Block, d3Thread >> > (pfDeviceMatI, pfDeviceMatP, pfMatIP);

	BoxFilter << <d3Block, d3Thread >> > (pfMatIP, pfmatMeanIP, nImageW, nImageH);
	//division << <d3Block, d3Thread >> > (pfmatTempMeanIP, pfMeanOneImage, pfmatMeanIP);
	HANDLE_ERROR(cudaFree(pfMatIP));
	//HANDLE_ERROR(cudaFree(pfmatTempMeanIP));

	multiple << <d3Block, d3Thread >> > (pfMeanI, pfMeanP, pfMeanIMeanP);
	substract << <d3Block, d3Thread >> > (pfmatMeanIP, pfMeanIMeanP, pfCovIP);
	HANDLE_ERROR(cudaFree(pfmatMeanIP));
	HANDLE_ERROR(cudaFree(pfMeanIMeanP));

	BoxFilter << <d3Block, d3Thread >> > (pfMatII, pfMeanII, nImageW, nImageH);
	//division << <d3Block, d3Thread >> > (pfTempMeanII, pfMeanOneImage, pfMeanII);
	HANDLE_ERROR(cudaFree(pfMatII));
	//HANDLE_ERROR(cudaFree(pfTempMeanII));
	multiple << <d3Block, d3Thread >> > (pfMeanI, pfMeanI, pfMeanIMeanI);
	substract << <d3Block, d3Thread >> > (pfMeanII, pfMeanIMeanI, pfVarI);

	HANDLE_ERROR(cudaFree(pfMeanII));
	HANDLE_ERROR(cudaFree(pfMeanIMeanI));

	add << <d3Block, d3Thread >> > (pfVarI, fEpsilon, pfDA);

	HANDLE_ERROR(cudaFree(pfVarI));

	division << <d3Block, d3Thread >> > (pfCovIP, pfDA, pfA);
	HANDLE_ERROR(cudaFree(pfCovIP));
	HANDLE_ERROR(cudaFree(pfDA));

	multiple << <d3Block, d3Thread >> > (pfA, pfMeanI, pfAMeanI);
	substract << <d3Block, d3Thread >> > (pfMeanP, pfAMeanI, pfB);

	HANDLE_ERROR(cudaFree(pfAMeanI));
	HANDLE_ERROR(cudaFree(pfMeanP));
	HANDLE_ERROR(cudaFree(pfMeanI));

	BoxFilter << <d3Block, d3Thread >> > (pfA, pfMeanA, nImageW, nImageH);
	//division << <d3Block, d3Thread >> > (pfTempMeanA, pfMeanOneImage, pfMeanA);
	BoxFilter << <d3Block, d3Thread >> > (pfB, pfMeanB, nImageW, nImageH);
	//division << <d3Block, d3Thread >> > (pfTempMeanB, pfMeanOneImage, pfMeanB);

	//HANDLE_ERROR(cudaFree(pfTempMeanA));
	//HANDLE_ERROR(cudaFree(pfMeanOneImage));
	//HANDLE_ERROR(cudaFree(pfTempMeanB));
	HANDLE_ERROR(cudaFree(pfA));
	HANDLE_ERROR(cudaFree(pfB));

	multiple << <d3Block, d3Thread >> > (pfMeanA, pfDeviceMatI, pfMeanAI);
	add << <d3Block, d3Thread >> > (pfMeanAI, pfMeanB, pfDeviceResultImg);

	HANDLE_ERROR(cudaMemcpy(pfResultImg, pfDeviceResultImg, sizeof(float) * nImageW * nImageH, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfMeanA));
	HANDLE_ERROR(cudaFree(pfMeanAI));
	HANDLE_ERROR(cudaFree(pfMeanB));
	HANDLE_ERROR(cudaFree(pfDeviceMatI));
	HANDLE_ERROR(cudaFree(pfDeviceMatP));
	HANDLE_ERROR(cudaFree(pfDeviceResultImg));
}

const int cnCCMW = 3;
const int cnCCMH = 4;
__constant__ float fCCM[cnCCMW * cnCCMH];
__constant__ int nHSVAlterTag[1];
__constant__ int nYUVAlterTag[1];
__constant__ int nHueV[1];
__constant__ int nSaturationV[1];
__constant__ int nSaturation1V[1];
__constant__ int nSaturation1BackupV[1];
__constant__ int nValueV[1];

__global__ void CCM(float* pfY, float* pfU, float* pfV, float* pfCY, float* pfCU, float* pfCV) {
	//int nX = __hadd(threadIdx.x, __mul24(blockIdx.x, blockDim.x));
	//int nY = __hadd(threadIdx.y, __mul24(blockIdx.y, blockDim.y));
	//int nOffset = __hadd(nX , __mul24(nY, __mul24(blockDim.x, gridDim.x)));

	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	float fR, fG, fB, fCR, fCG, fCB;


	//fR = pfY[nOffset] + 1.403f * (pfU[nOffset] - 128.0f);
	//fG = pfY[nOffset] - 0.714f * (pfU[nOffset] - 128.0f) - 0.344f * (pfV[nOffset] - 128.0f);
	//fB = pfY[nOffset] + 1.773f * (pfV[nOffset] - 128.0f);
	//YUV convertTo RGB
	fR = __fadd_rz(pfY[nOffset], __fmul_rz(1.403f, __fadd_rz(pfU[nOffset], -128.0f)));
	fG = __fadd_rz(pfY[nOffset], __fadd_rz(__fmul_rz(-0.714f, __fadd_rz(pfU[nOffset], -128.0f)), __fmul_rz(-0.344f, __fadd_rz(pfV[nOffset], -128.0f))));
	fB = __fadd_rz(pfY[nOffset], __fmul_rz(1.773f, __fadd_rz(pfV[nOffset], -128.0f)));

	float fCCCM[cnCCMW * cnCCMH];
	if (1 == nYUVAlterTag[0]) {
		float SMatrix[cnCCMW][cnCCMH];

		float fMA, fMB, fMC, fMD, fME, fMF, fMG, fMH, fMI,
			fMJ, fMK, fML, fMM, fMN, fMO, fMP, fMQ, fMR,
			fMS, fMT, fMU, fMV, fMW, fMX;

		float fS = nSaturation1V[0] / 100.0f;

		float rwgt = 0.3086f;
		float gwgt = 0.6094f;
		float bwgt = 0.0820f;

		fMA = (__fmul_rz(__fadd_rz(1.0f, -fS), rwgt) + fS);
		fMB = (__fmul_rz(__fadd_rz(1.0f, - fS), rwgt));
		fMC = (__fmul_rz(__fadd_rz(1.0f, - fS), rwgt));
		fMD = (__fmul_rz(__fadd_rz(1.0f, - fS), gwgt));
		fME = (__fmul_rz(__fadd_rz(1.0f, - fS), gwgt) + fS);
		fMF = (__fmul_rz(__fadd_rz(1.0f, - fS), gwgt));
		fMG = (__fmul_rz(__fadd_rz(1.0f, - fS), bwgt));
		fMH = (__fmul_rz(__fadd_rz(1.0f, - fS), bwgt));
		fMI = (__fmul_rz(__fadd_rz(1.0f, - fS), bwgt) + fS);

		//fMA = 1.0f;
		//fMB = 0.0f;
		//fMC = 0.0f;
		//fMD = 0.0f;
		//fME = 1.0f;
		//fMF = 0.0f;
		//fMG = 0.0f;
		//fMH = 0.0f;
		//fMI = 1.0f;

		SMatrix[0][0] = fMA; SMatrix[1][0] = fMB; SMatrix[2][0] = fMC; SMatrix[0][3] = 0.0f;
		SMatrix[0][1] = fMD; SMatrix[1][1] = fME; SMatrix[2][1] = fMF; SMatrix[1][3] = 0.0f;
		SMatrix[0][2] = fMG; SMatrix[1][2] = fMH; SMatrix[2][2] = fMI; SMatrix[2][3] = 0.0f;

		fMA = fCCM[0]; fMB = fCCM[1]; fMC = fCCM[2];   fMJ = fCCM[3];
		fMD = fCCM[4]; fME = fCCM[5]; fMF = fCCM[6];   fMK = fCCM[7];
		fMG = fCCM[8]; fMH = fCCM[9]; fMI = fCCM[10];   fML = fCCM[11];

		fMM = SMatrix[0][0]; fMN = SMatrix[0][1]; fMO = SMatrix[0][2];   fMV = SMatrix[0][3];
		fMP = SMatrix[1][0]; fMQ = SMatrix[1][1]; fMR = SMatrix[1][2];   fMW = SMatrix[1][3];
		fMS = SMatrix[2][0]; fMT = SMatrix[2][1]; fMU = SMatrix[2][2];   fMX = SMatrix[2][3];

		fCCCM[0] = fMM * fMA + fMN * fMD + fMO * fMG; fCCCM[1] = fMM * fMB + fMN * fME + fMO * fMH; fCCCM[2] = fMM * fMC + fMN * fMF + fMO * fMI; fCCCM[3] = fMM * fMJ + fMN * fMK + fMO * fML + fMV;
		fCCCM[4] = fMP * fMA + fMQ * fMD + fMR * fMG; fCCCM[5] = fMP * fMB + fMQ * fME + fMR * fMH; fCCCM[6] = fMP * fMC + fMQ * fMF + fMR * fMI; fCCCM[7] = fMP * fMJ + fMQ * fMK + fMR * fML + fMW;
		fCCCM[8] = fMS * fMA + fMT * fMD + fMU * fMG; fCCCM[9] = fMS * fMB + fMT * fME + fMU * fMH; fCCCM[10] = fMS * fMC + fMT * fMF + fMU * fMI; fCCCM[11] = fMS * fMJ + fMT * fMK + fMU * fML + fMX;

		fCR = __fadd_rz(__fadd_rz(fCCCM[3], __fmul_rz(fR, fCCCM[0])), __fadd_rz(__fmul_rz(fG, fCCCM[1]), __fmul_rz(fB, fCCCM[2])));
		fCG = __fadd_rz(__fadd_rz(fCCCM[7], __fmul_rz(fR, fCCCM[4])), __fadd_rz(__fmul_rz(fG, fCCCM[5]), __fmul_rz(fB, fCCCM[6])));
		fCB = __fadd_rz(__fadd_rz(fCCCM[11], __fmul_rz(fR, fCCCM[8])), __fadd_rz(__fmul_rz(fG, fCCCM[9]), __fmul_rz(fB, fCCCM[10])));
	}
	else {
		//Apply CCM
		fCR = __fadd_rz(__fadd_rz(fCCM[3], __fmul_rz(fR, fCCM[0])), __fadd_rz(__fmul_rz(fG, fCCM[1]), __fmul_rz(fB, fCCM[2])));
		fCG = __fadd_rz(__fadd_rz(fCCM[7], __fmul_rz(fR, fCCM[4])), __fadd_rz(__fmul_rz(fG, fCCM[5]), __fmul_rz(fB, fCCM[6])));
		fCB = __fadd_rz(__fadd_rz(fCCM[11], __fmul_rz(fR, fCCM[8])), __fadd_rz(__fmul_rz(fG, fCCM[9]), __fmul_rz(fB, fCCM[10])));
	}

	if (1 == nHSVAlterTag[0]) {
		//RGB To HSV
		float fNCR = __fdiv_rz(fCR, 255.0f);
		float fNCG = __fdiv_rz(fCG, 255.0f);
		float fNCB = __fdiv_rz(fCB, 255.0f);

		float fMax, fMin;
		if (fNCR > fNCG && fNCR > fNCB) {
			fMax = fNCR;
			if (fNCG < fNCB) {
				fMin = fNCG;
			}
			else {
				fMin = fNCB;
			}
		}
		if (fNCG > fNCR && fNCG > fNCB) {
			fMax = fNCG;
			if (fNCR < fNCB) {
				fMin = fNCR;
			}
			else {
				fMin = fNCB;
			}
		}
		if (fNCB > fNCR && fNCB > fNCG) {
			fMax = fNCB;
			if (fNCR < fNCG) {
				fMin = fNCR;
			}
			else {
				fMin = fNCG;
			}
		}
		float fDelta = __fadd_rz(fMax, -fMin);

		float fHue, fSaturation, fValue;
		if (fabs(fDelta) < 1e-6f) {
			fHue = 0.0f;
		}
		else if (fabs(__fadd_rz(fMax, -fNCR)) < 1e-6) {
			fHue = __fmul_rz(fmodf((__fadd_rz(fNCG, -fNCB) / fDelta), 6.0f), 60.0f);
		}
		else if (fabs(__fadd_rz(fMax, -fNCG)) < 1e-6) {
			fHue = __fmul_rz((__fadd_rz(__fadd_rz(fNCB, -fNCR) / fDelta, 2.0f)), 60.0f);
		}
		else if (fabs(__fadd_rz(fMax, -fNCB)) < 1e-6) {
			fHue = __fmul_rz((__fadd_rz(__fadd_rz(fNCR, -fNCG) / fDelta, 4.0f)), 60.0f);
		}

		if (fabs(fMax) < 1e-6f) {
			fSaturation = 0.0f;
		}
		else {
			fSaturation = __fdiv_rz(fDelta, fMax);
		}

		fValue = fMax;


		fHue = fHue + nHueV[0];

		if (fHue < 0.0f) {
			fHue = __fadd_rz(360.0f, fHue);
		}
		if (fHue > 360.0f) {
			fHue = __fadd_rz(fHue, -360.0f);
		}


		if (nSaturationV[0] >= 0) {
			fSaturation = __fadd_rz(fSaturation, __fmul_rz(__fadd_rz(1.0f, -fSaturation), __fdiv_rz(nSaturationV[0], 100.0f)));
		}
		else {
			fSaturation = __fadd_rz(fSaturation, __fmul_rz(fSaturation, __fdiv_rz(nSaturationV[0], 100.0f)));
		}

		if (nValueV[0] >= 0) {
			fValue = __fadd_rz(fValue, __fmul_rz(__fadd_rz(1.0f, -fValue), __fdiv_rz(nValueV[0], 100.0f)));
		}
		else {
			fValue = __fadd_rz(fValue, __fmul_rz(fValue, __fdiv_rz(nValueV[0], 100.0f)));
		}


		//HSV To RGB
		float fC, fX, fm;

		fC = __fmul_rz(fValue, fSaturation);
		fX = __fmul_rz(fC, 1.0f - fabs(__fadd_rz(fmodf(__fdiv_rz(fHue, 60), 2.0f), -1.0f)));
		fm = __fadd_rz(fValue, -fC);

		float fTR, fTG, fTB;

		if (fHue < 60.0f && fHue>0.0f) {
			fTR = fC;
			fTG = fX;
			fTB = 0.0;
		}
		else if (fHue < 120.0f && fHue>60.0f) {
			fTR = fX;
			fTG = fC;
			fTB = 0.0;
		}
		else if (fHue < 180.0f && fHue>120.0f) {
			fTR = 0.0f;
			fTG = fC;
			fTB = fX;
		}
		else if (fHue < 240.0f && fHue>180.0f) {
			fTR = 0.0f;
			fTG = fX;
			fTB = fC;
		}
		else if (fHue < 300.0f && fHue>240.0f) {
			fTR = fX;
			fTG = 0.0f;
			fTB = fC;
		}
		else if (fHue < 360.0f && fHue>300.0f) {
			fTR = fC;
			fTG = 0.0f;
			fTB = fX;
		}
		//fTR = 0.0f;

		fCR = __fmul_rz(__fadd_rz(fTR, fm), 255.0f);
		fCG = __fmul_rz(__fadd_rz(fTG, fm), 255.0f);
		fCB = __fmul_rz(__fadd_rz(fTB, fm), 255.0f);
	}




	//pfCY[nOffset] = 0.299f* fCR + 0.587f*fCG + 0.114f* fCB;
	//pfCU[nOffset] = 0.713f * (fCR - pfCY[nOffset]) + 128.0f;
	//pfCV[nOffset] = 0.564f * (fCB - pfCY[nOffset]) + 128.0f;

	//RGB convertTo YUV
	pfCY[nOffset] = __fadd_rz(__fmul_rz(0.299f, fCR), __fadd_rz(__fmul_rz(0.587f, fCG), __fmul_rz(0.114f, fCB)));
	pfCU[nOffset] = __fadd_rz(__fmul_rz(0.713f, __fadd_rz(fCR, -pfCY[nOffset])), 128.0f);
	pfCV[nOffset] = __fadd_rz(__fmul_rz(0.564f, __fadd_rz(fCB, -pfCY[nOffset])), 128.0f);

}

void CCMConvert(float* pfY, float* pfU, float* pfV,
	float* pfCY, float* pfCU, float* pfCV,
	float* pfCCM, int nImageW, int nImageH,
	int nHSVAlter, int nYUVAlter,
	int nHue, int nSaturation, int nValue,
	int nSaturation1) {
	float* pfDeviceY, * pfDeviceU, * pfDeviceV, * pfDeviceCY, * pfDeviceCU, * pfDeviceCV;

	int nImagePtrSz = sizeof(float) * nImageW * nImageH;
	HANDLE_ERROR(cudaMalloc(&pfDeviceY, nImagePtrSz));
	HANDLE_ERROR(cudaMalloc(&pfDeviceU, nImagePtrSz));
	HANDLE_ERROR(cudaMalloc(&pfDeviceV, nImagePtrSz));
	HANDLE_ERROR(cudaMalloc(&pfDeviceCY, nImagePtrSz));
	HANDLE_ERROR(cudaMalloc(&pfDeviceCU, nImagePtrSz));
	HANDLE_ERROR(cudaMalloc(&pfDeviceCV, nImagePtrSz));

	HANDLE_ERROR(cudaMemcpy(pfDeviceY, pfY, nImagePtrSz, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceU, pfU, nImagePtrSz, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceV, pfV, nImagePtrSz, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(fCCM, pfCCM, sizeof(float) * cnCCMW * cnCCMH));
	HANDLE_ERROR(cudaMemcpyToSymbol(nHSVAlterTag, &nHSVAlter, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(nYUVAlterTag, &nYUVAlter, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(nHueV, &nHue, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(nSaturationV, &nSaturation, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(nValueV, &nValue, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(nSaturation1V, &nSaturation1, sizeof(int)));
	//HANDLE_ERROR(cudaMemcpyToSymbol(nSaturation1BackupV, &nSaturation1, sizeof(int)));


	dim3 d3Block(nImageW / 16, nImageH / 16), d3Thread(16, 16);
	CCM << <d3Block, d3Thread >> > (pfDeviceY, pfDeviceU, pfDeviceV, pfDeviceCY, pfDeviceCU, pfDeviceCV);


	//nBackupS = nSaturation1;
	HANDLE_ERROR(cudaMemcpy(pfCY, pfDeviceCY, nImagePtrSz, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfCU, pfDeviceCU, nImagePtrSz, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfCV, pfDeviceCV, nImagePtrSz, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceY));
	HANDLE_ERROR(cudaFree(pfDeviceU));
	HANDLE_ERROR(cudaFree(pfDeviceV));
	HANDLE_ERROR(cudaFree(pfDeviceCY));
	HANDLE_ERROR(cudaFree(pfDeviceCU));
	HANDLE_ERROR(cudaFree(pfDeviceCV));
}


void FigureBrightnessPixelSz(const cv::Mat& matResizeImage, long* plBPSz, float fThresh) {
	*plBPSz = 0;

	const float* pfRzImage;
	int nStartC = static_cast<int>(0.2 * matResizeImage.cols + 0.5);
	int nEndC = static_cast<int>(0.8 * matResizeImage.cols + 0.5);

	int r, c;
	for (r = 0; r < matResizeImage.rows; r++) {
		pfRzImage = matResizeImage.ptr<const float>(r);
		for (c = nStartC; c < nEndC; c++) {
			if (pfRzImage[c] > fThresh) {
				(*plBPSz)++;
			}
		}
	}
}

void FigureDarknessPixelSz(const cv::Mat& matResizeImage, long* plDPSz) {
	*plDPSz = 0;

	const float* pfRzImage;
	int nStartC = static_cast<int>(0.2 * matResizeImage.cols + 0.5);
	int nEndC = static_cast<int>(0.8 * matResizeImage.cols + 0.5);

	int r, c;
	for (r = 0; r < matResizeImage.rows; r++) {
		pfRzImage = matResizeImage.ptr<const float>(r);
		for (c = nStartC; c < nEndC; c++) {
			if (pfRzImage[c] < 3.0f) {
				(*plDPSz)++;
			}
		}
	}
}


__global__ void cRGB2YUV(float* pfR, float* pfG, float* pfB,
	float* pfY, float* pfU, float* pfV) {

	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	pfY[nOffset] = 16.0f + 0.257f * pfR[nOffset] + 0.504f * pfG[nOffset] + 0.098f * pfB[nOffset];
	pfU[nOffset] = 128.0f - 0.148f * pfR[nOffset] - 0.291f * pfG[nOffset] + 0.439f * pfB[nOffset];
	pfV[nOffset] = 128.0f + 0.439f * pfR[nOffset] - 0.368f * pfG[nOffset] - 0.071f * pfB[nOffset];
}

__global__ void cYUV2RGB(float* pfY, float* pfU, float* pfV,
	float* pfR, float* pfG, float* pfB) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	pfR[nOffset] = 1.164f * (pfY[nOffset] - 16.0f) + 1.596f * (pfV[nOffset] - 128.0f);
	pfG[nOffset] = 1.164f * (pfY[nOffset] - 16.0f) - 0.392f * (pfU[nOffset] - 128.0f) - 0.812f * (pfV[nOffset] - 128.0f);
	pfB[nOffset] = 1.164f * (pfY[nOffset] - 16.0f) + 2.016f * (pfU[nOffset] - 128.0f);
}

void convertYUV2RGB(float* pfYChannel, float* pfUChannel, float* pfVChannel,
	float* pfRChannel, float* pfGChannel, float* pfBChannel,
	int nWidth, int nHeight) {
	float* pfDeviceYChannel, * pfDeviceUChannel, * pfDeviceVChannel;
	float* pfDeviceRChannel, * pfDeviceGChannel, * pfDeviceBChannel;

	HANDLE_ERROR(cudaMalloc(&pfDeviceYChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceUChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceVChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceBChannel, sizeof(float) * nWidth * nHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceYChannel, pfYChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceUChannel, pfUChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceVChannel, pfVChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nWidth / 16, nHeight / 16);
	cYUV2RGB << <d3Block, d3Thread >> > (pfDeviceYChannel, pfDeviceUChannel, pfDeviceVChannel, pfDeviceRChannel, pfDeviceGChannel, pfDeviceBChannel);

	HANDLE_ERROR(cudaMemcpy(pfRChannel, pfDeviceRChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfGChannel, pfDeviceGChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfBChannel, pfDeviceBChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceBChannel));
	HANDLE_ERROR(cudaFree(pfDeviceGChannel));
	HANDLE_ERROR(cudaFree(pfDeviceRChannel));
	HANDLE_ERROR(cudaFree(pfDeviceVChannel));
	HANDLE_ERROR(cudaFree(pfDeviceUChannel));
	HANDLE_ERROR(cudaFree(pfDeviceYChannel));

}

void convertRGB2YUV(float* pfRChannel, float* pfGChannel, float* pfBChannel,
	float* pfYChannel, float* pfUChannel, float* pfVChannel,
	int nWidth, int nHeight) {
	float* pfDeviceYChannel, * pfDeviceUChannel, * pfDeviceVChannel;
	float* pfDeviceRChannel, * pfDeviceGChannel, * pfDeviceBChannel;

	HANDLE_ERROR(cudaMalloc(&pfDeviceYChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceUChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceVChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceBChannel, sizeof(float) * nWidth * nHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceRChannel, pfRChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceGChannel, pfGChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceBChannel, pfBChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nWidth / 16, nHeight / 16);
	cRGB2YUV << <d3Block, d3Thread >> > (pfDeviceRChannel, pfDeviceGChannel, pfDeviceBChannel, pfDeviceYChannel, pfDeviceUChannel, pfDeviceVChannel);

	HANDLE_ERROR(cudaMemcpy(pfYChannel, pfDeviceYChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfUChannel, pfDeviceUChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfVChannel, pfDeviceVChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceBChannel));
	HANDLE_ERROR(cudaFree(pfDeviceGChannel));
	HANDLE_ERROR(cudaFree(pfDeviceRChannel));
	HANDLE_ERROR(cudaFree(pfDeviceVChannel));
	HANDLE_ERROR(cudaFree(pfDeviceUChannel));
	HANDLE_ERROR(cudaFree(pfDeviceYChannel));
}


__global__ void CCMMultiple(float* pfRChannel, float* pfGChannel, float* pfBChannel) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	pfRChannel[nOffset] = __fadd_rz(__fadd_rz(fCCM[3], __fmul_rz(pfRChannel[nOffset], fCCM[0])), __fadd_rz(__fmul_rz(pfGChannel[nOffset], fCCM[1]), __fmul_rz(pfBChannel[nOffset], fCCM[2])));
	pfGChannel[nOffset] = __fadd_rz(__fadd_rz(fCCM[7], __fmul_rz(pfRChannel[nOffset], fCCM[4])), __fadd_rz(__fmul_rz(pfGChannel[nOffset], fCCM[5]), __fmul_rz(pfBChannel[nOffset], fCCM[6])));
	pfBChannel[nOffset] = __fadd_rz(__fadd_rz(fCCM[11], __fmul_rz(pfRChannel[nOffset], fCCM[8])), __fadd_rz(__fmul_rz(pfGChannel[nOffset], fCCM[9]), __fmul_rz(pfBChannel[nOffset], fCCM[10])));
}



void ColorCorrectionM(float* pfRChannel, float* pfGChannel, float* pfBChannel, float* pfColorCorrectionM , int nwidth, int nheight) {
	float* pfDeviceRChannel, * pfDeviceGChannel, * pfDeviceBChannel;

	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannel, sizeof(float)* nwidth * nheight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannel, sizeof(float) * nwidth * nheight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceBChannel, sizeof(float) * nwidth * nheight));
	
	HANDLE_ERROR(cudaMemcpyToSymbol(fCCM, pfColorCorrectionM, sizeof(float)*12));
	HANDLE_ERROR(cudaMemcpy(pfDeviceRChannel, pfRChannel, sizeof(float) * nwidth * nheight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceGChannel, pfGChannel, sizeof(float) * nwidth * nheight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceBChannel, pfBChannel, sizeof(float) * nwidth * nheight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nwidth / 16, nheight / 16);
	CCMMultiple << <d3Block, d3Thread >> > (pfDeviceRChannel, pfDeviceGChannel, pfDeviceBChannel);

	HANDLE_ERROR(cudaMemcpy( pfRChannel, pfDeviceRChannel, sizeof(float) * nwidth * nheight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy( pfGChannel, pfDeviceGChannel, sizeof(float) * nwidth * nheight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy( pfBChannel, pfDeviceBChannel, sizeof(float) * nwidth * nheight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceRChannel));
	HANDLE_ERROR(cudaFree(pfDeviceGChannel));
	HANDLE_ERROR(cudaFree(pfDeviceBChannel));
}


__global__ void CRGB2HSV(float* pfRChannel, float* pfGChannel, float* pfBChannel, float* pfHChannel, float* pfSChannel, float* pfVChannel) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;


	float fNCR = __fdiv_rz(pfRChannel[nOffset], 255.0f);
	float fNCG = __fdiv_rz(pfGChannel[nOffset], 255.0f);
	float fNCB = __fdiv_rz(pfBChannel[nOffset], 255.0f);

	float fMax, fMin;
	if (fNCR > fNCG && fNCR > fNCB) {
		fMax = fNCR;
		if (fNCG < fNCB) {
			fMin = fNCG;
		}
		else {
			fMin = fNCB;
		}
	}
	if (fNCG > fNCR && fNCG > fNCB) {
		fMax = fNCG;
		if (fNCR < fNCB) {
			fMin = fNCR;
		}
		else {
			fMin = fNCB;
		}
	}
	if (fNCB > fNCR && fNCB > fNCG) {
		fMax = fNCB;
		if (fNCR < fNCG) {
			fMin = fNCR;
		}
		else {
			fMin = fNCG;
		}
	}
	float fDelta = __fadd_rz(fMax, -fMin);

	float fHue, fSaturation, fValue;

	if (fabs(fDelta) < 1e-6f) {
		fHue = 0.0f;
	}
	else if (fabs(__fadd_rz(fMax, -fNCR)) < 1e-6 && fNCG >= fNCB) {
		fHue = __fmul_rz(__fadd_rz(fNCG, -fNCB) / fDelta, 60.0f);
	}
	else if (fabs(__fadd_rz(fMax, -fNCR)) < 1e-6 && fNCG < fNCB) {
		fHue = __fmul_rz(__fadd_rz(fNCG, -fNCB) / fDelta, 60.0f) + 360.0f;
	}
	else if (fabs(__fadd_rz(fMax, -fNCG)) < 1e-6) {
		fHue = __fmul_rz((__fadd_rz(__fadd_rz(fNCB, -fNCR) / fDelta, 2.0f)), 60.0f);
	}
	else if (fabs(__fadd_rz(fMax, -fNCB)) < 1e-6) {
		fHue = __fmul_rz((__fadd_rz(__fadd_rz(fNCR, -fNCG) / fDelta, 4.0f)), 60.0f);
	}

	if (fabs(fMax) < 1e-6f) {
		fSaturation = 0.0f;
	}
	else {
		fSaturation = __fdiv_rz(fDelta, fMax);
	}

	fValue = fMax;

	pfHChannel[nOffset] = fHue;
	pfSChannel[nOffset] = fSaturation;
	pfVChannel[nOffset] = fValue;


}

__global__ void CHSV2RGB(float* pfHChannel, float* pfSChannel, float* pfVChannel, float* pfRChannel, float* pfGChannel, float* pfBChannel) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;


	float fC, fX, fm;

	fC = __fmul_rz(pfVChannel[nOffset], pfSChannel[nOffset]);
	fX = __fmul_rz(fC, 1.0f - fabs(__fadd_rz(fmodf(__fdiv_rz(pfHChannel[nOffset], 60), 2.0f), -1.0f)));
	fm = __fadd_rz(pfVChannel[nOffset], -fC);

	float fTR, fTG, fTB;

	if (pfHChannel[nOffset] < 60.0f && pfHChannel[nOffset]>0.0f) {
		fTR = fC;
		fTG = fX;
		fTB = 0.0;
	}
	else if (pfHChannel[nOffset] < 120.0f && pfHChannel[nOffset]>60.0f) {
		fTR = fX;
		fTG = fC;
		fTB = 0.0;
	}
	else if (pfHChannel[nOffset] < 180.0f && pfHChannel[nOffset]>120.0f) {
		fTR = 0.0f;
		fTG = fC;
		fTB = fX;
	}
	else if (pfHChannel[nOffset] < 240.0f && pfHChannel[nOffset]>180.0f) {
		fTR = 0.0f;
		fTG = fX;
		fTB = fC;
	}
	else if (pfHChannel[nOffset] < 300.0f && pfHChannel[nOffset]>240.0f) {
		fTR = fX;
		fTG = 0.0f;
		fTB = fC;
	}
	else if (pfHChannel[nOffset] < 360.0f && pfHChannel[nOffset]>300.0f) {
		fTR = fC;
		fTG = 0.0f;
		fTB = fX;
	}

	pfRChannel[nOffset] = __fmul_rz(__fadd_rz(fTR, fm), 255.0f);
	pfGChannel[nOffset] = __fmul_rz(__fadd_rz(fTG, fm), 255.0f);
	pfBChannel[nOffset] = __fmul_rz(__fadd_rz(fTB, fm), 255.0f);



}

void convertRGB2HSV(float* pfRChannel, float* pfGChannel, float* pfBChannel,
	float* pfHChannel, float* pfSChannel, float* pfVChannel, int nWidth, int nHeight) {
	float* pfDeviceRChannel, * pfDeviceGChannel, * pfDeviceBChannel,
			* pfDeviceHChannel, *pfDeviceSChannel, *pfDeviceVChannel;

	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannel, sizeof(float) * nWidth* nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceBChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceHChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceSChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceVChannel, sizeof(float) * nWidth * nHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceRChannel, pfRChannel, sizeof(float)* nWidth* nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceGChannel, pfGChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceBChannel, pfBChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nWidth / 16, nHeight / 16);
	CRGB2HSV << <d3Block, d3Thread >> > (pfDeviceRChannel, pfDeviceGChannel, pfDeviceBChannel, pfDeviceHChannel, pfDeviceSChannel, pfDeviceVChannel);

	HANDLE_ERROR(cudaMemcpy(pfHChannel, pfDeviceHChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfSChannel, pfDeviceSChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfVChannel, pfDeviceVChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceRChannel));
	HANDLE_ERROR(cudaFree(pfDeviceGChannel));
	HANDLE_ERROR(cudaFree(pfDeviceBChannel));
	HANDLE_ERROR(cudaFree(pfDeviceHChannel));
	HANDLE_ERROR(cudaFree(pfDeviceSChannel));
	HANDLE_ERROR(cudaFree(pfDeviceVChannel));
}


void convertHSV2RGB(float* pfHChannel, float* pfSChannel, float* pfVChannel,
	float* pfRChannel, float* pfGChannel, float* pfBChannel, int nWidth, int nHeight) {

	float* pfDeviceRChannel, * pfDeviceGChannel, * pfDeviceBChannel,
		* pfDeviceHChannel, * pfDeviceSChannel, * pfDeviceVChannel;

	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceBChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceHChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceSChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceVChannel, sizeof(float) * nWidth * nHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceHChannel, pfHChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceSChannel, pfSChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceVChannel, pfVChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nWidth / 16, nHeight / 16);
	CHSV2RGB << <d3Block, d3Thread >> > (pfDeviceHChannel, pfDeviceSChannel, pfDeviceVChannel,pfDeviceRChannel, pfDeviceGChannel, pfDeviceBChannel);

	HANDLE_ERROR(cudaMemcpy(pfRChannel, pfDeviceRChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfGChannel, pfDeviceGChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfBChannel, pfDeviceBChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceRChannel));
	HANDLE_ERROR(cudaFree(pfDeviceGChannel));
	HANDLE_ERROR(cudaFree(pfDeviceBChannel));
	HANDLE_ERROR(cudaFree(pfDeviceHChannel));
	HANDLE_ERROR(cudaFree(pfDeviceSChannel));
	HANDLE_ERROR(cudaFree(pfDeviceVChannel));
}

__global__ void enHSV(float* pfHChannel, float* pfSChannel, float* pfVChannel) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;
	
	pfHChannel[nOffset] = pfHChannel[nOffset] + nHueV[0];

	if (pfHChannel[nOffset] < 0.0f) {
		pfHChannel[nOffset] = __fadd_rz(360.0f, pfHChannel[nOffset]);
	}
	if (pfHChannel[nOffset] > 360.0f) {
		pfHChannel[nOffset] = __fadd_rz(pfHChannel[nOffset], -360.0f);
	}


	if (nSaturationV[0] >= 0) {
		pfSChannel[nOffset] = __fadd_rz(pfSChannel[nOffset], __fmul_rz(__fadd_rz(1.0f, -pfSChannel[nOffset]), __fdiv_rz(nSaturationV[0], 100.0f)));
		//pfSChannel[nOffset] = pfSChannel[nOffset] + (1.0f - pfSChannel[nOffset]) *(nSaturationV[0]/100.0f) ;
	}
	else {
		pfSChannel[nOffset] = __fadd_rz(pfSChannel[nOffset], __fmul_rz(pfSChannel[nOffset], __fdiv_rz(nSaturationV[0], 100.0f)));
		//pfSChannel[nOffset] = pfSChannel[nOffset] + pfSChannel[nOffset]* (nSaturationV[0]/100.0f);
	}

	if (nValueV[0] >= 0) {
		pfVChannel[nOffset] = __fadd_rz(pfVChannel[nOffset], __fmul_rz(__fadd_rz(1.0f, -pfVChannel[nOffset]), __fdiv_rz(nValueV[0], 100.0f)));
	}
	else {
		pfVChannel[nOffset] = __fadd_rz(pfVChannel[nOffset], __fmul_rz(pfVChannel[nOffset], __fdiv_rz(nValueV[0], 100.0f)));
	}
}

void EnhanceHSV(float* pfHChannel, float* pfSChannel, float* pfVChannel, 
							  int nWidth, int nHeight, int nHue, int nSaturation, int nValue){
	float* pfDeviceHChannel, * pfDeviceSChannel, * pfDeviceVChannel;

	HANDLE_ERROR(cudaMalloc(&pfDeviceHChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceSChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceVChannel, sizeof(float) * nWidth * nHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceHChannel, pfHChannel, sizeof(float)* nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceSChannel, pfSChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceVChannel, pfVChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpyToSymbol(nHueV, &nHue, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(nSaturationV, &nSaturation, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(nValueV, &nValue, sizeof(int)));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nWidth / 16, nHeight / 16);
	enHSV << <d3Block, d3Thread >> > (pfDeviceHChannel, pfDeviceSChannel, pfDeviceVChannel);

	HANDLE_ERROR(cudaMemcpy(pfHChannel, pfDeviceHChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfSChannel, pfDeviceSChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfVChannel, pfDeviceVChannel,  sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	
	HANDLE_ERROR(cudaFree(pfDeviceHChannel));
	HANDLE_ERROR(cudaFree(pfDeviceSChannel));
	HANDLE_ERROR(cudaFree(pfDeviceVChannel));

}


__global__ void FigureIHbV(float* pfRChannels, float* pfGChannels, float* pfIHb) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	if (pfGChannels[nOffset] < 0.1f)
		pfGChannels[nOffset] = 0.1f;
	if (pfRChannels[nOffset] < 0.001f)
		pfRChannels[nOffset] = 0.001f;

	pfIHb[nOffset] = 32 * log2(pfRChannels[nOffset] / pfGChannels[nOffset]);

}

void GenIHbImage(float* pfRChannels, float* pfGChannels, float* pfIHbImage,
	int nImageWidth, int nImageHeight) {
	float* pfDeviceRChannels, * pfDeviceGChannels, * pfDeviceIHbImage;

	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceIHbImage, sizeof(float) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceRChannels, pfRChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceGChannels, pfGChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);
	FigureIHbV << <d3Block, d3Thread >> > (pfDeviceRChannels, pfDeviceGChannels, pfDeviceIHbImage);
	HANDLE_ERROR(cudaMemcpy(pfIHbImage, pfDeviceIHbImage, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));


	HANDLE_ERROR(cudaFree(pfDeviceRChannels));
	HANDLE_ERROR(cudaFree(pfDeviceGChannels));
	HANDLE_ERROR(cudaFree(pfDeviceIHbImage));
}




float MeanImage1(cv::Mat matInputImage) {
	float fSum = 0.0f;
	int r, c;
	float* pfmatInputImage;
	for (r = 0; r < matInputImage.rows; r++) {
		pfmatInputImage = matInputImage.ptr<float>(r);
		for (c = 0; c < matInputImage.cols; c++) {
			fSum += pfmatInputImage[c];
		}
	}

	fSum /= matInputImage.cols * matInputImage.rows;
	return fSum;
}

int MeanImage2(cv::Mat matInputImage) {
	int n;
	float fSum = 0.0f;
	int r, c;
	int* pnmatInputImage;
	for (r = 0; r < matInputImage.rows; r++) {
		pnmatInputImage = matInputImage.ptr<int>(r);
		for (c = 0; c < matInputImage.cols; c++) {
			fSum += pnmatInputImage[c];
		}
	}

	fSum /= matInputImage.cols * matInputImage.rows;
	return static_cast<int>(fSum);
}

__global__ void AddRatio(float* pfIHbImage, float* pfEnhaceIHbImage, float fK) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	//if (pfIHbImage[nOffset] < 0.01f)
	//	pfIHbImage[nOffset] = 0.01f;

	pfEnhaceIHbImage[nOffset] = 0.01f * fK * pfIHbImage[nOffset] * pfIHbImage[nOffset];
}

__global__ void AddRatio1(float* pfIHbImage, float* pfEnhaceIHbImage, float fK) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	if (fabs(pfIHbImage[nOffset]) < 0.01f)
		pfIHbImage[nOffset] = 0.01f;

	float fCK = exp(-fabs(pfIHbImage[nOffset] - fMeanIHb[0]) / (4.0* fMeanIHb[0])) * fK;
	//float fCY = exp(-(fMeanY[0] - 127.5f) / 127.5f);
	pfEnhaceIHbImage[nOffset] = (pfIHbImage[nOffset] - fMeanIHb[0]) * fCK /** fCY*/ + fMeanIHb[0];
}

__global__ void AddRatio3(float* pfIHbImage, float* pfYImage, float* pfEnhaceIHbImage, float fK) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	if (fabs(pfIHbImage[nOffset]) < 0.01f)
		pfIHbImage[nOffset] = 0.01f;

	float fCK = exp(-fabs(pfIHbImage[nOffset] - fMeanIHb[0]) / (4.0f * fMeanIHb[0])) * fK;
	//float fCY = exp(-(fMeanY[0] - 127.5f) / 127.5f);
	float fCY = exp(-(pfYImage[nOffset] - fMeanY[0]) / (4.0f * fMeanY[0]));
	pfEnhaceIHbImage[nOffset] = (pfIHbImage[nOffset] - fMeanIHb[0]) * fCK * fCY + fMeanIHb[0];
}

__global__ void AddRatio3(int* pnIHbImage, int* pnYImagePtr, int* pnEnhaceIHbImage, int nK) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	if (pnIHbImage[nOffset] < 1)
		pnIHbImage[nOffset] = 0;

	//if (fMeanIHb[0] > 20.0f)
	//	fMeanIHb[0] = 0.8 * fMeanIHb[0];


	//float fCK = exp(-abs(pnIHbImage[nOffset] - nMeanIHb[0]) / (4.0f * nMeanIHb[0])) ;

	//float fCK = 1.0f - abs(pnIHbImage[nOffset] - nMeanIHb[0]) / (4.0f * nMeanIHb[0]);
	//if (fCK < 0.0f) {
	//	fCK = 0.01;
	//}
	//int nCK = (int)(fCK + 0.5f) << 7;

	//int nTempV = abs(pnIHbImage[nOffset] - nMeanIHb[0]) << 7;
	//int nCK = 128 - nTempV / (8 * nMeanIHb[0]);

	//if (nCK < 0) {
	//	nCK = 1;
	//}
	//int nTempY = abs(pnYImagePtr[nOffset] - nMeanY[0]) << 7;
	//int nCY = 128 - nTempV / (8 * nMeanY[0]);
	//
	//if (nCY < 0) {
	//	nCY = 1;
	//}

	//if (fCK < 0.0f) {
	//	fCK = 0.01f;
	//}
	//int nCK = (int)(fCK + 0.5f) << 7;

	//float fCY = exp(-abs(nMeanY[0] - 127.5f) / 127.5f);
	//int nCY = (int)(fCY+0.5f) << 7;

	long long  dlTestV = nMeanIHb[0] << 7;

	long long dlTempV = (long long)(pnIHbImage[nOffset] - nMeanIHb[0]) * /*(long long)nCK * (long long)nCY**/ (long long)(nK)+dlTestV;

	pnEnhaceIHbImage[nOffset] = dlTempV >> 7;
}

__global__ void AddRatio2(float* pfIHbImage, float* pfEnhaceIHbImage, float fK, int nImageW, int nImageH) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nBlockWidth = nBlockDimX + __mul24(nKernelRSz5, 2);
	const int nBlockHeight = nBlockDimY + __mul24(nKernelRSz5, 2);
	const int nShift = __mul24(threadIdx.y, nBlockWidth);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ float fConvData[(nBlockDimY + 2 * nKernelRSz5) * (nBlockDimX + 2 * nKernelRSz5)];

	if (threadIdx.x < nKernelRSz5 || threadIdx.x >= nBlockDimX - nKernelRSz5 ||
		threadIdx.y < nKernelRSz5 || threadIdx.y >= nBlockDimY - nKernelRSz5) {
		if ((nX - nKernelRSz5) < 0 || (nY - nKernelRSz5) < 0) {
			fConvData[threadIdx.x + nShift] = pfIHbImage[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift] = pfIHbImage[nOffset - __mul24(nKernelRSz5, nImageW) - nKernelRSz5];
		}

		if ((nX + nKernelRSz5) > nImageW - 1 || (nY - nKernelRSz5) < 0) {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift] = pfIHbImage[nOffset];
		}
		else {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift] = pfIHbImage[nOffset - __mul24(nKernelRSz5, nImageW) + nKernelRSz5];
		}

		if ((nX - nKernelRSz5) < 0 || (nY + nKernelRSz5) > nImageH - 1) {
			fConvData[threadIdx.x + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pfIHbImage[nOffset];
		}
		else {
			fConvData[threadIdx.x + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pfIHbImage[nOffset + __mul24(nKernelRSz5, nImageW) - nKernelRSz5];
		}

		if ((nX + nKernelRSz5) > nImageW - 1 || (nY + nKernelRSz5) > nImageH - 1) {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pfIHbImage[nOffset];
		}
		else {
			fConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pfIHbImage[nOffset + __mul24(nKernelRSz5, nImageW) + nKernelRSz5];
		}
		fConvData[threadIdx.x + nKernelRSz5 + nShift + __mul24(nKernelRSz5, nBlockWidth)] = pfIHbImage[nOffset];
	}
	else {
		fConvData[threadIdx.x + nKernelRSz5 + nShift + __mul24(nKernelRSz5, nBlockWidth)] = pfIHbImage[nOffset];
	}

	__syncthreads();

	int r, c;
	float fSumV = 0.0;
	for (r = 0; r < 5; r++) {
		for (c = 0; c < 5; c++) {
			fSumV += fConvData[(threadIdx.y + r) * nBlockWidth + (threadIdx.x + c)];
		}
	}
	float fMeanV = fSumV / 25.0f;

	if (fabs(pfIHbImage[nOffset]) < 0.01f)
		pfIHbImage[nOffset] = 0.01f;

	float fCK = exp(-fabs(pfIHbImage[nOffset] - fMeanV) / (4.0f * fMeanV)) * fK;
	//float fCY = exp(-(fMeanY[0] - 127.5f) / 127.5f);
	pfEnhaceIHbImage[nOffset] = (pfIHbImage[nOffset] - fMeanV) * fCK /** fCY*/ + fMeanV;
}

void EnhanceIHbImageInt1(int* pnIHbImagePtr, int* pnYImagePtr, int nAverageIHb, int nAverageY, int* pnEnhanceIHbImagePtr, int nK, int nImageWidth, int nImageHeight) {
	int* pnDeviceIHbImage, * pnDeviceEnhanceIHbImgePtr, * pnDeviceYImagePtr;

	HANDLE_ERROR(cudaMalloc(&pnDeviceIHbImage, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceEnhanceIHbImgePtr, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceYImagePtr, sizeof(int) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pnDeviceIHbImage, pnIHbImagePtr, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pnDeviceYImagePtr, pnYImagePtr, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpyToSymbol(nMeanIHb, &nAverageIHb, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(nMeanY, &nAverageY, sizeof(int)));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	AddRatio3 << <d3Block, d3Thread >> > (pnDeviceIHbImage, pnDeviceYImagePtr, pnDeviceEnhanceIHbImgePtr, nK);

	HANDLE_ERROR(cudaMemcpy(pnEnhanceIHbImagePtr, pnDeviceEnhanceIHbImgePtr, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(pnDeviceIHbImage));
	HANDLE_ERROR(cudaFree(pnDeviceEnhanceIHbImgePtr));
	HANDLE_ERROR(cudaFree(pnDeviceYImagePtr));
}

__global__ void FigureY(int* pnR, int* pnG, int* pnB, int* pnY) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	pnY[nOffset] = 218 * pnR[nOffset] + 732 * pnG[nOffset] + 74 * pnB[nOffset];
	pnY[nOffset] = pnY[nOffset] >> 10;
}

void GenYImageInt(int* pnRChannel, int* pnGChannel, int* pnBChannel,
	int* pnYChannel, int nImageWidth, int nImageHeight) {
	int* pnDeviceRChannel, * pnDeviceGChannel, * pnDeviceBChannel, * pnDeviceYChannel;

	HANDLE_ERROR(cudaMalloc(&pnDeviceRChannel, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceGChannel, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceBChannel, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceYChannel, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMemcpy(pnDeviceRChannel, pnRChannel, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pnDeviceGChannel, pnGChannel, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pnDeviceBChannel, pnBChannel, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);
	FigureY << <d3Block, d3Thread >> > (pnDeviceRChannel, pnDeviceGChannel, pnDeviceBChannel, pnDeviceYChannel);

	HANDLE_ERROR(cudaMemcpy(pnYChannel, pnDeviceYChannel, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(pnDeviceRChannel));
	HANDLE_ERROR(cudaFree(pnDeviceGChannel));
	HANDLE_ERROR(cudaFree(pnDeviceBChannel));
	HANDLE_ERROR(cudaFree(pnDeviceYChannel));
}

void EnhanceIHbImage(float* pfIHbImagePtr, float fAverageIHb, float fAverageY, float* pfEnhanceIHbImagePtr, float fK, int nImageWidth, int nImageHeight) {
	float* pfDeviceIHbImage, * pfDeviceEnhanceIHbImgePtr;

	HANDLE_ERROR(cudaMalloc(&pfDeviceIHbImage, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceEnhanceIHbImgePtr, sizeof(float) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceIHbImage, pfIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpyToSymbol(fMeanIHb, &fAverageIHb, sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(fMeanY, &fAverageY, sizeof(float)));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	AddRatio << <d3Block, d3Thread >> > (pfDeviceIHbImage, pfDeviceEnhanceIHbImgePtr, fK);

	HANDLE_ERROR(cudaMemcpy(pfEnhanceIHbImagePtr, pfDeviceEnhanceIHbImgePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(pfDeviceIHbImage));
	HANDLE_ERROR(cudaFree(pfDeviceEnhanceIHbImgePtr));
}

void EnhanceIHbImage1(float* pfIHbImagePtr, float fAverageIHb, float fAverageY, float* pfEnhanceIHbImagePtr, float fK, int nImageWidth, int nImageHeight) {
	float* pfDeviceIHbImage, * pfDeviceEnhanceIHbImgePtr;

	HANDLE_ERROR(cudaMalloc(&pfDeviceIHbImage, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceEnhanceIHbImgePtr, sizeof(float) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceIHbImage, pfIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpyToSymbol(fMeanIHb, &fAverageIHb, sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(fMeanY, &fAverageY, sizeof(float)));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	AddRatio1 << <d3Block, d3Thread >> > (pfDeviceIHbImage, pfDeviceEnhanceIHbImgePtr, fK);

	HANDLE_ERROR(cudaMemcpy(pfEnhanceIHbImagePtr, pfDeviceEnhanceIHbImgePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(pfDeviceIHbImage));
	HANDLE_ERROR(cudaFree(pfDeviceEnhanceIHbImgePtr));
}


void EnhanceIHbImage2(float* pfIHbImagePtr, float* pfEnhanceIHbImagePtr, float fK, int nImageWidth, int nImageHeight) {
	float* pfDeviceIHbImage, * pfDeviceEnhanceIHbImgePtr;

	HANDLE_ERROR(cudaMalloc(&pfDeviceIHbImage, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceEnhanceIHbImgePtr, sizeof(float) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceIHbImage, pfIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	//HANDLE_ERROR(cudaMemcpyToSymbol(fMeanIHb, &fAverageIHb, sizeof(float)));
	//HANDLE_ERROR(cudaMemcpyToSymbol(fMeanY, &fAverageY, sizeof(float)));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	AddRatio2 << <d3Block, d3Thread >> > (pfDeviceIHbImage, pfDeviceEnhanceIHbImgePtr, fK, nImageWidth, nImageHeight);

	HANDLE_ERROR(cudaMemcpy(pfEnhanceIHbImagePtr, pfDeviceEnhanceIHbImgePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(pfDeviceIHbImage));
	HANDLE_ERROR(cudaFree(pfDeviceEnhanceIHbImgePtr));
}

void EnhanceIHbImage3(float* pfIHbImagePtr, float* pfYImagePtr, float fAverageIHb, float fAverageY, float* pfEnhanceIHbImagePtr, float fK, int nImageWidth, int nImageHeight) {
	float* pfDeviceIHbImage, * pfDeviceEnhanceIHbImgePtr, *pfDeviceYImagePtr;

	HANDLE_ERROR(cudaMalloc(&pfDeviceIHbImage, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceEnhanceIHbImgePtr, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceYImagePtr, sizeof(float) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceIHbImage, pfIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceYImagePtr, pfYImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpyToSymbol(fMeanIHb, &fAverageIHb, sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(fMeanY, &fAverageY, sizeof(float)));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	AddRatio3 << <d3Block, d3Thread >> > (pfDeviceIHbImage, pfDeviceYImagePtr, pfDeviceEnhanceIHbImgePtr, fK);

	HANDLE_ERROR(cudaMemcpy(pfEnhanceIHbImagePtr, pfDeviceEnhanceIHbImgePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(pfDeviceIHbImage));
	HANDLE_ERROR(cudaFree(pfDeviceEnhanceIHbImgePtr));
	HANDLE_ERROR(cudaFree(pfDeviceYImagePtr));
}

__global__ void FigureIHbVInt(int* pnRChannels, int* pnGChannels, int* pnIHb) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	if (pnRChannels[nOffset] < 0.1f)
		pnRChannels[nOffset] = 1;

	int nMG = pnGChannels[nOffset] << 10;
	int nR, nG;
	if (pnGChannels[nOffset] < 1e-6f)
		pnIHb[nOffset] = 0;
	else {
		//nR = log2((double)(pnRChannels[nOffset])) * 1024.0;
		//nG = log2((double)(pnGChannels[nOffset])) * 1024.0;

		//pnIHb[nOffset] = 32* (nR - nG);

		pnIHb[nOffset] = nMG / pnRChannels[nOffset];
	}
}

void GenIHbImageInt(int* pnRChannels, int* pnGChannels, int* pnIHbImage,
	int nImageWidth, int nImageHeight) {
	int* pnDeviceRChannels, * pnDeviceGChannels, * pnDeviceIHbImage;

	HANDLE_ERROR(cudaMalloc(&pnDeviceRChannels, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceGChannels, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceIHbImage, sizeof(int) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pnDeviceRChannels, pnRChannels, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pnDeviceGChannels, pnGChannels, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);
	FigureIHbVInt << <d3Block, d3Thread >> > (pnDeviceRChannels, pnDeviceGChannels, pnDeviceIHbImage);
	HANDLE_ERROR(cudaMemcpy(pnIHbImage, pnDeviceIHbImage, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));


	HANDLE_ERROR(cudaFree(pnDeviceRChannels));
	HANDLE_ERROR(cudaFree(pnDeviceGChannels));
	HANDLE_ERROR(cudaFree(pnDeviceIHbImage));
}

__global__ void ConvertRGImage(float* pfRChannels, float* pfGChannels, float* pfIHbImagePtr) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	pfGChannels[nOffset] = pfRChannels[nOffset] / pow(2, pfIHbImagePtr[nOffset] / 32.0f);
	//pfRChannels[nOffset] = pfGChannels[nOffset] * pow(2, pfIHbImagePtr[nOffset] / 32.0f);
}

__global__ void ConvertRGImage(int* pnRChannels, int* pnGChannels, int* pnIHbImagePtr) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	//pnGChannels[nOffset] = pnRChannels[nOffset] / pow(2, pnIHbImagePtr[nOffset] / 32.0f);
	//int nTempIHb = pnIHbImagePtr[nOffset] >> 15;

	//int nM = 2 << nTempIHb;

	pnGChannels[nOffset] = (pnRChannels[nOffset] * pnIHbImagePtr[nOffset]) >> 10;
	//pfRChannels[nOffset] = pfGChannels[nOffset] * pow(2, pfIHbImagePtr[nOffset] / 32.0f);
}


void RecoverImage(int* pfRChannels, int* pfGChannels, int* pfIHbImagePtr, int nImageWidth, int nImageHeight) {
	int* pnDeviceRChannels, * pnDeviceGChannels, * pnDeviceIHbImagePtr;

	HANDLE_ERROR(cudaMalloc(&pnDeviceRChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceGChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pnDeviceRChannels, pfRChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pnDeviceGChannels, pfGChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pnDeviceIHbImagePtr, pfIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	ConvertRGImage << <d3Block, d3Thread >> > (pnDeviceRChannels, pnDeviceGChannels, pnDeviceIHbImagePtr);

	HANDLE_ERROR(cudaMemcpy(pfRChannels, pnDeviceRChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfGChannels, pnDeviceGChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));


	HANDLE_ERROR(cudaFree(pnDeviceRChannels));
	HANDLE_ERROR(cudaFree(pnDeviceGChannels));
	HANDLE_ERROR(cudaFree(pnDeviceIHbImagePtr));
}

void RecoverImage(float* pfRChannels, float* pfGChannels, float* pfIHbImagePtr, int nImageWidth, int nImageHeight) {
	float* pfDeviceRChannels, * pfDeviceGChannels, * pfDeviceIHbImagePtr;

	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceRChannels, pfRChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceGChannels, pfGChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceIHbImagePtr, pfIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	ConvertRGImage << <d3Block, d3Thread >> > (pfDeviceRChannels, pfDeviceGChannels, pfDeviceIHbImagePtr);

	HANDLE_ERROR(cudaMemcpy(pfRChannels, pfDeviceRChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfGChannels, pfDeviceGChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));


	HANDLE_ERROR(cudaFree(pfDeviceRChannels));
	HANDLE_ERROR(cudaFree(pfDeviceGChannels));
	HANDLE_ERROR(cudaFree(pfDeviceIHbImagePtr));
}

__global__ void ConvertRGImage1(float* pfRChannels, float* pfGChannels, float* pfBChannels, float* pfOriginIHbImagePtr, float* pfIHbImagePtr) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	//pfGChannels[nOffset] = pfRChannels[nOffset] / pow(2, pfIHbImagePtr[nOffset] / 32.0f);
	//pfGChannels[nOffset] = pfRChannels[nOffset] * pow(2, pfIHbImagePtr[nOffset] / 32.0f);
	float  fERRatio, fEGRatio, fEBRatio;

	float fDIHbImagePtr = pfOriginIHbImagePtr[nOffset] - pfIHbImagePtr[nOffset];
	float fAGray = (pfRChannels[nOffset] + pfGChannels[nOffset] + pfBChannels[nOffset])/3.0f;
	
	if (fAGray < 5.0f) {
		fERRatio = 0.0f;
		fEGRatio = 0.0f;
		fEBRatio = 0.0f;
	}
	else if (fDIHbImagePtr > 60.0f) {
		fERRatio = 0.0f;
		fEGRatio = 0.0f;
		fEBRatio = 0.0f;
	}
	else {
		fERRatio = cfREpsilon[0] * (fDIHbImagePtr) / (106.32 * (cfGEpsilon[0] - cfREpsilon[0]));
		fEGRatio = cfGEpsilon[0] * (fDIHbImagePtr) / (106.32 * (cfGEpsilon[0] - cfREpsilon[0]));
		fEBRatio = cfBEpsilon[0] * (fDIHbImagePtr) / (106.32 * (cfGEpsilon[0] - cfREpsilon[0]));
	}


	pfRChannels[nOffset] = pfRChannels[nOffset] * pow(10.0, fERRatio);
	pfGChannels[nOffset] = pfGChannels[nOffset] * pow(10.0, fEGRatio);
	pfBChannels[nOffset] = pfBChannels[nOffset] * pow(10.0, fEBRatio);
}

void RecoverImage(float* pfRChannels, float* pfGChannels, float* pfBChannels, float fREpsilon, float fGEpsilon, float fBEpsilon, float* pfOriginIHbImagePtr, float* pfIHbImagePtr, int nImageWidth, int nImageHeight) {
	float* pfDeviceRChannels, * pfDeviceGChannels, * pfDeviceBChannels, * pfDeviceIHbImagePtr, * pfDeviceOriginIHbImagePtr;

	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceBChannels, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceOriginIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceRChannels, pfRChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceGChannels, pfGChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceBChannels, pfBChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceIHbImagePtr, pfIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceOriginIHbImagePtr, pfOriginIHbImagePtr, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	HANDLE_ERROR(cudaMemcpyToSymbol(cfREpsilon, &fREpsilon, sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(cfGEpsilon, &fGEpsilon, sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(cfBEpsilon, &fBEpsilon, sizeof(float)));

	ConvertRGImage1 << <d3Block, d3Thread >> > (pfDeviceRChannels, pfDeviceGChannels, pfDeviceBChannels, pfDeviceOriginIHbImagePtr, pfDeviceIHbImagePtr);

	HANDLE_ERROR(cudaMemcpy(pfRChannels, pfDeviceRChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfGChannels, pfDeviceGChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfBChannels, pfDeviceBChannels, sizeof(float) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceRChannels));
	HANDLE_ERROR(cudaFree(pfDeviceGChannels));
	HANDLE_ERROR(cudaFree(pfDeviceBChannels));
	HANDLE_ERROR(cudaFree(pfDeviceIHbImagePtr));
	HANDLE_ERROR(cudaFree(pfDeviceOriginIHbImagePtr));
}

__global__ void AddRatio3(int* pnIHbImage, int* pnEnhaceIHbImage, int nK, int nImageW, int nImageH) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
	const int nBlockWidth = nBlockDimX + __mul24(nKernelRSz5, 2);
	const int nBlockHeight = nBlockDimY + __mul24(nKernelRSz5, 2);
	const int nShift = __mul24(threadIdx.y, nBlockWidth);
	int nOffset = nX + __mul24(nY, __mul24(blockDim.x, gridDim.x));

	__shared__ int nConvData[(nBlockDimY + 2 * nKernelRSz5) * (nBlockDimX + 2 * nKernelRSz5)];

	if (threadIdx.x < nKernelRSz5 || threadIdx.x >= nBlockDimX - nKernelRSz5 ||
		threadIdx.y < nKernelRSz5 || threadIdx.y >= nBlockDimY - nKernelRSz5) {
		if ((nX - nKernelRSz5) < 0 || (nY - nKernelRSz5) < 0) {
			nConvData[threadIdx.x + nShift] = pnIHbImage[nOffset];
		}
		else {
			nConvData[threadIdx.x + nShift] = pnIHbImage[nOffset - __mul24(nKernelRSz5, nImageW) - nKernelRSz5];
		}

		if ((nX + nKernelRSz5) > nImageW - 1 || (nY - nKernelRSz5) < 0) {
			nConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift] = pnIHbImage[nOffset];
		}
		else {
			nConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift] = pnIHbImage[nOffset - __mul24(nKernelRSz5, nImageW) + nKernelRSz5];
		}

		if ((nX - nKernelRSz5) < 0 || (nY + nKernelRSz5) > nImageH - 1) {
			nConvData[threadIdx.x + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pnIHbImage[nOffset];
		}
		else {
			nConvData[threadIdx.x + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pnIHbImage[nOffset + __mul24(nKernelRSz5, nImageW) - nKernelRSz5];
		}

		if ((nX + nKernelRSz5) > nImageW - 1 || (nY + nKernelRSz5) > nImageH - 1) {
			nConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pnIHbImage[nOffset];
		}
		else {
			nConvData[threadIdx.x + __mul24(2, nKernelRSz5) + nShift + __mul24(__mul24(2, nKernelRSz5), nBlockWidth)] = pnIHbImage[nOffset + __mul24(nKernelRSz5, nImageW) + nKernelRSz5];
		}
		nConvData[threadIdx.x + nKernelRSz5 + nShift + __mul24(nKernelRSz5, nBlockWidth)] = pnIHbImage[nOffset];
	}
	else {
		nConvData[threadIdx.x + nKernelRSz5 + nShift + __mul24(nKernelRSz5, nBlockWidth)] = pnIHbImage[nOffset];
	}

	__syncthreads();

	if (pnIHbImage[nOffset] < 1)
		pnIHbImage[nOffset] = 0;

	int r, c;
	int nSumV = 0;
	for (int r = 0; r < 5; r++) {
		for (int c = 0; c < 5; c++) {
			nSumV += nConvData[(threadIdx.y + r) * nBlockWidth + (threadIdx.x + c)];
		}
	}
	int nMeanV = nSumV * 41;
	nMeanV = nMeanV >> 10;

	int nTempV = abs(pnIHbImage[nOffset] - nMeanV) << 7;
	int nCK = 128 - nTempV / (4 * nMeanV);

	if (nCK < 0) {
		nCK = 1;
	}

	long long  dlTestV = nMeanV << 7;

	long long dlTempV = (long long)(pnIHbImage[nOffset] - nMeanV) * (long long)nK /** (long long)nCY*//** (long long)(128)*/ + dlTestV;

	pnEnhaceIHbImage[nOffset] = dlTempV >> 7;
}

void EnhanceIHbImageInt(int* pnIHbImagePtr, int* pnEnhanceIHbImagePtr, int nK, int nImageWidth, int nImageHeight) {
	int* pnDeviceIHbImage, * pnDeviceEnhanceIHbImgePtr;

	HANDLE_ERROR(cudaMalloc(&pnDeviceIHbImage, sizeof(int) * nImageWidth * nImageHeight));
	HANDLE_ERROR(cudaMalloc(&pnDeviceEnhanceIHbImgePtr, sizeof(int) * nImageWidth * nImageHeight));

	HANDLE_ERROR(cudaMemcpy(pnDeviceIHbImage, pnIHbImagePtr, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyHostToDevice));


	//HANDLE_ERROR(cudaMemcpyToSymbol(nMeanY, &nAverageY, sizeof(int)));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nImageWidth / 16, nImageHeight / 16);

	AddRatio3 << <d3Block, d3Thread >> > (pnDeviceIHbImage, pnDeviceEnhanceIHbImgePtr, nK, nImageWidth, nImageHeight);

	HANDLE_ERROR(cudaMemcpy(pnEnhanceIHbImagePtr, pnDeviceEnhanceIHbImgePtr, sizeof(int) * nImageWidth * nImageHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(pnDeviceIHbImage));
	HANDLE_ERROR(cudaFree(pnDeviceEnhanceIHbImgePtr));
}

__global__ void convertRGB(float* pfR, float* pfG, float* pfB) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	float fMax, fMin;
	if (pfG[nOffset] > pfB[nOffset]) {
		fMax = pfG[nOffset];
		fMin = pfB[nOffset];
	}
	else {
		fMax = pfB[nOffset];
		fMin = pfG[nOffset];
	}

	float fDiff = (fabs(pfR[nOffset] - pfB[nOffset]) + fabs(pfR[nOffset] - pfG[nOffset])) / 2.0f;

	pfR[nOffset] = pfR[nOffset] * (exp(fDiff / 255.0f));
	pfG[nOffset] = fMin;
	pfB[nOffset] = fMin;
}

void EnhanceRGB(float* pfR, float* pfG, float* pfB, int nWidth, int nHeight) {
	float* pfDeviceR, * pfDeviceG, * pfDeviceB;

	HANDLE_ERROR(cudaMalloc(&pfDeviceR, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceG, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceB, sizeof(float) * nWidth * nHeight));

	HANDLE_ERROR(cudaMemcpy(pfDeviceR, pfR, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceG, pfG, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceB, pfG, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));


	dim3 d3Thread(16, 16);
	dim3 d3Block(nWidth / 16, nHeight / 16);
	convertRGB << <d3Block, d3Thread >> > (pfDeviceR, pfDeviceG, pfDeviceB);

	HANDLE_ERROR(cudaMemcpy(pfR, pfDeviceR, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfG, pfDeviceG, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfB, pfDeviceB, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceR));
	HANDLE_ERROR(cudaFree(pfDeviceG));
	HANDLE_ERROR(cudaFree(pfDeviceB));
}

__global__ void cRGB(float* pfR, float* pfG, float* pfB) {
	int nX = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int nY = threadIdx.y + __mul24(blockIdx.y, blockDim.y);

	int nOffset = nY * __mul24(blockDim.x, gridDim.x) + nX;

	pfR[nOffset] = 0.5f * pfG[nOffset] + 0.5f * pfB[nOffset];
	pfG[nOffset] = pfG[nOffset] /*+ 0.6* pfR[nOffset]*/;
	pfB[nOffset] = pfB[nOffset] /*+ 0.2* pfR[nOffset]*/;
}

void ChangeRGB(float* pfRChannel, float* pfGChannel, float* pfBChannel, int nWidth, int nHeight) {
	float* pfDeviceRChannel, * pfDeviceGChannel, * pfDeviceBChannel;
	HANDLE_ERROR(cudaMalloc(&pfDeviceRChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceGChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMalloc(&pfDeviceBChannel, sizeof(float) * nWidth * nHeight));
	HANDLE_ERROR(cudaMemcpy(pfDeviceRChannel, pfRChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceGChannel, pfGChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(pfDeviceBChannel, pfBChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyHostToDevice));

	dim3 d3Thread(16, 16);
	dim3 d3Block(nWidth / 16, nHeight / 16);
	cRGB << <d3Block, d3Thread >> > (pfDeviceRChannel, pfDeviceGChannel, pfDeviceBChannel);

	HANDLE_ERROR(cudaMemcpy(pfRChannel, pfDeviceRChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfGChannel, pfDeviceGChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(pfBChannel, pfDeviceBChannel, sizeof(float) * nWidth * nHeight, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pfDeviceRChannel));
	HANDLE_ERROR(cudaFree(pfDeviceGChannel));
	HANDLE_ERROR(cudaFree(pfDeviceBChannel));
}