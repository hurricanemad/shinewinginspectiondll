
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#ifndef __CUDACC_RTC__
#define __CUDACC_RTC__
#endif

#include "cuda_texture_types.h"
#include "cuda.h"

#include <opencv2/core.hpp>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLib(name) "opencv_" name CV_VERSION_ID "d.lib"
#else
#define cvLib(name) "opencv_" name CV_VERSION_ID ".lib"
#endif

#pragma comment(lib, cvLib("core"))

#define __CUDA_INTERNAL_COMPILATION__
//#include "device_functions.h"
//#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__





__device__ const int nBlockDimX = 16;
__device__ const int nBlockDimY = 16;
__device__ const int nKernelSz = 3;
__device__ const int nKernelRSz = nKernelSz / 2;
__device__ const int nLaplacianKSz = 5;
__device__ const int nLaplacianKRSz = nLaplacianKSz / 2;
__device__ const int nMeanKernelSz = 5;
__device__ const int nMeanKernelRadius = nMeanKernelSz / 2;
__device__ const int nMultiple = 2;
__device__ const int nCEKernelSz = 5;
__device__ const int nCEKernelRadius = nCEKernelSz / 2;
__device__ const int nBoxFKernelSz = 5;
__device__ const int nBoxFKernelRadius = nBoxFKernelSz / 2;



__global__ void DiffImage(unsigned char* pucImagePtr1, unsigned char* pucImagePtr2, int* pucImage3);
__global__ void DiffImage(float* pfImagePtr1, float* pfImagePtr2, float* pfImage3);
__global__ void MeanImage(float* pfFilterPtr, int nImageW, int nImageH, int nKernelSzW, int nKernelSzH);
__global__ void MeanImage(float* pfSrcImagePtr, float* pfFilterPtr, int nImageW, int nImageH);
__global__ void FilterImage(unsigned char* pucSrcImagePtr, int* pnDiffImagePtr, float* pfFilterImagePtr, unsigned char* pucResultImgPtr, float* pfResultImagePtr, int nImageW, int nImageH, float fRRR);
__global__ void FilterImage(float* pfSrcImagePtr, float* pfDiffImagePtr, float* pfFilterImagePtr, float* pucResultImgPtr, float* pfResultImagePtr, int nImageW, int nImageH, float fRRR);
__global__ void ConvolutionRow(float* pfImagePtr, float* pfConvoImagePtr, int nImageW, int nImageH);
__global__ void ConvolutionCol(float* pfImagePtr, float* pfConvoImagePtr, int nImageW, int nImageH);
__global__ void LaplacianFilterRow1(float* pfImagePtr, float* pfLFImagePtr, int nImageW, int nImageH);
__global__ void LaplacianFilterRow2(float* pfImagePtr, float* pfLFImagePtr, int nImageW, int nImageH);
__global__ void LaplacianFilterCol1(float* pfImagePtr, float* pfLFImagePtr, int nImageW, int nImageH);
__global__ void LaplacianFilterCol2(float* pfImagePtr, float* pfLFImagePtr, int nImageW, int nImageH);
__global__ void BoxFilter(float* pfSrcImagePtr, float* pfFilterPtr, int nImageW, int nImageH);
__global__ void MeanImage(float* pfMeanImage, float* pfSrcImage, int nMeanFN);
__global__ void ConvertRGImage(int* pnRChannels, int* pnGChannels, int* pnIHbImagePtr);

void GenGauss1D(float* pfKernel, int nKernelW, float fSigma);
void GenLaplaceKernel1D5Sz(float* pfKernelX, float* pfKernelY);
void GenYImageInt(int* pnRChannel, int* pnGChannel, int* pnBChannel,
	int* pnYChannel, int nImageWidth, int nImageHeight);
void MANRFilter(unsigned char* pucSrcImagePtr, unsigned char* pucResultImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, double dRatio);
void MANRFilter(float* pfSrcImagePtr, float* pfDstImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, double dRatio);
void EnhanceImage(float* fImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, int nKernelW, int nEnhanceLevel, bool bLG);
void AINDANEEnhance(float* pfSrcImage, float* pfEnhanceImage, float fGamma, float fP, bool bCEnhance, int nImageW, int nImageH);
void CurveImageEnhance(float* pfSrcImage, float* pfEnhanceImage, double* pdCurveImagePts, double* pdCurveImageParams, int nCurveImageArraySz, int nImageW, int nImageH);

void GuidedFilter(float* pfMatI, float* pfMatP, float* pfResultImg, int nImageW, int nImageH, int nKernelSz, float fEpsilon);
void GaussianFilter(float* pfMatI, float* pfMatP, float* pfResultImg, int nImageW, int nImageH, int nKernelSz, double d);
void DarkMeanImage(float* pfMeanImage, float* pfSrcImage, int nImageW, int nImageH, int nMeanFN);
void DarkRefine(float* pfSrcImage, float* pfDarkImage, int nImageW, int nImageH);
void CCMConvert(float* pfY, float* pfU, float* pfV,
	            float* pfCY, float* pfCU, float* pfCV,
	            float* pfCCM, int nImageW, int nImageH,
	            int nHSVAlter, int nYUVAlter,
	            int nHue, int nSaturation, int nValue,
				int nSaturation1);

void FigureBrightnessPixelSz(const cv::Mat& matResizeImage, long* plBPSz, float fThresh);
void FigureDarknessPixelSz(const cv::Mat& matResizeImage, long* plDPSz);

void EnhanceImageChange(float* fImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, float fEnhanceSigma, float fEnhanceP);
void EnhanceImageChange1(float* pfBKImagePtr, float* fDetailImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, float fEnhanceSigma, float fEnhanceThresh, float fEnhanceP);
void EnhanceImageChange2(float* pfBKImagePtr, float* fDetailImagePtr, float* pfResultImagePtr, int nImageW, int nImageH, float fEnhanceThresh, float fEnhanceP);
void EnhanceIHbImage3(float* pfIHbImagePtr, float* pfYImagePtr, float fAverageIHb, float fAverageY, float* pfEnhanceIHbImagePtr, float fK, int nImageWidth, int nImageHeight);
void EnhanceIHbImageInt1(int* pnIHbImagePtr, int* pnYImagePtr, int nAverageIHb, int nAverageY, int* pnEnhanceIHbImagePtr, int nK, int nImageWidth, int nImageHeight);
void substractDB(float* pfOriginImage, float* pfBKImage, float* pfDetailImage, int nWidth, int nHeight);
void convertYUV2RGB(float* pfYChannel, float* pfUChannel, float* pfVChannel,
	float* pfRChannel, float* pfGChannel, float* pfBChannel,
	int nWidth, int nHeight);
void convertRGB2YUV(float* pfRChannel, float* pfGChannel, float* pfBChannel,
	float* pfYChannel, float* pfUChannel, float* pfVChannel,
	int nWidth, int nHeight);

void ColorCorrectionM(float* pfRChannel, float* pfGChannel, float* pfBChannel, float* pfColorCorrectionM, int nwidth, int nheight);

void convertRGB2HSV(float* pfRChannel, float* pfGChannel, float* pfBChannel,
									 float* pfHChannel, float* pfSChannel, float* pfVChannel, int nWidth, int nHeight);

void convertHSV2RGB(float* pfHChannel, float* pfSChannel, float* pfVChannel,
									 float* pfRChannel, float* pfGChannel, float* pfBChannel, int nWidth, int nHeight);

void EnhanceHSV(float* pfHChannel, float* pfSChannel, float* pfVChannel, int nWidth, int nHeight, int nHue, int nSaturation, int nValue);

void GenIHbImage(float* pfRChannels, float* pfGChannels, float* pfIHbImage,
	int nImageWidth, int nImageHeight);
void GenIHbImageInt(int* pnRChannels, int* pnGChannels, int* pnIHbImage,
	int nImageWidth, int nImageHeight);

float MeanImage1(cv::Mat matInputImage);
int MeanImage2(cv::Mat matInputImage);

void EnhanceIHbImage(float* pfIHbImagePtr, float fAverageIHb, float fAverageY, float* pfEnhanceIHbImagePtr, float fK, int nImageWidth, int nImageHeight);
void EnhanceIHbImage1(float* pfIHbImagePtr, float fAverageIHb, float fAverageY, float* pfEnhanceIHbImagePtr, float fK, int nImageWidth, int nImageHeight);
void EnhanceIHbImage2(float* pfIHbImagePtr, float* pfEnhanceIHbImagePtr, float fK, int nImageWidth, int nImageHeight);
void EnhanceIHbImageInt(int* pnIHbImagePtr, int* pnEnhanceIHbImagePtr, int nK, int nImageWidth, int nImageHeight);

void RecoverImage(float* pfRChannels, float* pfGChannels, float* pfIHbImagePtr, int nImageWidth, int nImageHeight);
void RecoverImage(int* pfRChannels, int* pfGChannels, int* pfIHbImagePtr, int nImageWidth, int nImageHeight);
void RecoverImage(float* pfRChannels, float* pfGChannels, float* pfBChannels, float fREpsilon, float fGEpsilon, float fBEpsilon, float* pfOriginIHbImagePtr, float* pfIHbImagePtr, int nImageWidth, int nImageHeight);

void EnhanceRGB(float* pfR, float* pfG, float* pfB, int nWidth, int nHeight);
void ChangeRGB(float* pfRChannel, float* pfGChannel, float* pfBChannel, int nWidth, int nHeight);

