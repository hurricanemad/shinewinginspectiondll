// pch.cpp: 与预编译标头对应的源文件

#include "pch.h"

// 当使用预编译的头时，需要使用此源文件，编译才能成功。
double dFormerRatio;
vector<cv::Mat>* gvmatPreImage = NULL;
double* pParam = NULL;
bool bEnhanceImage = false;
bool bEnhanceImageChange = false;
bool bDenoiceImage = false;
bool bLumaAdjust = false;
bool bCCMConvert = false;
bool bCCMEnable = false;
bool bChangeRGB = false;
bool bGuidedFilter = false;
bool bSegment = false;
bool bGammaCEEnhance = false;
bool bGammaImage = false;
bool bCurveImage = false;
bool bDarkRefine = false;
bool bOpenLight = false;
bool bLaplacianGaussEnable = false;
bool bGainAdd = false;
bool bExposureAdd = false;
bool bSubGuidedFilter = false;
bool bSaveVideo = false;
bool bSaveProcessVideo = false;
bool bFormerSaveProcessVideo = false;

float fA = 1062.4f;
float fB = -0.2723f;
float fC = 10.0f;
float fR = 260.0f;

int nProcessVideoCounter = 0;
int nProcessFrameCounter = 0;
//曲线灰度增强算法变量
double* pdCurveImagePts = NULL;
double* pdCurveImageParams = NULL;
int nCurveImageArraySz = 0;
cv::VideoWriter* vwVideoCapturePtr = NULL;
cv::VideoWriter* vwProcessVideoCapturePtr = NULL;


SerialPortControl* g_spcSerialController = NULL;

float fGFEpsilon = 5.0f;
float fSegEpsilon = 200.0f;
float fGamma = 0.0f;
float fCEPlus = 0.0f;
float fEnhanceP = 1.0f;
float fEnhanceThresh = 0.0;
float fColorEnhanceV = 2.2f;


cv::Mat* pmatDarkImage = NULL;
int nBUExposure;
int nLightV;
int nBUGain;
double* pdCurvePtXs = NULL, * pdCurvePtYs = NULL;
float* pfCCMatrix = NULL;

double* pdSubAreaBrenner;
double* pdSubAreaTenegrad;
double* pdSubAreaLaplacian;

double* pdUnformityVs;

double* pdRB, * pdRG, * pdRR;

int nCurvePtSz;

int nFormerAverageY = 0;
int nFormerAverageIHb = 0;

std::queue<long>qBPixelsSize;
//std::queue<int>qGain;

//std::mutex mutexCurvePts;
HANDLE  hMutexCurvePts = NULL;
cv::Mat* pmatPreImage = NULL;

double dBrenner = 0.0;
double  dTenegrad = 0.0;
double dLaplacian = 0.0;
double dBrightnessRatio = 0.0;

volatile int nHSVAble = 0;
volatile int nIHbEnhance = 0;

volatile int nYUVAble = 0;
volatile int nMatrixEnable = 0;
volatile int nHue = 0;
volatile int nSaturation = 0;
volatile int nSaturation1M = 0;
volatile int nValue = 0;
volatile long lBThreshPx = 0;
volatile long lDThreshPx = 0;
int nLeftPosFOV, nRightPosFOV;
int nLeftNegFOV, nRightNegFOV;

long lBrightPxSize = 0l;
long lDarkPxSize = 0l;
long lBrightFrameSz = 0l;
long lDelayFrameSz = 0l;
long lDarkFrameSz = 0l;
const float cfOlympusREpsilon = 7091.6537963317596f;
const float cfOlympusGEpsilon = 36489.288632926873f;
const float cfOlympusBEpsilon = 70189.207165583983f;
const float cfOurREpsilon = 34.762112168333452f;
const float cfOurGEpsilon = 173.34472748998573f;
const float cfOurBEpsilon = 156.38102981052560f;

const int cnHalfImageWidth = 640;
const int cnHalfImageHeight = 360;

struct FResult {
	double* pdFormula;
	int nCounter;

	FResult() {
		pdFormula = NULL;
		nCounter = 0;
	}

	FResult(int nESz) {
		if (pdFormula) {
			delete[] pdFormula;
		}
		pdFormula = new double[nESz];
		nCounter = nESz;
	}

	void resize(int nESz) {
		if (pdFormula) {
			delete[] pdFormula;
		}
		pdFormula = new double[nESz];

		memset(pdFormula, 0, nESz * sizeof(double));
		nCounter = nESz;
	}

	void GenEye(int nESz) {
		if (pdFormula) {
			delete[] pdFormula;
		}
		pdFormula = new double[nESz];
		nCounter = nESz;

		int n, r, c;
		int nRoot = sqrt(nESz);

		for (n = 0; n < nESz; n++) {
			r = n / nRoot;
			c = n % nRoot;
			if (r == c) {
				pdFormula[r * nRoot + c] = 1.0;
			}
			else {
				pdFormula[r * nRoot + c] = 0.0;
			}

		}
	}

	void zeros() {
		int n;
		for (n = 0; n < nCounter; n++) {
			pdFormula[n] = 0.0;
		}
	}

	~FResult() {
		delete[] pdFormula;
		nCounter = 0;
	}
};

#ifdef LOG_ENABLE
ofstream* pfLog = NULL;
#endif

#ifdef LOG_TIME_ENABLE
ofstream* pfLogTime = NULL;
#endif



double GetBrenner() {
	return dBrenner;
}

double GetTenegrad() {
	return dTenegrad;
}

double GetLaplacian() {
	return dLaplacian;
}

double* GetSubAreaBrenner() {
	return pdSubAreaBrenner;
}

double* GetSubAreaTenegrad() {
	return pdSubAreaTenegrad;
}

double* GetSubAreaLaplacian() {
	return pdSubAreaLaplacian;
}

int GetPosLeftFOV() {
	return nLeftPosFOV;
}

int GetPosRightFOV() {
	return nRightPosFOV;
}

int GetNegLeftFOV() {
	return nLeftNegFOV;
}

int GetNegRightFOV() {
	return nRightNegFOV;
}

double* GetUnformityVs() {
	return pdUnformityVs;
}

double* GetpRR() {
	return pdRR;
}

double* GetpRG() {
	return pdRG;
}

double* GetpRB() {
	return pdRB;
}

double GetBrighnessRatio() {
	return dBrightnessRatio;
}

void InitializeImage(int nImageW, int nImageH, double dRatio) {
	pmatPreImage = new cv::Mat(720, 1280, CV_8UC1);

	pdSubAreaBrenner = new double[9];
	pdSubAreaTenegrad = new double[9];
	pdSubAreaLaplacian = new double[9];

	pdUnformityVs = new double[8];

	pdRB = new double[5];
	pdRG = new double[5];
	pdRR = new double[5];
	memset(pdRB, 0, sizeof(double) * 5);
	memset(pdRG, 0, sizeof(double) * 5);
	memset(pdRR, 0, sizeof(double) * 5);
	//pParam = new double[257];
	//ReadParam(dRatio);
}

void ReleaseImage() {
	if (pmatPreImage) {
		delete pmatPreImage;
		pmatPreImage = NULL;
	}

	if (pdSubAreaBrenner) {
		delete[] pdSubAreaBrenner;
		pdSubAreaBrenner = NULL;
	}

	if (pdSubAreaTenegrad) {
		delete[] pdSubAreaTenegrad;
		pdSubAreaTenegrad = NULL;
	}

	if (pdSubAreaLaplacian) {
		delete[] pdSubAreaLaplacian;
		pdSubAreaLaplacian = NULL;
	}

	if (pdUnformityVs) {
		delete[] pdUnformityVs;
		pdUnformityVs = NULL;
	}
	if (pdRB) {
		delete[] pdRB;
		pdRB = NULL;
	}
	if (pdRG) {
		delete[] pdRG;
		pdRG = NULL;
	}
	if (pdRR) {
		delete[] pdRR;
		pdRR = NULL;
	}
	//if (pParam) {
	//    delete[] pParam;
	//    pParam = NULL;
	//}

}


int InitializeSerialControl() {
	g_spcSerialController = new SerialPortControl();

#ifdef LOG_ENABLE
	pfLog = new ofstream("TestLog.txt", std::ios_base::out);
#endif
	g_spcSerialController->FindPort();
	int nSeekSerialPort = g_spcSerialController->GetUARTNo();

	if (nSeekSerialPort >= 0 && nSeekSerialPort <= 256) {
#ifdef LOG_ENABLE
		*pfLog << "Seek serial port OK!" << nSeekSerialPort << endl;
#endif
		g_spcSerialController->InitPort();
		return nSeekSerialPort;
	}
	return 1;
}

void ReleaseSerialControl() {

	if (g_spcSerialController) {
		g_spcSerialController->ClosePort();
		delete g_spcSerialController;
		g_spcSerialController = NULL;
	}
#ifdef LOG_ENABLE
	delete pfLog;
#endif
}

void ReadParam(double dRatio) {
	std::ifstream fin;

	int nTRatio;
	nTRatio = static_cast<int>(dRatio * 5.0);

	int n;
	switch (nTRatio) {
	case 2:
		fin.open("LUT0.4.txt", std::ios_base::in);
		for (n = 0; n <= 256; n++) {
			fin >> pParam[n];
		}

		break;
	case 3:
		fin.open("LUT0.6.txt", std::ios_base::in);
		for (n = 0; n <= 256; n++) {
			fin >> pParam[n];
		}
		break;
	case 4:
		fin.open("LUT0.8.txt", std::ios_base::in);
		for (n = 0; n <= 256; n++) {
			fin >> pParam[n];
		}
		break;
	case 5:
		fin.open("LUT1.0.txt", std::ios_base::in);
		for (n = 0; n <= 256; n++) {
			fin >> pParam[n];
		}
		break;
	default:
		break;
	}
}

void StartVideoCapture(int strFIleName, int nFrameRate, int nWidth, int nHeight) {
	//ofstream fout("FileName.txt", ios_base::out|ios_base::ate);

	//fout << strFIleName;

	//fout.close();
	char cName[100];
	sprintf_s(cName, "C://Video//Video%d.mp4", strFIleName);
	vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);
	vwVideoCapturePtr = new cv::VideoWriter(cName, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), nFrameRate, cv::Size(nWidth, nHeight), compression_params);


}

void StopVideoCapture() {
	if (vwVideoCapturePtr) {
		vwVideoCapturePtr->release();
		delete vwVideoCapturePtr;
	}

}

void SaveVideo(uchar* pucImagePtr, int nImageW, int nImageH) {
	cv::Mat matYUVImg = cv::Mat(nImageH, nImageW, CV_8UC2, pucImagePtr);
	cv::Mat matProcessImg;

	cv::cvtColor(matYUVImg, matProcessImg, cv::COLOR_YUV2BGR_YUY2);
	(*vwVideoCapturePtr) << matProcessImg;

}

void SaveImage(uchar* pucImagePtr, int nImageW, int nImageH, int nNo) {
	cv::Mat matYUVImg = cv::Mat(nImageH, nImageW, CV_8UC2, pucImagePtr);
	cv::Mat matProcessImg;

	cv::cvtColor(matYUVImg, matProcessImg, cv::COLOR_YUV2BGR_YUY2);

	//cv::Mat matSaveImg = cv::Mat(nImageH, nImageW, CV_8UC4, pucImagePtr);
	char cNo[100];
	sprintf_s(cNo, "Image%d.bmp", nNo);
	cv::imwrite(cNo, matProcessImg);
}

/*******************************************************/
/*@@@@@函数名:convertYUY2ToY@@@@@@@*/
/*输入：*/
/*pucImagePtr 类型：unsigned char 说明：输入图像地址指针，图像为YUYV格式*/
/*nImageW 类型：int 说明：输入图像宽度*/
/*nImageH 类型：int 说明：输入图像高度*/
/*pucYImagePtr 类型：uchar* 说明：用于输出YUYV图像中的Y通道图像*/
/*输出：空*/
/******************************************************/

void convertYUY2ToY(uchar* pucImagePtr, int nImageW, int nImageH, uchar* pucYImagePtr) {

	for (int n = 0; n < nImageW * nImageH; n++) {
		pucYImagePtr[n] = pucImagePtr[2 * n];
	}
}


void FigurebrennerFloat(cv::Mat& matSrcImage, double& dBrenner) {
	int r, c;

	float* pmatSrcImage;
	dBrenner = 0.0;
	for (r = 0; r < matSrcImage.rows; r++) {
		pmatSrcImage = matSrcImage.ptr<float>(r);
		for (c = 0; c < matSrcImage.cols - 2; c++) {
			dBrenner += (pmatSrcImage[c + 2] - pmatSrcImage[c]) * (pmatSrcImage[c + 2] - pmatSrcImage[c]);
		}
	}
	dBrenner /= (matSrcImage.rows * matSrcImage.cols);
}

void FigureSobelFloat(cv::Mat& matSrcImage, double& dTenegrad) {

	cv::Mat matSobelX, matSobelY;
	cv::Sobel(matSrcImage, matSobelX, CV_64FC1, 1, 0, 3);
	cv::Sobel(matSrcImage, matSobelY, CV_64FC1, 0, 1, 3);

	int r, c;

	dTenegrad = 0.0;
	double* pdmatSobelX, * pdmatSobelY;
	for (r = 0; r < matSrcImage.rows; r++) {
		pdmatSobelX = matSobelX.ptr<double>(r);
		pdmatSobelY = matSobelY.ptr<double>(r);
		for (c = 0; c < matSrcImage.cols; c++) {
			dTenegrad += pdmatSobelX[c] * pdmatSobelX[c] + pdmatSobelY[c] * pdmatSobelY[c];
		}
	}
	dTenegrad /= (matSrcImage.rows * matSrcImage.cols);
}


void FigureLaplacianFloat(cv::Mat& matSrcImage, double& dLaplacian) {
	cv::Mat matLaplacianImage;

	cv::Laplacian(matSrcImage, matLaplacianImage, CV_32FC1, 1);

	int r, c;
	float* pdmatLapImage;
	dLaplacian = 0.0;
	for (r = 0; r < matSrcImage.rows; r++) {
		pdmatLapImage = matLaplacianImage.ptr<float>(r);
		for (c = 0; c < matSrcImage.cols; c++) {
			dLaplacian += pdmatLapImage[c] * pdmatLapImage[c];
		}
	}
	dLaplacian /= (matSrcImage.rows * matSrcImage.cols);
}


void Figurebrenner(cv::Mat& matSrcImage, double& dBrenner) {
	int r, c;

	uchar* pmatSrcImage;
	dBrenner = 0.0;
	for (r = 0; r < matSrcImage.rows; r++) {
		pmatSrcImage = matSrcImage.ptr<uchar>(r);
		for (c = 0; c < matSrcImage.cols - 2; c++) {
			dBrenner += (pmatSrcImage[c + 2] - pmatSrcImage[c]) * (pmatSrcImage[c + 2] - pmatSrcImage[c]);
		}
	}
	dBrenner /= (matSrcImage.rows * matSrcImage.cols);
}

void FigureSobel(cv::Mat& matSrcImage, double& dTenegrad) {

	cv::Mat matSobelX, matSobelY;
	cv::Sobel(matSrcImage, matSobelX, CV_64FC1, 1, 0, 3);
	cv::Sobel(matSrcImage, matSobelY, CV_64FC1, 0, 1, 3);

	int r, c;

	dTenegrad = 0.0;
	double* pdmatSobelX, * pdmatSobelY;
	for (r = 0; r < matSrcImage.rows; r++) {
		pdmatSobelX = matSobelX.ptr<double>(r);
		pdmatSobelY = matSobelY.ptr<double>(r);
		for (c = 0; c < matSrcImage.cols; c++) {
			dTenegrad += pdmatSobelX[c] * pdmatSobelX[c] + pdmatSobelY[c] * pdmatSobelY[c];
		}
	}
	dTenegrad /= (matSrcImage.rows * matSrcImage.cols);
}


void FigureLaplacian(cv::Mat& matSrcImage, double& dLaplacian) {
	cv::Mat matLaplacianImage;

	cv::Laplacian(matSrcImage, matLaplacianImage, CV_64FC1, 1);

	int r, c;
	double* pdmatLapImage;
	dLaplacian = 0.0;
	for (r = 0; r < matSrcImage.rows; r++) {
		pdmatLapImage = matLaplacianImage.ptr<double>(r);
		for (c = 0; c < matSrcImage.cols; c++) {
			dLaplacian += pdmatLapImage[c] * pdmatLapImage[c];
		}
	}
	dLaplacian /= (matSrcImage.rows * matSrcImage.cols);
}

//double dFormerBrenner;

void DiffImage(uchar* pmatSrcImage, uchar* pmatPreImage, float* pfDiffImage, int nImageW, int nImageH) {
	for (int n = 0; n < nImageW * nImageH; n++) {
		pfDiffImage[n] = pmatSrcImage[n] - pmatPreImage[n];
	}
}

void FilterImage(uchar* pucSrcImage, float* pfDiffImage, float* pfFilterDiffImage, uchar* pucPreImage, float* pfResultImage, int nImageW, int nImageH, double dRatio) {
	float fV;
	float fTA = fA * dRatio;
	float fRatio;

	for (int n = 0; n < nImageW * nImageH; n++) {
		if (pfFilterDiffImage[n] > 128.0f) {
			fV = 128.0f;
		}
		else {
			fV = pfFilterDiffImage[n];
		}

		if (fV < 1e-6) {
			fRatio = (exp(pow(2.0 / fTA, fB)) + (2.0) * dRatio * 5.0) / fR;
		}
		else if (fV <= 2.0f) {
			fRatio = (exp(pow(2.0 / fTA, fB)) + (2.0 - fV) * dRatio * 5.0) / fR;
		}
		else {
			fRatio = exp(pow(fV / fTA, fB)) / fR;
		}

		pucPreImage[n] = (unsigned char)(pucSrcImage[n] - fRatio * pfDiffImage[n] + 0.5f);
		pfResultImage[n] = pucSrcImage[n] - fRatio * pfDiffImage[n] + 0.5f;
	}
}

void MANRFilterCPU(uchar* pmatSrcImage, uchar* pmatPreImage, float* pfResultImage, int nImageW, int nImageH, double dRatio) {
	cv::Mat matDiffImage = cv::Mat(nImageH, nImageW, CV_32FC1);
	cv::Mat matFilterDiffImage;

	DiffImage(pmatSrcImage, pmatPreImage, matDiffImage.ptr<float>(0), nImageW, nImageH);
	cv::boxFilter(matDiffImage, matFilterDiffImage, CV_32FC1, cv::Size(5, 5));
	matFilterDiffImage = cv::abs(matFilterDiffImage);
	FilterImage(pmatSrcImage, matDiffImage.ptr<float>(0), matFilterDiffImage.ptr<float>(0), pmatPreImage, pfResultImage, nImageW, nImageH, dRatio);
	cv::Mat matTempPreImage = cv::Mat(nImageH, nImageW, CV_8UC1, pmatPreImage);
	cv::Mat matTempResultImage = cv::Mat(nImageH, nImageW, CV_32FC1, pfResultImage);
}

const double cdWidthRatio = 8.715755371245493e-01;
const double cdHeightRatio = 4.902612396325590e-01;

/*******************************************************/
/*@@@@@函数名:Filter1DImage@@@@@@@*/
/*函数说明：对图像执行一维均值滤波。*/
/*输入：*/
/*matSrcImage 类型：Mat 说明:输入图像*/
/*matFilterImage 类型：Mat 说明：输出滤波后的图像*/
/*nKernelSz 类型：int 说明：滤波核大小*/
/*输出：空*/
/******************************************************/

void Filter1DImage(cv::Mat& matSrcImage, cv::Mat& matFilterImage, int nKernelSz) {
	matFilterImage = cv::Mat(matSrcImage.rows, matSrcImage.cols, CV_64FC1);

	double* pmatSrcImage = matSrcImage.ptr<double>(0), * pmatFilterImage = matFilterImage.ptr<double>(0);

	int nTempKernelSz = static_cast<int>(nKernelSz / 2) + 1;
	int nHalfKernelSz = nTempKernelSz / 2;
	int c, n;
	double dMean = 0.0;
	int nC = 0;
	for (c = 0; c < matSrcImage.cols; c++) {
		dMean = 0.0;
		nC = 0;
		for (n = -nHalfKernelSz; n <= nHalfKernelSz; n++) {
			if (c + n >= 0 && c + n < matSrcImage.cols) {
				dMean += pmatSrcImage[c + n];
				nC++;
			}
		}
		pmatFilterImage[c] = dMean / nC;

	}
}

/*******************************************************/
/*@@@@@函数名:GenAnalyzeImage@@@@@@@*/
/*函数说明：从原图中提取视场角分析图像，即左上至右下与左下至右上像素图和两个像素图的一维差分图*/
/*输入：*/
/*matGrayImage 类型：Mat 说明：输入的分析图像即Y通道图像*/
/*matHImage 类型：Mat 说明：分析图像中左上至右下的像素图*/
/*matHImage1 类型：Mat 说明：分析图像中左下至右上的像素图*/
/*matDiffHImage 类型：Mat 说明：左上至右下像素图的一维差分图，即matHImage的差分图*/
/*matDiffHImage1 类型：Mat 说明：左下至右上像素图的一维差分图，即matHImage1的差分图*/
/*输出：空*/
/******************************************************/

void GenAnalyzeImage(cv::Mat& matGrayImage, cv::Mat& matHImage, cv::Mat matHImage1, cv::Mat& matDiffHImage, cv::Mat& matDiffHImage1) {
	double dWidth, dHeight;
	int nWFloor, nHFloor, nWCeil, nHCeil;
	int nWFloor1, nHFloor1, nWCeil1, nHCeil1;

	double* pdmatHImge = matHImage.ptr<double>(0);
	double* pdmatHImage1 = matHImage1.ptr<double>(0);
	double* pdmatDiffHImage = matDiffHImage.ptr<double>(0);
	double* pdmatDiffHImage1 = matDiffHImage1.ptr<double>(0);

	//uchar ucLeftTop, ucRightTop, ucLeftBottom, ucRightBottom;

	int nHalfW = matHImage.cols / 2;

	for (int n = 0; n < matHImage.cols; n++) {
		dWidth = n * cdWidthRatio;
		dHeight = n * cdHeightRatio;

		nWFloor = static_cast<int>(dWidth);
		nHFloor = static_cast<int>(dHeight);
		nWCeil = nWFloor + 1;
		nHCeil = nHFloor + 1;

		nWFloor1 = static_cast<int>(1280 - dWidth);
		nHFloor1 = static_cast<int>(dHeight);
		nWCeil1 = nWFloor1 + 1;
		nHCeil1 = nHFloor1 + 1;

		if (nWCeil >= 1280) {
			nWCeil = 1279;
		}
		if (nHCeil >= 720) {
			nHCeil = 719;
		}

		if (nWCeil1 < 0) {
			nWCeil1 = 0;
		}
		if (nHCeil1 >= 720) {
			nHCeil1 = 719;
		}

		if (matGrayImage.type() == 0) {
			uchar* pucTopPtr, * pucBottomPtr;
			uchar* pucTopPtr1, * pucBottomPtr1;

			pucTopPtr = matGrayImage.ptr<uchar>(nHFloor);
			pucBottomPtr = matGrayImage.ptr<uchar>(nHCeil);
			pucTopPtr1 = matGrayImage.ptr<uchar>(nHFloor1);
			pucBottomPtr1 = matGrayImage.ptr<uchar>(nHCeil1);

			//ofstream fout("Params.txt", ios_base::out);
			//fout << __LINE__ << endl;
			//fout.close();

			pdmatHImge[n] = (pucTopPtr[nWFloor] + pucTopPtr[nWCeil] + pucBottomPtr[nWFloor] + pucBottomPtr[nWCeil]) / 4.0;
			pdmatHImage1[n] = (pucTopPtr1[nWFloor1] + pucTopPtr1[nWCeil1] + pucBottomPtr1[nWFloor1] + pucBottomPtr1[nWCeil1]) / 4.0;
		}
		else if (matGrayImage.type() == 5) {
			float* pdTopPtr, * pdBottomPtr;
			float* pdTopPtr1, * pdBottomPtr1;

			pdTopPtr = matGrayImage.ptr<float>(nHFloor);
			pdBottomPtr = matGrayImage.ptr<float>(nHCeil);
			pdTopPtr1 = matGrayImage.ptr<float>(nHFloor1);
			pdBottomPtr1 = matGrayImage.ptr<float>(nHCeil1);

			pdmatHImge[n] = (pdTopPtr[nWFloor] + pdTopPtr[nWCeil] + pdBottomPtr[nWFloor] + pdBottomPtr[nWCeil]) / 4.0;
			pdmatHImage1[n] = (pdTopPtr1[nWFloor1] + pdTopPtr1[nWCeil1] + pdBottomPtr1[nWFloor1] + pdBottomPtr1[nWCeil1]) / 4.0;
		}

		//if (n) {
		//	pdmatDiffHImage[n - 1] = pdmatHImge[n] - pdmatHImge[n - 1];
		//	pdmatDiffHImage1[n - 1] = pdmatHImage1[n] - pdmatHImage1[n - 1];
		//}
	}

	cv::Mat matFilterImage, matFilterImage1;
	Filter1DImage(matHImage, matFilterImage, 5);
	Filter1DImage(matHImage1, matFilterImage1, 5);

	double* pdmatFilterImge = matFilterImage.ptr<double>(0);
	double* pdmatFilterImage1 = matFilterImage1.ptr<double>(0);

	for (int n = 0; n < matHImage.cols; n++) {
		if (n) {
			pdmatDiffHImage[n - 1] = pdmatFilterImge[n] - pdmatFilterImge[n - 1];
			pdmatDiffHImage1[n - 1] = pdmatFilterImage1[n] - pdmatFilterImage1[n - 1];
		}
	}
}



/*******************************************************/
/*@@@@@函数名:ThreshDiffImage@@@@@@@*/
/*函数说明：对滤波后的平均差分图执行阈值处理，对差分波动量小于dThreshDark和dThreshBrightness的像素点赋值-100 */
/*检测屏蔽了四角90像素图像。且因为成像在镜头大视场情况下，照度与清晰度都会下降。*/
/*检测对四角140像素范围内的图像，执行不同参数的阈值处理。即大于四角240像素的图像用dThreshDark进行阈值处理。*/
/*小于四角140像素的图像用dThreshBrightness进行阈值处理。*/
/*输入：*/
/*matDiffHImage 类型：Mat 说明：输入的左上至右下差分均值图*/
/*matDiffHImage1 类型：Mat 说明：输入的左下至右上差分均值图*/
/*matFilterDiffHImage 类型：Mat 说明：输出的左上至右下阈值图*/
/*matFilterDiffHImage1 类型：Mat 说明：输出的左下至右上阈值图*/
/*nFilterSz 类型：int 说明：保留参数暂时无用*/
/*nBoundary 类型：int 说明：保留参数暂时无用*/
/*dThreshDark 类型：double 说明：暗场阈值，即大于四边240像素图像的阈值*/
/*dThreshBrightness 类型：double 说明：亮场阈值，即处于四边240像素内图像的阈值*/
/*输出：空*/
/******************************************************/

void ThreshDiffImage(cv::Mat& matDiffHImage, cv::Mat& matDiffHImage1, cv::Mat& matFilterDiffHImage, cv::Mat& matFilterDiffHImage1, int nFilterSz, int nBoundary, double dThreshDark, double dThreshBrightness) {
	int nTempFilterSz = nFilterSz / 2 + 1;
	int nHalfFilterSz = nTempFilterSz / 2;

	int c, r;
	int nSumC;
	double dSum, dSum1;
	double dSigma, dSigma1;

	double* pdmatDiffHImage = matDiffHImage.ptr<double>(0), * pdmatDiffHImage1 = matDiffHImage1.ptr<double>(0);
	double* pdmatFilterDiffHImage = matFilterDiffHImage.ptr<double>(0), * pdmatFilterDiffHImage1 = matFilterDiffHImage1.ptr<double>(0);
	//cv::Mat matSigmaImage = cv::Mat(matDiffHImage.rows, matDiffHImage.cols, CV_64FC1);
	//double* pdmatSigmaImage = matSigmaImage.ptr<double>(0);

	cv::Mat matSigmaImageLefttopToRightbuttom = cv::Mat::zeros(1, matDiffHImage.cols, CV_64FC1);
	cv::Mat matSigmaImageRighttopToLeftbuttom = cv::Mat::zeros(1, matDiffHImage.cols, CV_64FC1);
	double *pmatSigmaImageLefttopToRightbuttom = matSigmaImageLefttopToRightbuttom.ptr<double>(0);
	double *pmatSigmaImageRighttopToLeftbuttom = matSigmaImageRighttopToLeftbuttom.ptr<double>(0);

	for (c = 0; c < matDiffHImage.cols; c++) {
		if (c < 90|| c >= matDiffHImage.cols - 90) {
			pdmatFilterDiffHImage1[c] = -100.0;
			pdmatFilterDiffHImage[c] = -100.0;
		}
		else {
			dSum = 0.0;
			dSum1 = 0.0;
			nSumC = 0;

			dSigma = 0.0;
			dSigma1 = 0.0;
			for (r = -nHalfFilterSz; r < nHalfFilterSz; r++) {
				if (c + r >= 0 && c + r < matDiffHImage.cols) {
					dSum += pdmatDiffHImage[c + r];
					dSum1 += pdmatDiffHImage1[c + r];
					nSumC++;
				}
			}
			dSum /= nSumC;
			dSum1 /= nSumC;

			for (r = -nHalfFilterSz; r < nHalfFilterSz; r++) {
				if (c + r >= 0 && c + r < matDiffHImage.cols) {
					dSigma += (pdmatDiffHImage[c + r] - dSum) * (pdmatDiffHImage[c + r] - dSum);
					dSigma1 += (pdmatDiffHImage1[c + r] - dSum1) * (pdmatDiffHImage1[c + r] - dSum1);
					nSumC++;
				}
			}

			//pdmatSigmaImage[c] = dSigma / nSumC;
			if (c < 240 || c > matDiffHImage.cols - 240) {
				if (dSigma / nSumC > dThreshDark) {
					pdmatFilterDiffHImage[c] = pdmatDiffHImage[c];
				}
				else {
					pdmatFilterDiffHImage[c] = -100.0;
				}

				if (dSigma1 / nSumC > dThreshDark) {
					pdmatFilterDiffHImage1[c] = pdmatDiffHImage1[c];
				}
				else {
					pdmatFilterDiffHImage1[c] = -100.0;
				}

				pmatSigmaImageLefttopToRightbuttom[c] = dSigma / nSumC;
				pmatSigmaImageRighttopToLeftbuttom[c] = dSigma1 / nSumC;
			}
			else {
				if (dSigma / nSumC > dThreshBrightness) {
					pdmatFilterDiffHImage[c] = pdmatDiffHImage[c];
				}
				else {
					pdmatFilterDiffHImage[c] = -100.0;
				}

				if (dSigma1 / nSumC > dThreshBrightness) {
					pdmatFilterDiffHImage1[c] = pdmatDiffHImage1[c];
				}
				else {
					pdmatFilterDiffHImage1[c] = -100.0;
				}

				pmatSigmaImageLefttopToRightbuttom[c] = dSigma / nSumC;
				pmatSigmaImageRighttopToLeftbuttom[c] = dSigma1 / nSumC;
			}
		}
	}
}

void FigureAngle(cv::Mat& matAngleImage, int& nLeftA, int& nRightA) {
	nLeftA = 0;
	nRightA = 0;

	vector<int>vnLeftSteepPos;
	vnLeftSteepPos.reserve(20);
	vector<int>vnRightSteepPos;
	vnRightSteepPos.reserve(20);
	vector<int>vnLeftAlterSteepPos;
	vector<int>vnRightAlterSteepPos;

	int c;
	int nHalfWidth = matAngleImage.cols / 2;
	//double dFormerV;
	double* pdmatAngleImage = matAngleImage.ptr<double>(0);
	for (c = 0; c < matAngleImage.cols; c++) {
		if (c >= 5 && c < nHalfWidth - 80) {
			if (pdmatAngleImage[c - 1] == -100.0 && pdmatAngleImage[c] > -100.0) {
				vnLeftSteepPos.push_back(c);
			}
		}

		if (c > nHalfWidth + 80 && c < matAngleImage.cols - 5) {
			if (pdmatAngleImage[c - 1] == -100.0 && pdmatAngleImage[c] > -100.0) {
				vnRightSteepPos.push_back(c);
			}
		}
	}

	if (vnLeftSteepPos.size() && vnRightSteepPos.size()) {
		vnLeftAlterSteepPos.reserve(vnLeftSteepPos.size());
		vnRightAlterSteepPos.reserve(vnRightSteepPos.size());

		int n;
		for (n = vnLeftSteepPos.size() - 1; n >= 1; n--) {
			if (vnLeftSteepPos[n] - vnLeftSteepPos[n - 1] > 25) {
				vnLeftAlterSteepPos.push_back(vnLeftSteepPos[n]);
			}
		}
		vnLeftAlterSteepPos.push_back(vnLeftSteepPos[0]);
		nLeftA = vnLeftAlterSteepPos.size();

		for (n = vnRightSteepPos.size() - 1; n >= 1; n--) {
			if (vnRightSteepPos[n] - vnRightSteepPos[n - 1] > 25) {
				vnRightAlterSteepPos.push_back(vnRightSteepPos[n]);
			}
		}
		vnRightAlterSteepPos.push_back(vnRightSteepPos[0]);
		nRightA = vnRightAlterSteepPos.size();
	}
}

/*******************************************************/
/*@@@@@函数名:FigureFOV@@@@@@@*/
/*函数说明：计算视场角，根据采集图像中左上至右下和右上至左下对角线像素包含明暗线对数来确定视场角角度。*/
/*输入：*/
/*matYImage 类型：Mat 说明：输入的检测图像，只包含Y通道*/
/*nLeftA0 类型：int& 说明：输出的计算结果，包含通过左上角图像计算出来的视场角结果*/
/*nRightA0 类型：int& 说明：输出的计算结果，包含通过右下角图像计算出来的视场角结果*/
/*nLeftA1 类型：int& 说明：输出的计算结果，包含通过左下角图像计算出来的视场角结果*/
/*nRightA1 类型：int& 说明：输出的计算结果，包含通过右上角图像计算出来的视场角结果*/
/*输出：空*/
/******************************************************/

void FigureFOV(cv::Mat matYImage, int& nLeftA0, int& nRightA0, int& nLeftA1, int& nRightA1) {
	int nL = sqrt(matYImage.cols * matYImage.cols + matYImage.rows * matYImage.rows);
	//cv::threshold(matGrayImage, matThreshImage, 0, 255, cv::THRESH_OTSU);
	cv::Mat matHImage = cv::Mat::zeros(1, nL, CV_64FC1);
	cv::Mat matHImage1 = cv::Mat::zeros(1, nL, CV_64FC1);
	cv::Mat matDiffHImage = cv::Mat::zeros(1, nL - 1, CV_64FC1);
	cv::Mat matDiffHImage1 = cv::Mat::zeros(1, nL - 1, CV_64FC1);
	cv::Mat matDiffFilterHImage = cv::Mat::zeros(1, nL - 1, CV_64FC1);
	cv::Mat matDiffFilterHImage1 = cv::Mat::zeros(1, nL - 1, CV_64FC1);

	cv::Mat matMeanDiffHImage, matMeanDiffHImage1;
	//生成分析图像，即从原图中提取出左上至右下和右上至左下对角线像素，并且计算两个对角线图像的差分图像。
	GenAnalyzeImage(matYImage, matHImage, matHImage1, matDiffHImage, matDiffHImage1);
	//对左上至右下像素差分图执行平均滤波执行驱噪
	Filter1DImage(matDiffHImage, matMeanDiffHImage, 5);
	//对右上至左下像素差分图执行平均滤波执行驱噪
	Filter1DImage(matDiffHImage1, matMeanDiffHImage1, 5);

	//cv::imwrite("matDiffHImage.bmp", matDiffHImage);
	//cv::imwrite("matDiffHImage1.bmp", matDiffHImage1);

	//对左上至右下和右上至左下像素差分滤波图执行阈值处理
	ThreshDiffImage(matMeanDiffHImage, matMeanDiffHImage1, matDiffFilterHImage, matDiffFilterHImage1, 9, 240, 1.2, 6.0);
	//MeanDiffImage(matHImage, matHImage1, matDiffHImage, matDiffHImage1);

	//根据左上至右下像素差分阈值图计算左上与右下视场角值
	FigureAngle(matDiffFilterHImage, nLeftA0, nRightA0);
	//根据右上至左下像素差分阈值图计算右上与左下视场角值
	FigureAngle(matDiffFilterHImage1, nLeftA1, nRightA1);
}

void FigureMeanV(cv::Mat& matSubArea1, double& dCenterV) {
	int r, c;

	dCenterV = 0.0;
	if (matSubArea1.type() == 0) {
		uchar* pucmatSubArea;
		for (r = 0; r < matSubArea1.rows; r++) {
			pucmatSubArea = matSubArea1.ptr<uchar>(r);
			for (c = 0; c < matSubArea1.cols; c++) {
				dCenterV += pucmatSubArea[c];
			}
		}
	}
	else if (matSubArea1.type() == 5) {
		float* pfmatSubArea;
		for (r = 0; r < matSubArea1.rows; r++) {
			pfmatSubArea = matSubArea1.ptr<float>(r);
			for (c = 0; c < matSubArea1.cols; c++) {
				dCenterV += pfmatSubArea[c];
			}
		}
	}
	dCenterV /= matSubArea1.rows * matSubArea1.cols;
}

void FigureBrightnessRatio(cv::Mat& matSrcImage, double dThresh, double& dBrightnessRatio) {
	int nTotalSz = matSrcImage.rows * matSrcImage.cols;

	int nBrightnessNum = 0;
	if (matSrcImage.type() == 0) {
		uchar* pucmatSrcImage = matSrcImage.ptr<uchar>(0);
		for (int n = 0; n < nTotalSz; n++) {
			if (pucmatSrcImage[n] > dThresh) {
				nBrightnessNum++;
			}
		}
		dBrightnessRatio = nBrightnessNum / static_cast<double>(nTotalSz) * 100.0;
	}
	else if (matSrcImage.type() == 5) {
		float* pfmatSrcImage = matSrcImage.ptr<float>(0);
		for (int n = 0; n < nTotalSz; n++) {
			if (pfmatSrcImage[n] > dThresh) {
				nBrightnessNum++;
			}
		}
		dBrightnessRatio = nBrightnessNum / static_cast<double>(nTotalSz) * 100.0;
	}
}

void FigureBrightnessRatio(cv::Mat& matSrcImage, cv::Mat& matMaskImage, double dThresh, double& dBrightnessRatio) {
	int nTotalSz = matSrcImage.rows * matSrcImage.cols;

	int nBrightnessNum = 0;
	if (matSrcImage.type() == 0) {
		uchar* pucmatSrcImage = matSrcImage.ptr<uchar>(0);
		uchar* pucmatMaskImage = matMaskImage.ptr<uchar>(0);
		for (int n = 0; n < nTotalSz; n++) {
			if (pucmatSrcImage[n] > dThresh && !pucmatMaskImage[n]) {
				nBrightnessNum++;
			}
		}
		dBrightnessRatio = nBrightnessNum / static_cast<double>(nTotalSz) * 100.0;
	}
	else if (matSrcImage.type() == 5) {
		float* pfmatSrcImage = matSrcImage.ptr<float>(0);
		uchar* pucmatMaskImage = matMaskImage.ptr<uchar>(0);
		for (int n = 0; n < nTotalSz; n++) {
			if (pfmatSrcImage[n] > dThresh && !pucmatMaskImage[n]) {
				nBrightnessNum++;
			}
		}
		dBrightnessRatio = nBrightnessNum / static_cast<double>(nTotalSz) * 100.0;
	}
}



void FigureParameters(uchar* pucImagePtr, int nImageW, int nImageH, int nFrameNo, double dRatio, int nEnFilter, double dBRThresh, int nEnDefinition, int nEnFov, int nEnBR, int nEnUnformity) {

	nLeftPosFOV = 0, nRightPosFOV = 0;
	nLeftNegFOV = 0, nRightNegFOV = 0;

	cv::Mat matMaskImage = cv::Mat::zeros(nImageH, nImageW, CV_8UC1);
	cv::rectangle(matMaskImage, cv::Rect(0, 0, 200, 200), cv::Scalar(255.0, 255.0, 255.0), -1);
	cv::rectangle(matMaskImage, cv::Rect(nImageW - 200, nImageH - 200, 200, 200), cv::Scalar(255.0, 255.0, 255.0), -1);
	cv::rectangle(matMaskImage, cv::Rect(0, nImageH - 200, 200, 200), cv::Scalar(255.0, 255.0, 255.0), -1);
	cv::rectangle(matMaskImage, cv::Rect(nImageW - 200, 0, 200, 200), cv::Scalar(255.0, 255.0, 255.0), -1);

	cv::circle(matMaskImage, cv::Point(200, 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	cv::circle(matMaskImage, cv::Point(nImageW - 200, 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	cv::circle(matMaskImage, cv::Point(200, nImageH - 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	cv::circle(matMaskImage, cv::Point(nImageW - 200, nImageH - 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	//cv::imwrite("MaskImage.bmp", matMaskImage);

	//这段if语句包含的代码包含时域滤波功能。
	if (nEnFilter) {
		if (nFrameNo == 0) {
			*pmatPreImage = cv::Mat(nImageH, nImageW, CV_8UC1);
			//从YUY2格式的图像中提取Y分量。
			convertYUY2ToY(pucImagePtr, nImageW, nImageH, pmatPreImage->ptr<uchar>(0));
			dFormerRatio = dRatio;
		}
		else {
			cv::Mat matYImage = cv::Mat(nImageH, nImageW, CV_8UC1);
			cv::Mat matFloatY = cv::Mat(nImageH, nImageW, CV_32FC1);

			//if (fabs(dRatio - dFormerRatio) > 1e-6) {
			//    ReadParam(dRatio);
			//}

			//从YUY2格式的图像中提取Y分量
			convertYUY2ToY(pucImagePtr, nImageW, nImageH, matYImage.ptr<uchar>(0));

			MANRFilterCPU(matYImage.ptr<uchar>(0), pmatPreImage->ptr<uchar>(0), matFloatY.ptr<float>(0), nImageW, nImageH, dRatio);

			cv::Mat matSubArea1, matSubArea2, matSubArea3, matSubArea4, matSubArea5, matSubArea6, matSubArea7, matSubArea8, matSubArea9;

			matSubArea1 = matFloatY(cv::Rect(0.15 * nImageW, 0.2 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
			matSubArea2 = matFloatY(cv::Rect(0.45 * nImageW, 0.2 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
			matSubArea3 = matFloatY(cv::Rect(0.75 * nImageW, 0.2 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
			matSubArea4 = matFloatY(cv::Rect(0.15 * nImageW, 0.5 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
			matSubArea5 = matFloatY(cv::Rect(0.45 * nImageW, 0.5 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
			matSubArea6 = matFloatY(cv::Rect(0.75 * nImageW, 0.5 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
			matSubArea7 = matFloatY(cv::Rect(0.15 * nImageW, 0.8 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
			matSubArea8 = matFloatY(cv::Rect(0.45 * nImageW, 0.8 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
			matSubArea9 = matFloatY(cv::Rect(0.75 * nImageW, 0.8 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));

			if (nEnDefinition) {
				FigurebrennerFloat(matFloatY, dBrenner);
				FigureSobelFloat(matFloatY, dTenegrad);
				FigureLaplacianFloat(matFloatY, dLaplacian);

				FigurebrennerFloat(matSubArea1, pdSubAreaBrenner[0]);
				FigureSobelFloat(matSubArea1, pdSubAreaTenegrad[0]);
				FigureLaplacianFloat(matSubArea1, pdSubAreaLaplacian[0]);
				FigurebrennerFloat(matSubArea2, pdSubAreaBrenner[1]);
				FigureSobelFloat(matSubArea2, pdSubAreaTenegrad[1]);
				FigureLaplacianFloat(matSubArea2, pdSubAreaLaplacian[1]);
				FigurebrennerFloat(matSubArea3, pdSubAreaBrenner[2]);
				FigureSobelFloat(matSubArea3, pdSubAreaTenegrad[2]);
				FigureLaplacianFloat(matSubArea3, pdSubAreaLaplacian[2]);
				FigurebrennerFloat(matSubArea4, pdSubAreaBrenner[3]);
				FigureSobelFloat(matSubArea4, pdSubAreaTenegrad[3]);
				FigureLaplacianFloat(matSubArea4, pdSubAreaLaplacian[3]);
				FigurebrennerFloat(matSubArea5, pdSubAreaBrenner[4]);
				FigureSobelFloat(matSubArea5, pdSubAreaTenegrad[4]);
				FigureLaplacianFloat(matSubArea5, pdSubAreaLaplacian[4]);
				FigurebrennerFloat(matSubArea6, pdSubAreaBrenner[5]);
				FigureSobelFloat(matSubArea6, pdSubAreaTenegrad[5]);
				FigureLaplacianFloat(matSubArea6, pdSubAreaLaplacian[5]);
				FigurebrennerFloat(matSubArea7, pdSubAreaBrenner[6]);
				FigureSobelFloat(matSubArea7, pdSubAreaTenegrad[6]);
				FigureLaplacianFloat(matSubArea7, pdSubAreaLaplacian[6]);
				FigurebrennerFloat(matSubArea8, pdSubAreaBrenner[7]);
				FigureSobelFloat(matSubArea8, pdSubAreaTenegrad[7]);
				FigureLaplacianFloat(matSubArea8, pdSubAreaLaplacian[7]);
				FigurebrennerFloat(matSubArea9, pdSubAreaBrenner[8]);
				FigureSobelFloat(matSubArea9, pdSubAreaTenegrad[8]);
				FigureLaplacianFloat(matSubArea9, pdSubAreaLaplacian[8]);
			}

			if (nEnFov) {
				FigureFOV(matFloatY, nLeftPosFOV, nRightNegFOV, nRightPosFOV, nLeftNegFOV);
			}

			if (nEnBR) {
				FigureBrightnessRatio(matFloatY, matMaskImage, dBRThresh, dBrightnessRatio);
			}

			if (nEnUnformity) {
				double dLTV, dCenterTV, dRTV, dLCenterV, dCenterV, dRCenterV, dLBV, dCenterBV, dRBV;
				FigureMeanV(matSubArea1, dLTV);
				FigureMeanV(matSubArea2, dCenterTV);
				FigureMeanV(matSubArea3, dRTV);
				FigureMeanV(matSubArea4, dLCenterV);
				FigureMeanV(matSubArea5, dCenterV);
				FigureMeanV(matSubArea6, dRCenterV);
				FigureMeanV(matSubArea7, dLBV);
				FigureMeanV(matSubArea8, dCenterBV);
				FigureMeanV(matSubArea9, dRBV);

				if (dCenterV < 0.001) {
					dCenterV = 0.001;
				}

				pdUnformityVs[0] = dLTV / dCenterV;
				pdUnformityVs[1] = dCenterTV / dCenterV;
				pdUnformityVs[2] = dRTV / dCenterV;
				pdUnformityVs[3] = dLCenterV / dCenterV;
				pdUnformityVs[4] = dRCenterV / dCenterV;
				pdUnformityVs[5] = dLBV / dCenterV;
				pdUnformityVs[6] = dCenterBV / dCenterV;
				pdUnformityVs[7] = dRBV / dCenterV;
			}
		}
	}
	//这段else语句包含的代码不包含时域滤波功能。
	else {
		cv::Mat matYImage = cv::Mat(nImageH, nImageW, CV_8UC1);
		//从YUY2格式的图像中提取Y分量
		convertYUY2ToY(pucImagePtr, nImageW, nImageH, matYImage.ptr<uchar>(0));


		Figurebrenner(matYImage, dBrenner);
		FigureSobel(matYImage, dTenegrad);
		FigureLaplacian(matYImage, dLaplacian);


		cv::Mat matSubArea1, matSubArea2, matSubArea3, matSubArea4, matSubArea5,
			matSubArea6, matSubArea7, matSubArea8, matSubArea9;



		//cv::imwrite("matYImage.bmp", matYImage);

		//ofstream foutLog("LOG.txt", ios_base::out);
		//foutLog << dBrightnessRatio << endl;
		//foutLog.close();

		matSubArea1 = matYImage(cv::Rect(0.15 * nImageW, 0.2 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
		matSubArea2 = matYImage(cv::Rect(0.45 * nImageW, 0.2 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
		matSubArea3 = matYImage(cv::Rect(0.75 * nImageW, 0.2 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
		matSubArea4 = matYImage(cv::Rect(0.15 * nImageW, 0.5 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
		matSubArea5 = matYImage(cv::Rect(0.45 * nImageW, 0.5 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
		matSubArea6 = matYImage(cv::Rect(0.75 * nImageW, 0.5 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
		matSubArea7 = matYImage(cv::Rect(0.15 * nImageW, 0.8 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
		matSubArea8 = matYImage(cv::Rect(0.45 * nImageW, 0.8 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));
		matSubArea9 = matYImage(cv::Rect(0.75 * nImageW, 0.8 * nImageH - 0.05 * nImageW, 0.1 * nImageW, 0.1 * nImageW));

		if (nEnDefinition) {
			Figurebrenner(matSubArea1, pdSubAreaBrenner[0]);
			FigureSobel(matSubArea1, pdSubAreaTenegrad[0]);
			FigureLaplacian(matSubArea1, pdSubAreaLaplacian[0]);
			Figurebrenner(matSubArea2, pdSubAreaBrenner[1]);
			FigureSobel(matSubArea2, pdSubAreaTenegrad[1]);
			FigureLaplacian(matSubArea2, pdSubAreaLaplacian[1]);
			Figurebrenner(matSubArea3, pdSubAreaBrenner[2]);
			FigureSobel(matSubArea3, pdSubAreaTenegrad[2]);
			FigureLaplacian(matSubArea3, pdSubAreaLaplacian[2]);
			Figurebrenner(matSubArea4, pdSubAreaBrenner[3]);
			FigureSobel(matSubArea4, pdSubAreaTenegrad[3]);
			FigureLaplacian(matSubArea4, pdSubAreaLaplacian[3]);
			Figurebrenner(matSubArea5, pdSubAreaBrenner[4]);
			FigureSobel(matSubArea5, pdSubAreaTenegrad[4]);
			FigureLaplacian(matSubArea5, pdSubAreaLaplacian[4]);
			Figurebrenner(matSubArea6, pdSubAreaBrenner[5]);
			FigureSobel(matSubArea6, pdSubAreaTenegrad[5]);
			FigureLaplacian(matSubArea6, pdSubAreaLaplacian[5]);
			Figurebrenner(matSubArea7, pdSubAreaBrenner[6]);
			FigureSobel(matSubArea7, pdSubAreaTenegrad[6]);
			FigureLaplacian(matSubArea7, pdSubAreaLaplacian[6]);
			Figurebrenner(matSubArea8, pdSubAreaBrenner[7]);
			FigureSobel(matSubArea8, pdSubAreaTenegrad[7]);
			FigureLaplacian(matSubArea8, pdSubAreaLaplacian[7]);
			Figurebrenner(matSubArea9, pdSubAreaBrenner[8]);
			FigureSobel(matSubArea9, pdSubAreaTenegrad[8]);
			FigureLaplacian(matSubArea9, pdSubAreaLaplacian[8]);

		}
		if (nEnFov) {
			
			FigureFOV(matYImage, nLeftPosFOV, nRightNegFOV, nRightPosFOV, nLeftNegFOV);
		}
		if (nEnBR) {
			FigureBrightnessRatio(matYImage, matMaskImage, dBRThresh, dBrightnessRatio);
		}

		double dLTV, dCenterTV, dRTV, dLCenterV, dCenterV, dRCenterV, dLBV, dCenterBV, dRBV;
		FigureMeanV(matSubArea1, dLTV);
		FigureMeanV(matSubArea2, dCenterTV);
		FigureMeanV(matSubArea3, dRTV);
		FigureMeanV(matSubArea4, dLCenterV);
		FigureMeanV(matSubArea5, dCenterV);
		FigureMeanV(matSubArea6, dRCenterV);
		FigureMeanV(matSubArea7, dLBV);
		FigureMeanV(matSubArea8, dCenterBV);
		FigureMeanV(matSubArea9, dRBV);

		pdUnformityVs[0] = dLTV / dCenterV;
		pdUnformityVs[1] = dCenterTV / dCenterV;
		pdUnformityVs[2] = dRTV / dCenterV;
		pdUnformityVs[3] = dLCenterV / dCenterV;
		pdUnformityVs[4] = dRCenterV / dCenterV;
		pdUnformityVs[5] = dLBV / dCenterV;
		pdUnformityVs[6] = dCenterBV / dCenterV;
		pdUnformityVs[7] = dRBV / dCenterV;
	}
}

void ConverYUY2ToYUV(uchar* pucImagePtr, cv::Mat& matY, cv::Mat& matU, cv::Mat& matV, int nImageW, int nImageH) {
	matY = cv::Mat(nImageH, nImageW, CV_8UC1);
	matU = cv::Mat(nImageH, nImageW, CV_8UC1);
	matV = cv::Mat(nImageH, nImageW, CV_8UC1);

	uchar* pucProcessImgY, * pucProcessImgU, * pucProcessImgV;
	pucProcessImgY = matY.ptr<uchar>(0);
	pucProcessImgU = matU.ptr<uchar>(0);
	pucProcessImgV = matV.ptr<uchar>(0);


	int m = 0;
	for (int n = 0; n < nImageW * nImageH; n++) {
		pucProcessImgY[n] = pucImagePtr[2 * n];

		m = n / 2;
		pucProcessImgU[n] = pucImagePtr[4 * m + 1];
		pucProcessImgV[n] = pucImagePtr[4 * m + 3];
	}
}

void ConverYUVToYUY2(cv::Mat& matY, cv::Mat& matU, cv::Mat& matV, uchar* pucImagePtr, int nImageW, int nImageH) {
	uchar* pucProcessImgY, * pucProcessImgU, * pucProcessImgV;
	pucProcessImgY = matY.data;
	pucProcessImgU = matU.data;
	pucProcessImgV = matV.data;

	int m = 0;
	for (int n = 0; n < nImageW * nImageH; n++) {
		pucImagePtr[2 * n] = pucProcessImgY[n];

		m = n / 2;
		if (!(n % 2)) {
			pucImagePtr[4 * m + 1] = pucProcessImgU[n];
			pucImagePtr[4 * m + 3] = pucProcessImgV[n];
		}
	}
}

void ConverYUVToYUY2(cv::Mat& matYUV, uchar* pucImagePtr, int nImageW, int nImageH) {
	cv::Vec3b* pv3bYUVImage = matYUV.ptr<cv::Vec3b>(0);

	int m = 0;
	for (int n = 0; n < nImageW * nImageH; n++) {
		pucImagePtr[2 * n] = pv3bYUVImage[n][0];

		m = n / 2;
		if (!(n % 2)) {
			pucImagePtr[4 * m + 1] = pv3bYUVImage[n][1];
			pucImagePtr[4 * m + 3] = pv3bYUVImage[n][2];
		}
	}
}

void MaskImage(cv::Mat& matRGBImage, cv::Mat& matMaskImage) {
	cv::Vec3b* pv3bmatRGBImage = matRGBImage.ptr<cv::Vec3b>(0);
	cv::Vec3b* pv3bmatMaskImage = matMaskImage.ptr<cv::Vec3b>(0);

	for (int n = 0; n < matRGBImage.rows * matRGBImage.cols; n++) {
		if (pv3bmatMaskImage[n][0] || pv3bmatMaskImage[n][1] || pv3bmatMaskImage[n][2]) {
			pv3bmatRGBImage[n] = pv3bmatMaskImage[n];
		}
	}
}

void FigureCross(uchar* pucImagePtr, int nImageW, int nImageH) {
	//std::ofstream fout("Log.txt", std::ios_base::out);
	vector<cv::Mat>vmatYUVImage;
	vmatYUVImage.resize(3);
	//fout << __LINE__ << endl;
	ConverYUY2ToYUV(pucImagePtr, vmatYUVImage[0], vmatYUVImage[1], vmatYUVImage[2], nImageW, nImageH);
	cv::Mat matYUVImage, matRGBImage;
	cv::merge(vmatYUVImage, matYUVImage);
	//fout << __LINE__ << endl;
	cv::cvtColor(matYUVImage, matRGBImage, cv::COLOR_YUV2BGR);
	//fout << __LINE__ << endl;
	cv::line(matRGBImage, cv::Point(matRGBImage.cols / 2, matRGBImage.rows / 2 - 20), cv::Point(matRGBImage.cols / 2, matRGBImage.rows / 2 + 20), cv::Scalar(255.0, 255.0, 0.0), 2);
	cv::line(matRGBImage, cv::Point(matRGBImage.cols / 2 - 20, matRGBImage.rows / 2), cv::Point(matRGBImage.cols / 2 + 20, matRGBImage.rows / 2), cv::Scalar(255.0, 255.0, 0.0), 2);
	//fout << __LINE__ << endl;
	//cv::Mat matMaskImage = cv::Mat::zeros(nImageH, nImageW, CV_8UC3);
	//cv::rectangle(matMaskImage, cv::Rect(0, 0, 200, 200), cv::Scalar(0.0, 255.0, 255.0), -1);
	//cv::rectangle(matMaskImage, cv::Rect(nImageW - 200, nImageH - 200, 200, 200), cv::Scalar(0.0, 255.0, 255.0), -1);
	//cv::rectangle(matMaskImage, cv::Rect(0, nImageH - 200, 200, 200), cv::Scalar(0.0, 255.0, 255.0), -1);
	//cv::rectangle(matMaskImage, cv::Rect(nImageW - 200, 0, 200, 200), cv::Scalar(0.0, 255.0, 255.0), -1);

	//cv::circle(matMaskImage, cv::Point(200, 200), 200,cv::Scalar(0.0, 0.0, 0.0), -1);
	//cv::circle(matMaskImage, cv::Point(nImageW - 200, 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	//cv::circle(matMaskImage, cv::Point(200, nImageH - 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	//cv::circle(matMaskImage, cv::Point(nImageW - 200, nImageH - 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);

	//MaskImage(matRGBImage, matMaskImage);

	cv::cvtColor(matRGBImage, matYUVImage, cv::COLOR_BGR2YUV);

	//fout << __LINE__ << endl;
	//fout.close();
	cv::split(matYUVImage, vmatYUVImage);

	cv::Mat matUCYImage, matUCUImage, matUCVImage;

	//vmatYUVImage[0].convertTo(matUCYImage, CV_8UC1);
	//vmatYUVImage[1].convertTo(matUCUImage, CV_8UC1);
	//vmatYUVImage[2].convertTo(matUCVImage, CV_8UC1);
	//fout << __LINE__ << endl;
	ConverYUVToYUY2(vmatYUVImage[0], vmatYUVImage[1], vmatYUVImage[2], pucImagePtr, nImageW, nImageH);
	// fout << __LINE__ << endl;
}

void FigureCorners(uchar* pucImagePtr, int nImageW, int nImageH) {
	//std::ofstream fout("Log.txt", std::ios_base::out);
	vector<cv::Mat>vmatYUVImage;
	vmatYUVImage.resize(3);
	//fout << __LINE__ << endl;
	ConverYUY2ToYUV(pucImagePtr, vmatYUVImage[0], vmatYUVImage[1], vmatYUVImage[2], nImageW, nImageH);
	cv::Mat matYUVImage, matRGBImage;
	cv::merge(vmatYUVImage, matYUVImage);
	//fout << __LINE__ << endl;
	cv::cvtColor(matYUVImage, matRGBImage, cv::COLOR_YUV2BGR);
	//fout << __LINE__ << endl;
	//cv::line(matRGBImage, cv::Point(matRGBImage.cols / 2, matRGBImage.rows / 2 - 20), cv::Point(matRGBImage.cols / 2, matRGBImage.rows / 2 + 20), cv::Scalar(255.0, 255.0, 0.0), 2);
	//cv::line(matRGBImage, cv::Point(matRGBImage.cols / 2 - 20, matRGBImage.rows / 2), cv::Point(matRGBImage.cols / 2 + 20, matRGBImage.rows / 2), cv::Scalar(255.0, 255.0, 0.0), 2);
	//fout << __LINE__ << endl;
	cv::Mat matMaskImage = cv::Mat::zeros(nImageH, nImageW, CV_8UC3);
	cv::rectangle(matMaskImage, cv::Rect(0, 0, 200, 200), cv::Scalar(0.0, 255.0, 255.0), -1);
	cv::rectangle(matMaskImage, cv::Rect(nImageW - 200, nImageH - 200, 200, 200), cv::Scalar(0.0, 255.0, 255.0), -1);
	cv::rectangle(matMaskImage, cv::Rect(0, nImageH - 200, 200, 200), cv::Scalar(0.0, 255.0, 255.0), -1);
	cv::rectangle(matMaskImage, cv::Rect(nImageW - 200, 0, 200, 200), cv::Scalar(0.0, 255.0, 255.0), -1);

	cv::circle(matMaskImage, cv::Point(200, 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	cv::circle(matMaskImage, cv::Point(nImageW - 200, 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	cv::circle(matMaskImage, cv::Point(200, nImageH - 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);
	cv::circle(matMaskImage, cv::Point(nImageW - 200, nImageH - 200), 200, cv::Scalar(0.0, 0.0, 0.0), -1);

	MaskImage(matRGBImage, matMaskImage);

	cv::cvtColor(matRGBImage, matYUVImage, cv::COLOR_BGR2YUV);

	//fout << __LINE__ << endl;
	//fout.close();
	cv::split(matYUVImage, vmatYUVImage);

	cv::Mat matUCYImage, matUCUImage, matUCVImage;

	//vmatYUVImage[0].convertTo(matUCYImage, CV_8UC1);
	//vmatYUVImage[1].convertTo(matUCUImage, CV_8UC1);
	//vmatYUVImage[2].convertTo(matUCVImage, CV_8UC1);
	//fout << __LINE__ << endl;
	ConverYUVToYUY2(vmatYUVImage[0], vmatYUVImage[1], vmatYUVImage[2], pucImagePtr, nImageW, nImageH);
	// fout << __LINE__ << endl;
}

void FlipImage(uchar* pucImagePtr, int nImageW, int nImageH) {
	cv::Mat matSrcImage = cv::Mat(nImageH, nImageW, CV_8UC2, pucImagePtr);
	cv::Mat matFlipImage = cv::Mat(nImageH, nImageW, CV_8UC2);

	//cv::flip(matSrcImage, matFlipImage, -1);

	int r, c;
	cv::Vec4b* puc2SrcImage = matSrcImage.ptr<cv::Vec4b>(0);
	cv::Vec4b* puc2FlipImage = matFlipImage.ptr<cv::Vec4b>(0);
	for (int n = 0; n < matSrcImage.rows * (matSrcImage.cols / 2); n++) {
		r = n / (matSrcImage.cols / 2);
		c = n % (matSrcImage.cols / 2);

		puc2FlipImage[n][0] = puc2SrcImage[(720 - r) * (matSrcImage.cols / 2) + (640 - c)][2];
		puc2FlipImage[n][1] = puc2SrcImage[(720 - r) * (matSrcImage.cols / 2) + (640 - c)][1];
		puc2FlipImage[n][2] = puc2SrcImage[(720 - r) * (matSrcImage.cols / 2) + (640 - c)][0];
		puc2FlipImage[n][3] = puc2SrcImage[(720 - r) * (matSrcImage.cols / 2) + (640 - c)][3];
	}

	memcpy_s(pucImagePtr, nImageW * nImageH * 2, matFlipImage.ptr<uchar>(0), nImageW * nImageH * 2);
}

void GetMaxV(cv::Mat& matSrcImage, double& dMax, int nShieldR, int nShieldC) {
	int r, c;

	uchar* pucmatSrcImage;
	dMax = 0.0;
	for (r = nShieldR; r < matSrcImage.rows - nShieldR; r++) {
		pucmatSrcImage = matSrcImage.ptr<uchar>(r);
		for (c = nShieldC; c < matSrcImage.cols - nShieldC; c++) {
			if (pucmatSrcImage[c] > dMax) {
				dMax = pucmatSrcImage[c];
			}
		}
	}
}




void DestinationFunc(double* dEff, double* nX, double* nY, int nPNum, FResult& frTemp) {
	//frTemp.pdFormula = new double[nPNum];
	//frTemp.nCounter = nPNum;

	int n;
	for (n = 0; n < nPNum; n++) {
		frTemp.pdFormula[n] = dEff[2] * exp(-((nX[n] - dEff[0]) * (nX[n] - dEff[0]) / (2 * dEff[3] * dEff[3]) + (nY[n] - dEff[1]) * (nY[n] - dEff[1]) / (2 * dEff[4] * dEff[4])));
	}
}

void DestinationFunc(double* dEff, double* nX, double* nY, int nPNum, double* pdTemp) {
	//frTemp.pdFormula = new double[nPNum];
	//frTemp.nCounter = nPNum;

	int n;
	for (n = 0; n < nPNum; n++) {
		pdTemp[n] = dEff[2] * exp(-((nX[n] - dEff[0]) * (nX[n] - dEff[0]) / (2 * dEff[3] * dEff[3]) + (nY[n] - dEff[1]) * (nY[n] - dEff[1]) / (2 * dEff[4] * dEff[4])));
	}
}

void JacobianFunc(double* dEff, double* nX, double* nY, int nPNum, FResult& frTemp) {
	//frTemp.pdFormula = new double[5*nPNum];
	//frTemp.nCounter = 5*nPNum;
	int n;
	for (n = 0; n < nPNum; n++) {
		frTemp.pdFormula[n] = -(dEff[2] * exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4])) * (2 * dEff[0] - 2 * nX[n])) / (2 * dEff[3] * dEff[3]);
	}

	for (n = 0; n < nPNum; n++) {
		frTemp.pdFormula[nPNum + n] = -(dEff[2] * exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4])) * (2 * dEff[1] - 2 * nY[n])) / (2 * dEff[4] * dEff[4]);
	}

	for (n = 0; n < nPNum; n++) {
		frTemp.pdFormula[2 * nPNum + n] = exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4]));
	}

	for (n = 0; n < nPNum; n++) {
		frTemp.pdFormula[3 * nPNum + n] = (dEff[2] * exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4])) * ((dEff[0] - nX[n]) * (dEff[0] - nX[n]))) / (dEff[3] * dEff[3] * dEff[3]);
	}

	for (n = 0; n < nPNum; n++) {
		frTemp.pdFormula[4 * nPNum + n] = (dEff[2] * exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4])) * ((dEff[1] - nY[n]) * (dEff[1] - nY[n]))) / (dEff[4] * dEff[4] * dEff[4]);
	}
}

void JacobianFunc(double* dEff, double* nX, double* nY, int nPNum, double* pdTemp) {
	//frTemp.pdFormula = new double[5*nPNum];
	//frTemp.nCounter = 5*nPNum;


	int n;
	for (n = 0; n < nPNum; n++) {
		pdTemp[n] = -(dEff[2] * exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4])) * (2 * dEff[0] - 2 * nX[n])) / (2 * dEff[3] * dEff[3]);
	}


	for (n = 0; n < nPNum; n++) {
		pdTemp[nPNum + n] = -(dEff[2] * exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4])) * (2 * dEff[1] - 2 * nY[n])) / (2 * dEff[4] * dEff[4]);
	}

	for (n = 0; n < nPNum; n++) {
		pdTemp[2 * nPNum + n] = exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4]));
	}

	for (n = 0; n < nPNum; n++) {
		pdTemp[3 * nPNum + n] = (dEff[2] * exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4])) * ((dEff[0] - nX[n]) * (dEff[0] - nX[n]))) / (dEff[3] * dEff[3] * dEff[3]);
	}

	for (n = 0; n < nPNum; n++) {
		pdTemp[4 * nPNum + n] = (dEff[2] * exp(-((dEff[0] - nX[n]) * (dEff[0] - nX[n])) / (2 * dEff[3] * dEff[3]) - ((dEff[1] - nY[n]) * (dEff[1] - nY[n])) / (2 * dEff[4] * dEff[4])) * ((dEff[1] - nY[n]) * (dEff[1] - nY[n]))) / (dEff[4] * dEff[4] * dEff[4]);
	}
}

void DiffMat(cv::Mat& matInput, FResult& frJFData, FResult& frDeltaData) {

	uchar* pucmatInput = matInput.ptr<uchar>(0);
	for (int n = 0; n < matInput.rows * matInput.cols; n++) {
		frDeltaData.pdFormula[n] = pucmatInput[n] - frJFData.pdFormula[n];
	}
}

void MatSquare(FResult& frJFData, int nR, int nC, FResult& frResult) {
	int r, c;
	//int nn, nm;
	int n, m;
	//double dTemp;
	for (n = 0; n < frJFData.nCounter; n++) {
		r = n / nC;
		c = n % nC;

		for (m = 0; m < nR; m++) {
			frResult.pdFormula[r * nR + m] += frJFData.pdFormula[n] * frJFData.pdFormula[m * nC + c];
		}
	}
}

void NumMultiple(FResult& frMat1, double dLamda, FResult& frMat2) {
	for (int n = 0; n < frMat1.nCounter; n++) {
		frMat2.pdFormula[n] = dLamda * frMat1.pdFormula[n];
	}
}

void MatAdd(FResult& frMat1, FResult& frMat2, FResult& frMat3) {
	for (int n = 0; n < frMat1.nCounter; n++) {
		frMat3.pdFormula[n] = frMat1.pdFormula[n] + frMat2.pdFormula[n];
	}
}

void VectorMulti(FResult& frMat1, int nR, int nC, FResult& frMat2, FResult& frResult) {
	int m, n;


	for (m = 0; m < nR; m++) {
		for (n = 0; n < nC; n++) {
			frResult.pdFormula[m] += frMat1.pdFormula[m * nC + n] * frMat2.pdFormula[n];
		}
	}
}

void InvMatrix(FResult& FRE, int nN, FResult& InvFRE) {
	memcpy(InvFRE.pdFormula, FRE.pdFormula, sizeof(double) * nN * nN);
	int* ipiv = new int[nN];
	int nInfo = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, nN, nN, InvFRE.pdFormula, nN, ipiv);
	int nInfo1 = LAPACKE_dgetri(LAPACK_ROW_MAJOR, nN, InvFRE.pdFormula, nN, ipiv);
	delete[] ipiv;
}

void MatTranslate(double* pdMat1, int nR, int nC, double* pdMat2) {
	int r, c;

	for (r = 0; r < nR; r++) {
		for (c = 0; c < nC; c++) {
			pdMat2[c * nR + r] = pdMat1[r * nC + c];
		}
	}
}

void MatMulti(double* pdMat1, int nR, int nC, int nK, double* pdMat2, double* pdMat3) {
	int n, r, c, k;
	memset(pdMat3, 0, sizeof(double) * nR * nK);
	for (n = 0; n < nR * nC; n++) {
		r = n / nC;
		c = n % nC;
		for (k = 0; k < nK; k++) {
			pdMat3[r * nK + k] += pdMat1[n] * pdMat2[c * nK + k];
		}
	}
}

void DiagMat(double* pdD, int nN, double* pdDM) {

	int n;
	for (n = 0; n < nN; n++) {
		pdDM[n * nN + n] = pdD[n];
	}
}

void InvMatrix1(FResult& FRE, int nN, FResult& InvFRE) {
	//double* pdS = new double[nN];
	//double* pdInvS = new double[nN];
	//double* pdInvSMat = new double[nN * nN];
	//double* pdU = new double[nN * nN];
	//double* pdVt = new double[nN * nN];
	//double* pdSuperb = new double[nN * nN];

	double pdS[5];
	double pdInvS[5];
	double pdInvSMat[25];
	double pdU[25];
	double pdVt[25];
	double pdSuperb[25];

	double pdFRE[25];

	memcpy(pdFRE, FRE.pdFormula, sizeof(double) * 25);

	LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
		nN, nN, pdFRE, nN, pdS, pdU, nN, pdVt,
		nN, pdSuperb);

	//double* pdV = new double[nN * nN], * pdUt = new double[nN * nN];
	double pdV[25], pdUt[25];

	MatTranslate(pdVt, nN, nN, pdV);
	MatTranslate(pdU, nN, nN, pdUt);

	for (int n = 0; n < nN; n++) {
		pdInvS[n] = 1.0 / pdS[n];
	}
	memset(pdInvSMat, 0, 25 * sizeof(double));
	DiagMat(pdInvS, 5, pdInvSMat);
	//printMatrix(pdInvSMat, 5, 5);

   // double* pdTemp = new double[25];
	double pdTemp[25];
	memset(pdTemp, 0, sizeof(double) * 25);
	////memset(InvFRE.pdFormula, 0, sizeof(double) * nN * nN);
	InvFRE.zeros();
	MatMulti(pdV, nN, nN, nN, pdInvSMat, pdTemp);
	MatMulti(pdTemp, nN, nN, nN, pdUt, InvFRE.pdFormula);

	//delete[] pdTemp;
	//delete[] pdV;
	//delete[] pdUt;
	//delete[] pdSuperb;
	//delete[] pdVt;
	//delete[] pdU;
	//delete[] pdInvSMat;
	//delete[] pdInvS;
	//delete[] pdS;
}



void MatMulti(FResult& frMat1, int nR, int nC, int nK, FResult& frMat2, FResult& frMat3) {
	int n, r, c, k;
	memset(frMat3.pdFormula, 0, sizeof(double) * nR * nK);
	for (n = 0; n < nR * nC; n++) {
		r = n / nC;
		c = n % nC;
		for (k = 0; k < nK; k++) {
			frMat3.pdFormula[r * nK + k] += frMat1.pdFormula[n] * frMat2.pdFormula[c * nK + k];
		}
	}
}

void convertFR(FResult& frRDst, FResult& frRSrc) {
	int n;
	for (n = 0; n < frRDst.nCounter; n++) {
		frRDst.pdFormula[n] = frRSrc.pdFormula[n];
	}
}

void LMOptimize(cv::Mat& matInput, double* pdParam, int nMaxLoop, double dRes, double dMaxV, double* pdR) {
	double dLamda = 0.01;
	int nUpdateJ = 1;

	double dR[5];

	double* panX = new double[matInput.rows * matInput.cols];
	double* panY = new double[matInput.rows * matInput.cols];

	int n, m, r, c;

	int nHalfW = (matInput.cols) / 2;
	int nHalfH = (matInput.rows) / 2;
	for (m = 0; m < matInput.rows * matInput.cols; m++) {
		r = m / (matInput.cols) + 2;
		c = m % (matInput.cols) + 2;
		panX[m] = (c - cnHalfImageWidth) / static_cast<double>(cnHalfImageWidth);
		panY[m] = (r - cnHalfImageHeight) / static_cast<double>(cnHalfImageHeight);
	}


	FResult frDFData, frJFData, frDeltaData, frH, frHlm,
		frE, frEye, frEResult, frJDelta, frInvHlm, frDp, frR, frRlm, frLmDFData, frLmDeltaData, frELm;
	//double* pdDFData = NULL, *pdJFData = NULL, *pdDeltaData = NULL , *pdH = NULL;
	double dME;

	frR.resize(5);
	dR[0] = 0.0, dR[1] = 0.0, dR[2] = dMaxV, dR[3] = 1.0, dR[4] = 1.0;
	frR.pdFormula[0] = 0.0; frR.pdFormula[1] = 0.0; frR.pdFormula[2] = dMaxV; frR.pdFormula[3] = 1.0; frR.pdFormula[4] = 1.0;
	frRlm.resize(5);
	int nTotalSz = matInput.rows * matInput.cols;
	frDFData.resize(nTotalSz);
	frLmDFData.resize(nTotalSz);
	frLmDeltaData.resize(nTotalSz);
	frELm.resize(1);
	frJFData.resize(5 * nTotalSz);
	frH.resize(25);
	frHlm.resize(25);
	frE.resize(1);
	frEye.GenEye(25);
	frEResult.resize(25);
	frJDelta.resize(5);
	frInvHlm.resize(25);
	frDp.resize(5);
	frDeltaData.resize(nTotalSz);
	//InitEye(frEye, 5);

	for (n = 0; n < nMaxLoop; n++) {
		if (nUpdateJ == 1) {
			//pdDFData = new double[nTotalSz];
			DestinationFunc(frR.pdFormula, panX, panY, nTotalSz, frDFData);
			//cv::Mat matDFData = cv::Mat(matInput.rows, matInput.cols, CV_64FC1, frDFData.pdFormula);
			//printMatrix(frDFData.pdFormula, nTotalSz, 1);
			//pdJFData = new double[5*nTotalSz];
			JacobianFunc(frR.pdFormula, panX, panY, nTotalSz, frJFData);
			//pdDeltaData = new double[nTotalSz];
			//JacobianFunc(dR, panX, panY, nTotalSz, pdJFData);

			DiffMat(matInput, frDFData, frDeltaData);

			//for (r = 0; r < 5; r++) {
			//    for (c = 0; c < 5; c++) {
			//        cout << frH.pdFormula[5 * r + c] << " ";
			//    }
			//    cout << endl;
			//}
			//pdH = new double[25];
			//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 5, 5, nTotalSz, 1.0, pdJFData, nTotalSz, pdJFData, nTotalSz, 0.0, pdH, 5);
			frH.zeros();
			MatSquare(frJFData, 5, nTotalSz, frH);
			//printMatrix(frH.pdFormula, 5, 5);
			if (n == 0) {
				frE.zeros();
				MatSquare(frDeltaData, 1, frDeltaData.nCounter, frE);
			}
			//cout << frE.pdFormula[0] << endl;
		   // int r, c;

			int n = 0;
		}

		if (!isfinite(dLamda))
			break;
		NumMultiple(frEye, dLamda, frEResult);

		MatAdd(frH, frEResult, frHlm);
		frJDelta.zeros();
		VectorMulti(frJFData, 5, nTotalSz, frDeltaData, frJDelta);
		//printMatrix(frHlm.pdFormula, 5, 5);

		InvMatrix(frHlm, 5, frInvHlm);
		//FResult frTemp;
		//frTemp.resize(25);
		//MatMulti(frHlm, 5, 5, 5, frInvHlm, frTemp);

		//printMatrix(frTemp.pdFormula, 5, 5);
		frDp.zeros();
		MatMulti(frInvHlm, 5, 5, 1, frJDelta, frDp);
		//printMatrix(frDp.pdFormula, 5, 1);
		//printMatrix(frInvHlm.pdFormula, 5, 5);
		MatAdd(frR, frDp, frRlm);
		//printMatrix(frRlm.pdFormula, 5, 1);

		DestinationFunc(frRlm.pdFormula, panX, panY, nTotalSz, frLmDFData);
		DiffMat(matInput, frLmDFData, frLmDeltaData);
		frELm.zeros();
		MatSquare(frLmDeltaData, 1, frLmDeltaData.nCounter, frELm);

		dME = frELm.pdFormula[0] / nTotalSz;
		if (frELm.pdFormula[0] / nTotalSz < dRes)
			break;
		if (frELm.pdFormula[0] < frE.pdFormula[0]) {
			dLamda = dLamda / 10.0;
			convertFR(frR, frRlm);
			//printMatrix(frR.pdFormula, 5, 1);
			convertFR(frE, frELm);
			//cout << setprecision(10) << dME << endl;
			nUpdateJ = 1;
		}
		else {
			nUpdateJ = 0;
			dLamda = dLamda * 10.0;
		}
	}
	for (int n = 0; n < 5; n++) {
		pdR[n] = frR.pdFormula[n];
	}

	if (panX)
		delete[] panX;
	if (panY)
		delete[] panY;
}

//void ConverYUY2ToYUV(uchar* pucImagePtr, cv::Mat& matY, cv::Mat& matU, cv::Mat& matV, int nImageW, int nImageH) {
//	matY = cv::Mat(nImageH, nImageW, CV_32FC1);
//	matU = cv::Mat(nImageH, nImageW, CV_32FC1);
//	matV = cv::Mat(nImageH, nImageW, CV_32FC1);
//
//	uchar* pucProcessImgY, * pucProcessImgU, * pucProcessImgV;
//	pucProcessImgY = matY.ptr<uchar>(0);
//	pucProcessImgU = matU.ptr<uchar>(0);
//	pucProcessImgV = matV.ptr<uchar>(0);
//
//	int m = 0;
//	for (int n = 0; n < nImageW * nImageH; n++) {
//		pucProcessImgY[n] = pucImagePtr[2 * n];
//		m = n / 2;
//		pucProcessImgU[n] = pucImagePtr[4 * m + 1];
//		pucProcessImgV[n] = pucImagePtr[4 * m + 3];
//	}
//}

void LensShading(uchar* pbImagePtr, int nImageWidth, int nImageHeight) {
	vector<cv::Mat>vmatYUVImage;
	vmatYUVImage.resize(3);
	ConverYUY2ToYUV(pbImagePtr, vmatYUVImage[0], vmatYUVImage[1], vmatYUVImage[2], nImageWidth, nImageHeight);
	cv::Mat matYUVImage, matRGBImage;
	cv::merge(vmatYUVImage, matYUVImage);
	cv::cvtColor(matYUVImage, matRGBImage, cv::COLOR_YUV2BGR);

	vector<cv::Mat>vmatChannelImage;
	cv::split(matRGBImage, vmatChannelImage);

	double dBLMParam[5];
	double dGLMParam[5];
	double dRLMParam[5];

	double dMaxB, dMaxG, dMaxR;
	double dResB, dResG, dResR;

	GetMaxV(vmatChannelImage[0], dMaxB, 5, 5);
	GetMaxV(vmatChannelImage[1], dMaxG, 5, 5);
	GetMaxV(vmatChannelImage[2], dMaxR, 5, 5);

	int nMaxL = 10000;
	dResB = 1e-4;
	dResG = 1e-4;
	dResR = 1e-4;

	cv::Rect rectArea = cv::Rect(2, 2, matRGBImage.cols - 4, matRGBImage.rows - 4);
	cv::Mat matBImage = vmatChannelImage[0](rectArea).clone();
	cv::Mat matGImage = vmatChannelImage[1](rectArea).clone();
	cv::Mat matRImage = vmatChannelImage[2](rectArea).clone();

	//double dRB[5], dRG[5], dRR[5];

	double dStartT = static_cast<double>(cv::getTickCount());
	LMOptimize(matBImage, dBLMParam, nMaxL, dResB, dMaxB, pdRB);
	LMOptimize(matGImage, dGLMParam, nMaxL, dResG, dMaxG, pdRG);
	LMOptimize(matRImage, dRLMParam, nMaxL, dResR, dMaxR, pdRR);

	std::ofstream fout("ShadingParam.txt", std::ios_base::out);

	fout << "The parameters of shading is:" << endl;

	for (int n = 0; n < 5; n++) {
		fout << pdRB[n] << endl;
	}
	for (int n = 0; n < 5; n++) {
		fout << pdRG[n] << endl;
	}
	for (int n = 0; n < 5; n++) {
		fout << pdRR[n] << endl;
	}
	fout.close();
}

bool JudgeEmpty(double* pdRData) {
	for (int n = 0; n < 5; n++) {
		if (pdRData[n] != 0)
			return false;
	}

	return true;
}

void RecoveryImage(cv::Mat& matSrcImage, double* pdR, cv::Mat& matDstImage) {
	matDstImage = cv::Mat(matSrcImage.rows, matSrcImage.cols, CV_8UC1);

	uchar* pucmatSrcImage = matSrcImage.ptr<uchar>(0);
	uchar* pucDstImage = matDstImage.ptr<uchar>(0);

	int n, r, c;
	double  dTR, dTC;
	double dRaio;
	for (n = 0; n < matSrcImage.cols * matSrcImage.rows; n++) {
		r = n / matSrcImage.cols;
		c = n % matSrcImage.cols;
		dTC = (c - matSrcImage.cols / 2) / static_cast<double>(matSrcImage.cols / 2);
		dTR = (r - matSrcImage.rows / 2) / static_cast<double>(matSrcImage.rows / 2);

		dRaio = 1.0 / exp(-((dTC - pdR[0]) * (dTC - pdR[0]) / (2 * pdR[3] * pdR[3])) - ((dTR - pdR[1]) * (dTR - pdR[1]) / (2 * pdR[4] * pdR[4])));
		if (pucmatSrcImage[n] * dRaio > 255.0) {
			pucDstImage[n] = 255;
		}
		else	if (pucmatSrcImage[n] * dRaio < 0.0) {
			pucDstImage[n] = 0;
		}
		else {
			pucDstImage[n] = static_cast<uchar>(pucmatSrcImage[n] * dRaio + 0.5);
		}
	}
}

void DeShading(uchar* pbImagePtr, int nImageWidth, int nImageHeight, int nMode) {
	if (nMode == 1) {
		if (JudgeEmpty(pdRB) && JudgeEmpty(pdRG) && JudgeEmpty(pdRR)) {
			return;
		}
		else {
			//ofstream fout("Log.txt", ios_base::out);
			//fout << "Shading mode is:" << nMode << endl;
			vector<cv::Mat>vmatYUVImage;
			vmatYUVImage.resize(3);
			ConverYUY2ToYUV(pbImagePtr, vmatYUVImage[0], vmatYUVImage[1], vmatYUVImage[2], nImageWidth, nImageHeight);
			cv::Mat matYUVImage, matRGBImage, matRecoveryRGBImage, matRecoveryYUVImage;
			cv::merge(vmatYUVImage, matYUVImage);
			cv::cvtColor(matYUVImage, matRGBImage, cv::COLOR_YUV2BGR);

			vector<cv::Mat>vmatChannelImage;
			vector<cv::Mat>vmatRecoveryImage;
			vmatRecoveryImage.resize(3);
			cv::split(matRGBImage, vmatChannelImage);

			if (!JudgeEmpty(pdRB)) {
				RecoveryImage(vmatChannelImage[0], pdRB, vmatRecoveryImage[0]);
			}
			if (!JudgeEmpty(pdRG)) {
				RecoveryImage(vmatChannelImage[1], pdRG, vmatRecoveryImage[1]);
			}
			if (!JudgeEmpty(pdRR)) {
				RecoveryImage(vmatChannelImage[2], pdRR, vmatRecoveryImage[2]);
			}


			cv::merge(vmatRecoveryImage, matRecoveryRGBImage);
			cv::cvtColor(matRecoveryRGBImage, matRecoveryYUVImage, cv::COLOR_BGR2YUV);

			ConverYUVToYUY2(matRecoveryYUVImage, pbImagePtr, nImageWidth, nImageHeight);
			//fout.close();
		}
	}
	else if (nMode == 2) {
		//ofstream fout("Log.txt", ios_base::out);
		//fout << "Shading mode is:" << nMode << endl;
		vector<cv::Mat>vmatYUVImage;
		vmatYUVImage.resize(3);
		ConverYUY2ToYUV(pbImagePtr, vmatYUVImage[0], vmatYUVImage[1], vmatYUVImage[2], nImageWidth, nImageHeight);
		cv::Mat matYUVImage, matRGBImage, matRecoveryRGBImage, matRecoveryYUVImage;
		cv::merge(vmatYUVImage, matYUVImage);
		cv::cvtColor(matYUVImage, matRGBImage, cv::COLOR_YUV2BGR);

		vector<cv::Mat>vmatChannelImage;
		vector<cv::Mat>vmatRecoveryImage;
		vmatRecoveryImage.resize(3);
		cv::split(matRGBImage, vmatChannelImage);

		std::ifstream fin("ShadingParam.txt", std::ios_base::in);
		char cName[40];
		fin.getline(cName, 40);

		double dTemp;
		for (int n = 0; n < 5; n++) {
			fin >> dTemp;
			pdRB[n] = dTemp;
		}
		for (int n = 0; n < 5; n++) {
			fin >> dTemp;
			pdRG[n] = dTemp;
		}
		for (int n = 0; n < 5; n++) {
			fin >> dTemp;
			pdRR[n] = dTemp;
		}
		fin.close();

		RecoveryImage(vmatChannelImage[0], pdRB, vmatRecoveryImage[0]);
		RecoveryImage(vmatChannelImage[1], pdRG, vmatRecoveryImage[1]);
		RecoveryImage(vmatChannelImage[2], pdRR, vmatRecoveryImage[2]);

		cv::merge(vmatRecoveryImage, matRecoveryRGBImage);
		cv::cvtColor(matRecoveryRGBImage, matRecoveryYUVImage, cv::COLOR_BGR2YUV);

		ConverYUVToYUY2(matRecoveryYUVImage, pbImagePtr, nImageWidth, nImageHeight);
		//fout.close();
	}
}

int OpenSerialPort(int nInputSerialPort) {

	if (g_spcSerialController) {
		g_spcSerialController->SetUARTNo(nInputSerialPort);
		g_spcSerialController->InitPort();
		bool bResult = g_spcSerialController->GetSerialPortInitSuccess();

#ifdef LOG_ENABLE
		//*pfLog << "OpenSerialPort TestLog is:" << g_spcSerialController->m_nTestLog << endl;
		*pfLog << "OpenSerialPort Judgement Result is:" << bResult << endl;
#endif

		if (bResult)
			return g_spcSerialController->GetUARTNo();
	}
	else {
		g_spcSerialController = new SerialPortControl();
		g_spcSerialController->SetUARTNo(nInputSerialPort);
		g_spcSerialController->InitPort();

		bool bResult = g_spcSerialController->GetSerialPortInitSuccess();
#ifdef LOG_ENABLE
		//*pfLog << "OpenSerialPort TestLog is:" << g_spcSerialController->m_nTestLog << endl;
		*pfLog << "OpenSerialPort Judgement Result is:" << bResult << endl;
#endif
		if (bResult)
			return g_spcSerialController->GetUARTNo();
	}

	return -1;
}


int SetGain(int nGV) {
	if (g_spcSerialController) {
		char cCommand[100];

		sprintf_s(cCommand, "g%d\r\n", nGV);

#ifdef LOG_ENABLE
		*pfLog << "The gain command is:" << cCommand << endl;
		*pfLog << "The gain command length is:" << strlen(cCommand) << endl;
#endif
		g_spcSerialController->SendData(cCommand, strlen(cCommand));

		return 1;
	}
	return -1;
}


int SetExposure(int nExV) {
	if (g_spcSerialController) {
		char cCommand[100];

		sprintf_s(cCommand, "e%d\r\n", nExV);

#ifdef LOG_ENABLE
		*pfLog << "The exposure command is:" << cCommand << endl;
		*pfLog << "The exposure command length is:" << strlen(cCommand) << endl;
#endif
		g_spcSerialController->SendData(cCommand, strlen(cCommand));

		return 1;
	}
	return -1;
}

int SetCommand(char* pcCSharpString) {
	if (g_spcSerialController) {



		//int nCommand = strlen(pcCSharpString);


		string strCommand(pcCSharpString);

#ifdef LOG_ENABLE
		*pfLog << "The input command is:" << strCommand << endl;
#endif

		char cCommand[100];
		memcpy(cCommand, strCommand.c_str(), strCommand.length());
		cCommand[strCommand.length()] = '\r';
		cCommand[strCommand.length() + 1] = '\n';
		cCommand[strCommand.length() + 2] = '\0';

#ifdef LOG_ENABLE
		*pfLog << "The command is:" << cCommand << endl;
#endif

		g_spcSerialController->SendData(cCommand, strlen(cCommand) - 1);
		return 1;
	}
	return -1;
}