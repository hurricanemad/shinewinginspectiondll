// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H


#ifdef SIMPLEDLL_EXPORT
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif
// 添加要在此处预编译的标头



#include "framework.h"

#include <vector>
#include <fstream>
#include <mutex>
#include <queue>


//#include "book.h"

using namespace std;
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLib(name) "opencv_" name CV_VERSION_ID "d.lib"
#else
#define cvLib(name) "opencv_" name CV_VERSION_ID ".lib"
#endif

//#ifdef _DEBUG
//#define LOG_ENABLE
//#endif


//#define LOG_TIME_ENABLE

#pragma comment(lib, cvLib("core"))
#pragma comment(lib, cvLib("highgui"))
#pragma comment(lib, cvLib("imgcodecs"))
#pragma comment(lib, cvLib("imgproc"))
#pragma comment(lib, cvLib("videoio"))
#pragma comment(lib, "libblas.lib")
#pragma comment(lib, "liblapack.lib")
#pragma comment(lib, "liblapacke.lib")

extern "C" DLL_EXPORT void FigureParameters(uchar * pucImagePtr, int nImageW, int nImageH, int nFrameNo, double dRatio, int nEnFilter, double dThresh, int nEnDefinition, int nEnFov, int nEnBR, int nEnUnformity);
extern "C" DLL_EXPORT void SaveImage( uchar * pucImagePtr, int nImageW, int nImageH, int nNo);
extern "C" DLL_EXPORT double GetBrenner();
extern "C" DLL_EXPORT double GetTenegrad();
extern "C" DLL_EXPORT double GetLaplacian();
extern "C" DLL_EXPORT double GetBrighnessRatio();

extern "C" DLL_EXPORT int GetPosLeftFOV();
extern "C" DLL_EXPORT int GetPosRightFOV();
extern "C" DLL_EXPORT int GetNegLeftFOV();
extern "C" DLL_EXPORT int GetNegRightFOV();

extern "C" DLL_EXPORT double* GetSubAreaBrenner();
extern "C" DLL_EXPORT double* GetSubAreaTenegrad();
extern "C" DLL_EXPORT double* GetSubAreaLaplacian();
extern "C" DLL_EXPORT double* GetUnformityVs();

extern "C" DLL_EXPORT double* GetpRR();
extern "C" DLL_EXPORT double* GetpRG();
extern "C" DLL_EXPORT double* GetpRB();

extern "C" DLL_EXPORT void StartVideoCapture(int strFIleName, int nFrameRate, int nWidth, int nHeight);
extern "C" DLL_EXPORT void StopVideoCapture();
extern "C" DLL_EXPORT void SaveVideo(uchar * pucImagePtr, int nImageW, int nImageH);
extern "C" DLL_EXPORT void InitializeImage(int nImageW, int nImageH, double dRatio);
extern "C" DLL_EXPORT void ReleaseImage();
extern "C" DLL_EXPORT void FigureCross(uchar* pucImagePtr, int nImageW, int nImageH);
extern "C" DLL_EXPORT void FigureCorners(uchar* pucImagePtr, int nImageW, int nImageH);
extern "C" DLL_EXPORT void FlipImage(uchar* pucImagePtr, int nImageW, int nImageH);
extern "C" DLL_EXPORT void LensShading(uchar * pbImagePtr, int nImageWidth, int nImageHeight);
extern "C" DLL_EXPORT void DeShading(uchar * pbImagePtr, int nImageWidth, int nImageHeight, int nMode);


#endif //PCH_H
