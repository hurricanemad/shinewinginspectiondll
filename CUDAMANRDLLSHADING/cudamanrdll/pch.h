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
#include "SerialControl.h"

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

#ifdef _DEBUG
#define LOG_ENABLE
#endif


//#define LOG_TIME_ENABLE

#pragma comment(lib, cvLib("core"))
#pragma comment(lib, cvLib("highgui"))
#pragma comment(lib, cvLib("imgcodecs"))
#pragma comment(lib, cvLib("imgproc"))
#pragma comment(lib, cvLib("videoio"))
#pragma comment(lib, "libblas.lib")
#pragma comment(lib, "liblapack.lib")
#pragma comment(lib, "liblapacke.lib")

struct EstimatedResult {
    int nDefinitionFarAlready;
    int nDefinitionNearAlready;
    int nFovAlready;
    int nBrAlready;
    int nUnformityAlready;
    int nDarkAreaAlready;
};

/*******************************************************/
/*@@@@@函数名:FigureParameters@@@@@@@*/
/*函数说明：根据输入YUY2图像执行检测。*/
/*输入：*/
/*pucImagePtr 类型：unsigned char 说明：输入图像地址指针，图像为YUYV格式*/
/*nImageW 类型：int 说明：输入图像宽度*/
/*nImageH 类型：int 说明：输入图像高度*/
/*nFrameNo 类型：int 说明：输入帧号,用于时域滤波*/
/*dRatio 类型：double 说明：时域滤波参数，用于调整时域滤波强度*/
/*nEnFilter 类型：int 说明：是否进行时域滤波处理，0表示不进行滤波处理，1表示进行滤波处理*/
/*dThresh 类型：double 说明：图像亮区检测阈值参数，用于判断图像中超过此参数阈值的图像区域范围*/
/*nEnDefinition 类型：int 说明：是否进行图像清晰度检测，0表示不进行清晰度检测，1表示进行清晰度检测*/
/*nEnFov 类型：int 说明：是否进行视场角检测，0表示不进行视场角检测，1表示进行视场角检测*/
/*nEnBR 类型：int 说明：是否进行亮度比检测，0表示不进行亮度比检测，1表示进行亮度比检测*/
/*nEnUnformity 类型：int 说明：是否进行图像均匀度检测，0表示不进行均匀度检测，1表示进行均匀度检测 */
/*perTemp 类型：EstimatedResult(自定义变量) 说明：用于向C#输出信息，指示各性能检测是否完成。*/
/*输出：*/
/******************************************************/

extern "C" DLL_EXPORT void FigureParameters(uchar * pucImagePtr,
                                                                            int nImageW, int nImageH,
                                                                            int nFrameNo, double dRatio, 
                                                                            int nEnFilter, double dThresh,
                                                                            int nEnDefinition, int nEnFov,
                                                                            int nEnBR, int nEnUnformity,
                                                                            int nEnWB, EstimatedResult* perTemp);
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
extern "C" DLL_EXPORT int InitializeSerialControl();
extern "C" DLL_EXPORT void ReleaseSerialControl();

extern "C" DLL_EXPORT void FigureCross(uchar* pucImagePtr, int nImageW, int nImageH);
extern "C" DLL_EXPORT void FigureCorners(uchar* pucImagePtr, int nImageW, int nImageH);
extern "C" DLL_EXPORT void FlipImage(uchar* pucImagePtr, int nImageW, int nImageH);
extern "C" DLL_EXPORT void LensShading(uchar * pbImagePtr, int nImageWidth, int nImageHeight);
extern "C" DLL_EXPORT void DeShading(uchar * pbImagePtr, int nImageWidth, int nImageHeight, int nMode);

extern "C" DLL_EXPORT int OpenSerialPort(int nInputSerialPort);
extern "C" DLL_EXPORT int SetGain(int nGV);
extern "C" DLL_EXPORT int SetExposure(int nExV);
extern "C" DLL_EXPORT int SetCommand(char* pcCSharpString);

#endif //PCH_H
