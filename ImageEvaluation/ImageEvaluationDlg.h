
// ImageEvaluationDlg.h: 헤더 파일
//

#pragma once

#include <math.h>
#include <io.h>
#include <GL/glut.h>
#include <GL/GLAUX.H>
#include <GL/glext.h>
#include <GL/glfw3.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ESMNvenc.h"
#include "cuda.cuh"
#include "video_reader.h"
//#include "NvTranscoder.h"
#include "getSSIM.h"
using namespace cv;
// CImageEvaluationDlg 대화 상자
class CImageEvaluationDlg : public CDialogEx
{
// 생성입니다.
public:
	CImageEvaluationDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_IMAGEEVALUATION_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	HGLRC m_hRC;
	HDC m_hDC;
	HWND hwNative;
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnDestroy();
	afx_msg void OnCbnSelchangeCombo1();
	int VideoRender();
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();

	int CropImage();
	CComboBox m_crop_combo;
	CComboBox m_bitrate_combo;
	double getSSIM(Mat &img,Mat &img1, int blocksize);
	double getPSNR(Mat& img, Mat& img1);
	CStatic m_psnr;
	CStatic m_ssim;
	CString pathName;
	
	afx_msg void OnBnClickedButton4();
	CStatic m_filename;
	int renderFlag = 0;
	int EncoderVideo(Mat& img_src,unsigned char* m_pBitstream, long long	pts);
	int GetFileList(LPCTSTR filePath);
	int Muxing(VideoReaderState* state, unsigned char* encodeBuffer);
	TCHAR szBuffer[512];
	afx_msg void OnBnClickedButton5();
	afx_msg void OnBnClickedButton6();
	CStatic m_dName;
	CString fileName;

	uint8_t*				m_dptr;
	size_t					m_nDptrPitch;
	ESMNvenc::CONTEXT_T		m_nvEncoderCtx;
	ESMNvenc *				m_nvEncoder;
	int						m_nBitstreamCapacity;
	uint8_t*				m_pBitstream;

	CUdeviceptr* cuDeviceptr;
	CUdeviceptr* cuDeviceptr1;
	CUdeviceptr* cuDeviceptr2;


	CUdeviceptr dptr;
	size_t pitch;
	size_t pitch1;
	size_t pitch2;
	long long	bitstreamTimestamp = 0;
	int	bitstreamSize = 0;
	int encoderExtradataSize = 0;
	int m_nOutputFPS = 30;
	uint8_t* encoderExtradata = NULL;

	unsigned int cudaPitch = 0;
	//_CudaEncoding cudaEncoding;

	int MakeOutFile(VideoReaderState* state);
	int response;
	int stream_index = 0;
	int* stream_mapping = NULL;
	int stream_mapping_size = 0;
	AVFormatContext* av_dm_ctx;
	AVPacket av_o_packet;
	FILE* outputFile;
	AVOutputFormat* av_o_format = NULL;
	int EndFile();



	int CudaEncode(Mat& img_src);
};
