
// ImageEvaluationDlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "ImageEvaluation.h"
#include "ImageEvaluationDlg.h"
#include "afxdialogex.h"
#include <ESMLocks.h>

//#include "cuda.cuh"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.
VideoReaderState vr_state;
int64_t pts;
GLFWwindow* window;
GLFWwindow* cropwindow;
struct YV12DemensionInfo {
	int w;  
	int h; 
	int strideY; 
	int strideC; 
	int offsetV; 
	int offsetU; 
	int yPlaneW; 
	int yPlaneH; 
	int cPlaneW; 
	int cPlaneH;
	int yPlaneSize;
	int uvPlaneSize; 
	int totalBytes; 
};
void computeYVDemensionInfo(YV12DemensionInfo* p, int w, int h)
{
	int pad;
	int strideYPlane;
	int strideCPlane; 
	pad = 16 - w % 16;
	if (pad == 16)
	{
		pad = 0;
	}
	strideYPlane = w + pad;
	pad = 16 - (strideYPlane / 2) % 16;
	if (pad == 16) pad = 0;
	strideCPlane = (strideYPlane / 2) + pad;
	p->w = w;
	p->h = h;
	p->yPlaneW = w;
	p->yPlaneH = h;
	p->cPlaneW = strideYPlane / 2;
	p->cPlaneH = h / 2;
	p->strideY = strideYPlane;
	p->strideC = strideCPlane;
	p->offsetV = strideYPlane * h;
	p->offsetU = strideYPlane * h + strideCPlane * (h / 2);
	p->yPlaneSize = strideYPlane * h;
	p->uvPlaneSize = strideCPlane * (h / 2);
	p->totalBytes = p->yPlaneSize + p->uvPlaneSize * 2;
}
class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CImageEvaluationDlg 대화 상자



CImageEvaluationDlg::CImageEvaluationDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_IMAGEEVALUATION_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CImageEvaluationDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO1, m_crop_combo);
	DDX_Control(pDX, IDC_COMBO2, m_bitrate_combo);
	DDX_Control(pDX, IDC_PSNR, m_psnr);
	DDX_Control(pDX, IDC_SSIM, m_ssim);
	DDX_Control(pDX, IDC_FILENAME, m_filename);
	DDX_Control(pDX, IDC_DName, m_dName);
}

BEGIN_MESSAGE_MAP(CImageEvaluationDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_DESTROY()
	ON_CBN_SELCHANGE(IDC_COMBO1, &CImageEvaluationDlg::OnCbnSelchangeCombo1)
	ON_BN_CLICKED(IDC_BUTTON1, &CImageEvaluationDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CImageEvaluationDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CImageEvaluationDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CImageEvaluationDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CImageEvaluationDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CImageEvaluationDlg::OnBnClickedButton6)
END_MESSAGE_MAP()


// CImageEvaluationDlg 메시지 처리기

BOOL CImageEvaluationDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	m_crop_combo.AddString(_T("10"));
	m_crop_combo.AddString(_T("20"));
	m_crop_combo.AddString(_T("30"));
	m_crop_combo.AddString(_T("40"));
	m_crop_combo.AddString(_T("50"));

	m_bitrate_combo.AddString(_T("5"));
	m_bitrate_combo.AddString(_T("10"));
	m_bitrate_combo.AddString(_T("15"));
	m_bitrate_combo.AddString(_T("20"));
	m_crop_combo.SetCurSel(0);
	m_bitrate_combo.SetCurSel(0);



	m_dptr = NULL;
	m_nDptrPitch = 0;
	m_nBitstreamCapacity = 1024 * 1024 * 2;
	m_pBitstream = (uint8_t*)malloc(m_nBitstreamCapacity);
	

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CImageEvaluationDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CImageEvaluationDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CImageEvaluationDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



int CImageEvaluationDlg::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CDialogEx::OnCreate(lpCreateStruct) == -1)
		return -1;
	return 0;
}


void CImageEvaluationDlg::OnSize(UINT nType, int cx, int cy)
{
	CDialogEx::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
}


void CImageEvaluationDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
}


void CImageEvaluationDlg::OnCbnSelchangeCombo1()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


int CImageEvaluationDlg::VideoRender()
{
	// TODO: 여기에 구현 코드 추가.
	
	char* filepath = new char[pathName.GetLength()];
	strcpy(filepath, CT2A(pathName));
	//GetFileList(szBuffer);
	//EncoderVideo();

	//if (!video_reader_open(&vr_state, "C:\\Users\\4DReplay\\Desktop\\Muxingtest.mp4")) {
	//	printf("Couldn't open video file\n");
	//	CString str;
	//	str = (LPSTR)"Couldn't open video file";
	//	MessageBox(str);
	//	return 1;
	//}
	if (!video_reader_open(&vr_state, filepath)) {
		printf("Couldn't open video file\n");
		CString str;
		str = (LPSTR)"Couldn't open video file";
		MessageBox(str);
		return 1;
	}
	else {
		renderFlag = 0;
	}
	if (!glfwInit()) {
		printf("Couldn't init GLFW\n");
		return 1;
	}
	window = glfwCreateWindow(1920, 1080, "Play Video", NULL, NULL);
	cropwindow = glfwCreateWindow(1920, 1080, "Play CropVideo", NULL, NULL);

	if (!window) {
		printf("Couldn't open window\n");
		return 1;
	}



	const int frame_width = vr_state.width;
	const int frame_height = vr_state.height;
	const int output_width = 1920;
	const int output_height = 1080;
	uint8_t* frame_rgb_data = new uint8_t[frame_width * frame_height * 3];
	uint8_t* frame_bgr_data = new uint8_t[frame_width * frame_height * 3];
	uint8_t* frame_crop_data = new uint8_t[output_width * output_height * 3];
	cudaMallocPitch(&cuDeviceptr, &pitch, output_width, output_height);
	int nextpts = 0;
	int ptr = 0;

	outputFile = fopen("encode.264", "wb");
	char* outpath = "encode1.h264";
	while (!glfwWindowShouldClose(window)&&renderFlag==0) {
		
		
		glfwMakeContextCurrent(window);

		GLuint tex_handle;
		glGenTextures(1, &tex_handle);
		glBindTexture(GL_TEXTURE_2D, tex_handle);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		int window_width, window_height;
		glfwGetFramebufferSize(window, &window_width, &window_height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, window_width, window_height, 0, -1, 1);
		glMatrixMode(GL_MODELVIEW);

		if (!video_reader_read_frame(&vr_state, frame_rgb_data, frame_bgr_data, &pts)) {
			printf("Couldn't load video frame\n");
			return 0;
		}
		if (nextpts == pts && nextpts!=0) {
			break;
		}

		static bool first_frame = true;
		if (first_frame) {
			glfwSetTime(0.0);
			first_frame = false;
		}
		glBindTexture(GL_TEXTURE_2D, tex_handle);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_width, frame_height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb_data);

		//OpenCV Crop
		
		CString str;
		m_crop_combo.GetLBText(m_crop_combo.GetCurSel(), str);
		int cropsize = _ttoi(str);

		Mat mat(frame_height, frame_width, CV_8UC3, frame_bgr_data, frame_width * 3);
		Mat cropmat(frame_height, frame_width, CV_8UC3, frame_bgr_data, frame_width * 3);

		int cropwidth = frame_width * sqrt(100 - cropsize) / 10;
		int cropheight = frame_height * sqrt(100 - cropsize) / 10;
		int cornerX = (frame_width - frame_width * sqrt(100 - cropsize) / 10) / 2;
		int cornerY = (frame_height - frame_height * sqrt(100 - cropsize) / 10) / 2;
		cv::Rect crop_region(cornerX, cornerY, cropwidth, cropheight);

		cropmat = cropmat(crop_region);

		resize(cropmat, cropmat, Size(output_width, output_height), 0, 0, INTER_LINEAR);
		
		memcpy(frame_crop_data, cropmat.data, output_width * output_height *3);
		cropmat.rows;
		cropmat.cols;
		EncoderVideo(cropmat, m_pBitstream,pts);
		Mat m(Size(output_width, output_height), CV_8UC3,255);
		Mat m1(Size(output_width*1.5, output_height), CV_8UC3, 255);
		if (pts == 0) {
			MakeOutFile(&vr_state);
			//_CudaEncoding cudaEncoding;
			//cudaEncoding.Init(filepath, outpath);
			//cudaEncoding.CreateOutFile();
			//cudaEncoding.CreateEncoder();
		}
		//get PSNR, SSIM
		/*cropmat.convertTo(cropmat, CV_64F);
		mat.convertTo(mat, CV_64F);
		double psnr = getPSNR(cropmat, mat);
		str.Format(_T("%.2lf"), psnr);
		m_psnr.SetWindowTextW(str);*/
		
		
		//double ssim = getSSIM(cropmat, mat, 15);
		////3^3*2^3*5   / 2^7*3*5
		//str.Format(_T("%.2lf"), ssim);
		//m_ssim.SetWindowTextW(str);
		//


		nextpts = pts;
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, tex_handle);
		glBegin(GL_QUADS);
		glTexCoord2d(0, 0); glVertex2i(0, 0);
		glTexCoord2d(1, 0); glVertex2i(frame_width, 0);
		glTexCoord2d(1, 1); glVertex2i(frame_width, frame_height);
		glTexCoord2d(0, 1); glVertex2i(0, frame_height);
		glEnd();
		glDisable(GL_TEXTURE_2D);

		glfwSwapBuffers(window);


		glfwMakeContextCurrent(cropwindow);
		glGenTextures(1, &tex_handle);
		glBindTexture(GL_TEXTURE_2D, tex_handle);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glfwGetFramebufferSize(cropwindow, &window_width, &window_height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, window_width, window_height, 0, -1, 1);
		glMatrixMode(GL_MODELVIEW);

		glBindTexture(GL_TEXTURE_2D, tex_handle);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, output_width, output_height, 0, GL_BGR, GL_UNSIGNED_BYTE, frame_crop_data);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, output_width, output_height, 0, GL_BGR, GL_UNSIGNED_BYTE, frame_crop_data);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, tex_handle);
		glBegin(GL_QUADS);
		glTexCoord2d(0, 0); glVertex2i(0, 0);
		glTexCoord2d(1, 0); glVertex2i(frame_width, 0);
		glTexCoord2d(1, 1); glVertex2i(frame_width, frame_height);
		glTexCoord2d(0, 1); glVertex2i(0, frame_height);
		glEnd();
		glDisable(GL_TEXTURE_2D);

		glfwSwapBuffers(cropwindow);

		glfwPollEvents();
	}
	//printf("%d\n", vr_state.av_format_ctx->bit_rate);
	EndFile();
	//cudaEncoding.EndCudaEncoding();
	video_reader_close(&vr_state);
	return 0;
}



void CImageEvaluationDlg::OnBnClickedButton1()
{
	//SendMessage(WM_CLOSE, ID_APP_EXIT, NULL);
	
	VideoRender();
	//glfwSetWindowShouldClose(window, GLFW_TRUE);
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


int CImageEvaluationDlg::CropImage()
{
	// TODO: 여기에 구현 코드 추가.
	return 0;
}

double CImageEvaluationDlg::getSSIM(Mat& img, Mat& img1, int blocksize)
{
	// TODO: 여기에 구현 코드 추가.
	//int a = 10;
	//int b = 20;
	int c =0;
	//CGPUACC gpuacc;
	//
	//int nbBlockPerHeight = img.rows / blocksize;
	//int nbBlockPerWidth = img.cols / blocksize;
	//double sd;
	////gpuacc.getSigma(img, nbBlockPerHeight, nbBlockPerWidth, 15, &sd);
	//
	////int m =
	////gpuacc.getSigma(img,img.c);
	//gpuacc.sum_cuda(a, b, &c);
	//gpuacc.getSigma(img, nbBlockPerHeight, nbBlockPerWidth, 15, &sd);
	//CString str;
	//str.Format(_T("%lf"), sd);
	//MessageBox(str);
	
	
	return c;
}


double CImageEvaluationDlg::getPSNR(Mat& img, Mat& img1)
{
	double MSE = 0;
	int height = img.rows;
	int width = img.cols;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			MSE += (img.at<double>(i, j) - img1.at<double>(i, j)) * (img.at<double>(i, j) - img1.at<double>(i, j));
	MSE /= height * width;
	int D = 255;

	// TODO: 여기에 구현 코드 추가.
	return (10 * sqrt((D * D) / MSE));
}


//Close Window
void CImageEvaluationDlg::OnBnClickedButton2()
{
	try {
		renderFlag = 1;
		glfwDestroyWindow(window);
		glfwDestroyWindow(cropwindow);
	}
	catch (Exception e) {}
}


//Get Video File
void CImageEvaluationDlg::OnBnClickedButton3()
{
	static TCHAR BASED_CODE szFilter[] = _T("동영상 파일(*.avi;*.wmv;*.mp4;*.mkv;)|*.avi;*.mp4;*.wmv;*.mkv;|All Files(*.*)|*.*||");
	CFileDialog dlg(TRUE, NULL,NULL, OFN_HIDEREADONLY, szFilter);
	if (IDOK == dlg.DoModal())
	{
		pathName = dlg.GetPathName();
		fileName = dlg.GetFileName();
		m_filename.SetWindowTextW(fileName);
		//int n= pathName.Replace(L"\\", L"\\\\");
	}
}

//Close App
void CImageEvaluationDlg::OnBnClickedButton4()
{
	SendMessage(WM_CLOSE, ID_APP_EXIT, NULL);
}


int CImageEvaluationDlg::EncoderVideo(Mat& img_src, uint8_t* m_pBitstream, long long	pts)
{
	CString str;
	m_bitrate_combo.GetLBText(m_bitrate_combo.GetCurSel(), str);
	int bitratesize = _ttoi(str);

	int outputWidth = 1920;
	int outputHeight = 1080;
	int encoderCodec = ESMNvenc::VIDEO_CODEC_T::AVC;
	int encoderProfile = ESMNvenc::AVC_PROFILE_T::HP;
	int encoderBitrate = 1024 * 1024 * bitratesize;
	// 5,10,15,20

	if (m_nvEncoder == NULL) 
	{
		m_nvEncoder = new ESMNvenc();
		m_nvEncoderCtx.deviceIndex = 0;
		m_nvEncoderCtx.width = outputWidth;
		m_nvEncoderCtx.height = outputHeight;
		m_nvEncoderCtx.codec = encoderCodec;
		m_nvEncoderCtx.profile = encoderProfile;
		m_nvEncoderCtx.bitrate = encoderBitrate;
		m_nvEncoderCtx.fps = m_nOutputFPS;
		m_nvEncoderCtx.colorspace = ESMNvenc::COLORSPACE_T::YV12;
		m_nvEncoder->Initialize(&m_nvEncoderCtx);

		//CUresult cret = ::cuMemAllocPitch((CUdeviceptr*)&pFrame, &cuPitchConverted, _context->width, (_context->height >> 1) * 3, 16);
		::cudaMallocPitch(&m_dptr, &m_nDptrPitch, outputWidth, (outputHeight >> 1) * 3);
	}
	cv::Mat cvtMat((img_src.rows >> 1) * 3, img_src.cols, CV_8UC1);
	cv::cuda::GpuMat encMat((outputHeight >> 1) * 3, outputWidth, CV_8UC1, m_dptr, m_nDptrPitch);
	cv::cvtColor(img_src, cvtMat, COLOR_BGR2YUV_YV12);


	encMat.upload(cvtMat);





	/*
	uint8_t* lumaDst = m_dptr;
	uint8_t* chromaDst = lumaDst + m_nDptrPitch * outputHeight;
	int32_t strideSrc = cvtMat.step;
	uint8_t* lumaSrc = cvtMat.data;
	uint8_t* chromaSrc = lumaSrc + strideSrc * outputHeight;
	::cudaMemcpy2D(lumaDst, m_nDptrPitch, lumaSrc, strideSrc, outputWidth, outputHeight, cudaMemcpyHostToDevice);
	::cudaMemcpy2D(chromaDst, m_nDptrPitch, chromaSrc, strideSrc, outputWidth, outputHeight >> 1, cudaMemcpyHostToDevice);
	//::cudaMemcpy2D(m_dptr, m_nDptrPitch, encodeMat.data, outputWidth, outputWidth, outputHeight, cudaMemcpyHostToDevice);
	//::cudaMemcpy2D(m_dptr + m_nDptrPitch * outputHeight, m_nDptrPitch, encodeMat.data + outputWidth * outputHeight, outputWidth, outputWidth, outputHeight >> 1, cudaMemcpyHostToDevice);
	//::cudaMemcpy2D(m_dptr + m_nDptrPitch * outputHeight + (m_nDptrPitch >> 1) * (outputHeight >> 1), m_nDptrPitch >> 1, encodeMat.data + encodeMat.step * outputHeight + (encodeMat.step >> 1) * (outputHeight >> 1), encodeMat.step >> 1, outputWidth >> 1, outputHeight >> 1, cudaMemcpyHostToDevice);
	*/
	m_nvEncoder->Encode(m_dptr, m_nDptrPitch, pts, m_pBitstream, m_nBitstreamCapacity, bitstreamSize, bitstreamTimestamp);
	
	if (bitstreamSize > 0)
	{
		::fwrite(m_pBitstream, 1, bitstreamSize, outputFile);
	}

	return 0;
}


int CImageEvaluationDlg::Muxing(VideoReaderState* state, unsigned char* encodeBuffer)
{
	auto& av_format_ctx = state->av_format_ctx;

	response = av_read_frame(av_format_ctx, &av_o_packet);
	
	//0,1 video, audio
	AVStream* in_stream, * out_stream;

	av_o_packet.size = bitstreamSize;
	av_o_packet.data = encodeBuffer;
	//memcpy(encodeBuffer, av_o_packet.data, av_o_packet.size);
	//memcpy(av_o_packet.data, encodeBuffer, bitstreamSize);
	in_stream = av_format_ctx->streams[av_o_packet.stream_index];
	if (av_o_packet.stream_index >= stream_mapping_size ||
		stream_mapping[av_o_packet.stream_index] < 0) {
		av_packet_unref(&av_o_packet);
		//continue;
	}
	av_o_packet.stream_index = stream_mapping[av_o_packet.stream_index];
	out_stream = av_dm_ctx->streams[av_o_packet.stream_index];
	//av_o_packet.pts = av_rescale_q_rnd(av_o_packet.pts, in_stream->time_base, out_stream->time_base, AV_ROUND_INF);
	//av_o_packet.dts = av_rescale_q_rnd(av_o_packet.dts, in_stream->time_base, out_stream->time_base, AV_ROUND_INF);
	//av_o_packet.duration = av_rescale_q(av_o_packet.duration, in_stream->time_base, out_stream->time_base);
	av_o_packet.pos = -1;
	//nextpts = av_o_packet.pts;
	//nextdts = av_o_packet.dts;
	response = av_interleaved_write_frame(av_dm_ctx, &av_o_packet);
	if (response < 0) {
		printf("Error muxing packet\n");
		//break;
	}
	av_packet_unref(&av_o_packet);
	
	
	//av_write_trailer(av_dm_ctx);
	
	return 0;
}


int CImageEvaluationDlg::GetFileList(LPCTSTR filePath)
{  
	CFileFind finder;
	CString path =  _T("\\*.*");
	CString video = _T(".mp4");
	bool working = finder.FindFile(filePath+path);
	CString fileName;
	int count = 0;
	while (working) {
		working = finder.FindNextFileW();
		if (finder.IsDots()) continue;

		if (finder.IsDirectory()) {
			/// 하위폴더를 뒤져본다.
			if (GetFileList(finder.GetFilePath())) {
				return 0;
			}
		}
		else {
			CString curfile = finder.GetFileName();
			if (curfile.Find(video)>0) {
				//TransCoding
			}
		}
	}
	return 0;
}


void CImageEvaluationDlg::OnBnClickedButton5()
{
	BROWSEINFO BrInfo;
	

	::ZeroMemory(&BrInfo, sizeof(BROWSEINFO));
	::ZeroMemory(szBuffer, 512);

	BrInfo.hwndOwner = GetSafeHwnd();
	BrInfo.lpszTitle = _T("파일이 저장될 폴더를 선택하세요");
	BrInfo.ulFlags = BIF_NEWDIALOGSTYLE | BIF_EDITBOX | BIF_RETURNONLYFSDIRS;
	LPITEMIDLIST pItemIdList = ::SHBrowseForFolder(&BrInfo);
	::SHGetPathFromIDList(pItemIdList, szBuffer);

	
	CString str;
	str.Format(_T("%s"), szBuffer);
	m_dName.SetWindowTextW(str);
	//GetFileList(szBuffer);
}

void CImageEvaluationDlg::OnBnClickedButton6()
{
	CFileFind finder;
	CString path = _T("\\*.*");

	if (finder.FindFile(szBuffer)) {
		GetFileList(szBuffer);
	}
	else {
		CString str;
		str = (LPSTR)"Couldn't Choose Directory";
		MessageBox(str);
	}
	
	
	
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


int CImageEvaluationDlg::MakeOutFile(VideoReaderState* state)
{
	auto& av_format_ctx = state->av_format_ctx;
	char outFile[100] = "Muxingtest.mp4";
	av_o_format = av_guess_format(NULL, outFile, NULL);

	stream_mapping_size = av_format_ctx->nb_streams;
	stream_mapping = (int*)av_mallocz_array(stream_mapping_size, sizeof(*stream_mapping));
	if (!stream_mapping) {
		printf("Could not mapping stream\n");
	}
	response = avformat_alloc_output_context2(&av_dm_ctx, NULL, NULL, outFile);
	if (response < 0) {
		printf("Could not allocate output context\n");
	}
	for (int i = 0; i < av_format_ctx->nb_streams; i++) {
		AVStream* out_stream;
		AVStream* in_stream;
		in_stream = av_format_ctx->streams[i];
		in_stream->codecpar->bit_rate = m_nvEncoderCtx.bitrate;
		in_stream->codecpar->width = m_nvEncoderCtx.width;
		in_stream->codecpar->height = m_nvEncoderCtx.height;

		AVCodecParameters* incodecpar = in_stream->codecpar;
		if (incodecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
			incodecpar->codec_type != AVMEDIA_TYPE_VIDEO &&
			incodecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
			stream_mapping[i] = -1;
			continue;
		}
		stream_mapping[i] = stream_index++;
		out_stream = avformat_new_stream(av_dm_ctx, NULL);
		if (!out_stream) {
			printf("Failed allocating output stream\n");
		}

		response = avcodec_parameters_copy(out_stream->codecpar, incodecpar);
		out_stream->codecpar->bit_rate;
		out_stream->codecpar->width;
		out_stream->codecpar->height;
		if (response < 0) {
			printf("Failed parametser copy\n");
		}
		out_stream->codecpar->codec_tag = 0;
	}
	
	av_dump_format(av_dm_ctx, 0, outFile, 1);
	if (!(av_o_format->flags & AVFMT_NOFILE)) {
		response = avio_open(&av_dm_ctx->pb, outFile, AVIO_FLAG_WRITE);
		if (response < 0) {
			printf("Could not open output file");

		}
	}
	//av_dm_ctx->bit_rate = m_nvEncoderCtx.bitrate;
	response = avformat_write_header(av_dm_ctx, NULL);
	if (response < 0) {
		fprintf(stderr, "Error occurred when opening output file\n");

	}
	// TODO: 여기에 구현 코드 추가.
	return 0;
}


int CImageEvaluationDlg::EndFile()
{
	//av_write_trailer(av_dm_ctx);
	//avformat_free_context(av_dm_ctx);
	//cudaFree(cuDeviceptr);
	//m_nvEncoder->Release();
	//// TODO: 여기에 구현 코드 추가.
	return 0;
}

int CImageEvaluationDlg::CudaEncode(Mat& img_src)
{
	///*char* filepath = new char[pathName.GetLength()];
	//strcpy(filepath, CT2A(pathName));
	//char* outpath = "C:\\Users\\4DReplay\\Desktop\\encode1.h264";
	//int outputWidth = 1920;
	//int outputHeight = 1080;
	//*/
	//void* dptr=0;

	//Mat encodeMat(img_src.rows, img_src.cols, CV_8UC3);
	//img_src.copyTo(encodeMat);
	//Mat encodeMat2(img_src.rows, img_src.cols, CV_8UC3);
	//Mat cudaMat(img_src.rows, img_src.cols * 1.5, CV_8UC1);

	//cv::cvtColor(img_src, encodeMat, COLOR_BGR2YUV_YV12);
	//cudaMallocPitch(&dptr,&pitch, outputWidth*1.5, outputHeight);
	//cudaMemcpy2D(&dptr, pitch, encodeMat.data, outputWidth*1.5, outputWidth*1.5, outputHeight, cudaMemcpyHostToDevice);
	//cv::cuda::GpuMat m(cv::Size(img_src.rows, img_src.cols), CV_8UC3, (void*)encodeMat.data, pitch);

	//cv::Mat m1(m.size(), m.type());
	//m.download(m1);

	//
	//m.upload(encodeMat);
	///*_CudaEncoding cudaEncoding;
	//cudaEncoding.Init(filepath,outpath);
	//cudaEncoding.CreateOutFile();
	//cudaEncoding.CreateEncoder();*/
	//cudaEncoding.TransCoding(encodeMat, dptr);
	//cudaMemcpy2D(img_src.data, outputWidth, &dptr, pitch, outputWidth, outputHeight, cudaMemcpyDeviceToHost);

	//
	//cudaFree(&dptr);
	//	//cudaMemcpy2D(cudaMat.data, outputWidth * 1.5, cuDeviceptr, pitch, outputWidth * 1.5, outputHeight, cudaMemcpyDeviceToHost);
	//	//cv::imshow("Test", cudaMat);
	//	//cv::cuda::GpuMat dimg(cv::Size(outputWidth, outputHeight), CV_8UC1, (void*)(cuDeviceptr), pitch);

	return 0;
}