#ifndef video_reader_h
#define video_reader_h
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <inttypes.h>
#include <libswresample/swresample.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>

#include <libavutil/timestamp.h>
#include <libavutil/samplefmt.h>
}

struct VideoReaderState {
    int width, height;
    AVRational time_base;

    AVFormatContext* av_format_ctx;

    AVCodecContext* av_codec_ctx;
    AVCodecContext* av_audio_codec_ctx;

    int video_stream_index;
    int audio_stream_index;

    AVFrame* av_frame;
    AVFrame* av_audio_frame;
    AVFrame* gl_frame;
    AVPacket* av_packet;
    SwsContext* sws_scaler_ctx;

};

bool video_reader_open(VideoReaderState* state, const char* filename);
bool video_reader_read_frame(VideoReaderState* state, uint8_t* frame_rgb_buffer, uint8_t* frame_bgr_buffer, int64_t* pts);
void video_reader_close(VideoReaderState* state);
#endif