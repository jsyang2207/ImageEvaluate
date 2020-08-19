#include "pch.h"
#include "video_reader.h"

#include <Simd/SimdBase.h>
#include <Simd\SimdBaseYuvToBgr.cpp>
#include <Simd\SimdBaseBgrToYuv.cpp>
// av_err2str returns a temporary array. This doesn't work in gcc.
// This function can be used as a replacement for av_err2str.


static const char* av_make_error(int errnum) {
    static char str[AV_ERROR_MAX_STRING_SIZE];
    memset(str, 0, sizeof(str));
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}



bool video_reader_open(VideoReaderState* state, const char* filename) {

    auto& width = state->width;
    auto& height = state->height;
    auto& time_base = state->time_base;
    auto& av_format_ctx = state->av_format_ctx;
    auto& av_codec_ctx = state->av_codec_ctx;
    auto& av_audio_codec_ctx = state->av_audio_codec_ctx;

    auto& video_stream_index = state->video_stream_index;
    auto& audio_stream_index = state->audio_stream_index;
    auto& av_frame = state->av_frame;
    auto& av_audio_frame = state->av_audio_frame;
    auto& av_packet = state->av_packet;

    av_format_ctx = avformat_alloc_context();

    if (!av_format_ctx) {
        printf("Couldn't created AVFormatContext\n");
        return false;
    }

    int ret;
    char error[64];
    ret = avformat_open_input(&av_format_ctx, filename, NULL, NULL);
    printf("%s\n", av_make_error(ret));
    if (avformat_open_input(&av_format_ctx, filename, NULL, NULL) != 0) {

        printf("Couldn't open video file\n");
        return false;
    }
    //Video Codec Open
    if (avformat_find_stream_info(av_format_ctx, NULL) < 0) {
        printf("Could not find stream infomation\n");
        return false;
    }
    video_stream_index = -1;
    AVCodecParameters* av_codec_params = NULL;
    AVCodec* av_codec = NULL;
    int VSI = av_find_best_stream(av_format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (VSI < 0) {
        printf("Could not find stream in input file\n");
        return false;
    }
    else {
        for (int i = 0; i < av_format_ctx->nb_streams; ++i) {
            av_codec_params = av_format_ctx->streams[i]->codecpar;
            av_codec = avcodec_find_decoder(av_codec_params->codec_id);
            if (!av_codec) {
                continue;
            }
            if (av_codec_params->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index = i;
                width = av_codec_params->width;
                height = av_codec_params->height;
                time_base = av_format_ctx->streams[i]->time_base;

                break;
            }
        }
    }
    if (video_stream_index == -1) {
        printf("Couldn't find valid video stream inside file\n");
        return false;
    }
    av_codec_ctx = avcodec_alloc_context3(av_codec);
    if (!av_codec_ctx) {
        printf("Couldn't create AVCodecContext\n");
        return false;
    }
    if (avcodec_parameters_to_context(av_codec_ctx, av_codec_params) < 0) {
        printf("Couldn't initialize AVCodecContext\n");
        return false;
    }
    if (avcodec_open2(av_codec_ctx, av_codec, NULL) < 0) {
        printf("Couldn't open codec\n");
        return false;
    }

    //Audio Codec Open
    /*
    audio_stream_index = -1;
    AVCodecParameters* av_audio_codec_params = NULL;
    av_codec = NULL;

    int ASI = av_find_best_stream(av_format_ctx, AVMEDIA_TYPE_AUDIO, -1, VSI, NULL, 0);
    if (ASI < 0) {
        printf("error string : %s\n", av_make_error_string(error, 64, ASI));
        printf("Could not find audio in input file\n");
        return false;
    }
    else {
        for (int i = 0; i < av_format_ctx->nb_streams; ++i) {
            av_audio_codec_params = av_format_ctx->streams[i]->codecpar;
            av_codec = avcodec_find_decoder(av_audio_codec_params->codec_id);
            if (!av_codec) {
                continue;
            }
        }
    }
    if (audio_stream_index != -1) {
        printf("Couldn't find valid audio stream inside file\n");
        return false;
    }

    av_codec = avcodec_find_decoder(av_audio_codec_params->codec_id);
    if (!av_codec) {
        printf("Couldn't find decoder\n");
    }
    av_audio_codec_ctx = avcodec_alloc_context3(av_codec);
    if (!av_audio_codec_ctx) {
        printf("Couldn't create Audio AVCodecContext\n");
        return false;
    }
    if (avcodec_parameters_to_context(av_audio_codec_ctx, av_audio_codec_params) < 0) {
        printf("Couldn't initialize AVCodecContext\n");
        return false;
    }

    if (avcodec_open2(av_audio_codec_ctx, av_codec, NULL) < 0) {
        printf("Couldn't open codec\n");
        return false;
    }


    int nb_planes;
    nb_planes = av_sample_fmt_is_planar(av_audio_codec_ctx->sample_fmt) ? av_audio_codec_ctx->channels : 1;

    */

    av_frame = av_frame_alloc();
    if (!av_frame) {
        printf("Couldn't allocate AVFrame\n");
        return false;
    }
    av_audio_frame = av_frame_alloc();
    if (!av_audio_frame) {
        printf("Couldn't allocate Audio AVFrame\n");
        return false;
    }
    //av_dump_format(av_format_ctx, 0, filename, 1);
    av_packet = av_packet_alloc();
    if (!av_packet) {
        printf("Couldn't allocate AVPacket\n");
        return false;
    }

    return true;
}


bool video_reader_read_frame(VideoReaderState* state, uint8_t* frame_rgb_buffer, uint8_t* frame_bgr_buffer,int64_t* pts) {
    auto& width = state->width;
    auto& height = state->height;
    auto& time_base = state->time_base;
    auto& av_format_ctx = state->av_format_ctx;
    auto& av_codec_ctx = state->av_codec_ctx;
    auto& av_audio_codec_ctx = state->av_audio_codec_ctx;

    auto& video_stream_index = state->video_stream_index;
    auto& audio_stream_index = state->audio_stream_index;
    auto& av_frame = state->av_frame;
    auto& gl_frame = state->gl_frame;
    auto& av_audio_frame = state->av_audio_frame;
    auto& av_packet = state->av_packet;
    int response;
    while (av_read_frame(av_format_ctx, av_packet) >= 0) {
        //&& av_packet->stream_index != 1
        
        if (av_packet->stream_index != video_stream_index) {
            av_packet_unref(av_packet);
            continue;
        }
        
        /*
        else if (av_packet->stream_index == 1) {
            response = avcodec_send_packet(av_audio_codec_ctx, av_packet);
            if (response < 0) {
                printf("Failed to decode packet: %s\n", av_make_error(response));
                return false;
            }
            response = avcodec_receive_frame(av_audio_codec_ctx, av_audio_frame);
            if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                av_packet_unref(av_packet);
                continue;
            }
            else if (response < 0) {
                printf("Failed to decode packet: %s\n", av_make_error(response));
                return false;
            }
        }*/
        else if (av_packet->stream_index == video_stream_index) {
            response = avcodec_send_packet(av_codec_ctx, av_packet);
            if (response < 0) {
                printf("Failed to decode packet: %s\n", av_make_error(response));
                return false;
            }
            response = avcodec_receive_frame(av_codec_ctx, av_frame);
            if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                av_packet_unref(av_packet);
                continue;
            }
            else if (response < 0) {
                printf("Failed to decode packet: %s\n", av_make_error(response));
                return false;
            }

            //av_image_copy(video_dst_data, video_dst_linesize, (const uint8_t**)(av_frame->data)
            //    , av_frame->linesize, av_codec_ctx->pix_fmt, av_codec_ctx->width, av_codec_ctx->height);
            //fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);
        }
      
        av_packet_unref(av_packet);
        break;
    }
    
    *pts = av_frame->pts;
    AVFrame* frame = av_frame_alloc();
    frame->width = width;
    frame->height = height;
    frame->format = AV_PIX_FMT_YUV420P;

    SwsContext* sws_scaler_ctx;
    SwsContext* sws_scaler_ctx1;
    sws_scaler_ctx = sws_getContext(width, height, AV_PIX_FMT_YUV420P, 1920, 1080, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
    sws_scaler_ctx1 = sws_getContext(width, height, AV_PIX_FMT_YUV420P, 1920, 1080, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);


    gl_frame = av_frame_alloc();

    uint8_t* dest[4] = { frame_rgb_buffer, NULL, NULL, NULL };
    int dest_linesize[4] = { width * 3, 0, 0, 0 };
    sws_scale(sws_scaler_ctx, av_frame->data, av_frame->linesize, 0, av_frame->height, dest, dest_linesize);

    uint8_t* dest1[4] = { frame_bgr_buffer, NULL, NULL, NULL };
    int dest_linesize1[4] = { width * 3, 0, 0, 0 };
    sws_scale(sws_scaler_ctx1, av_frame->data, av_frame->linesize, 0, av_frame->height, dest1, dest_linesize1);


    //uint8_t* inputBufferY;
    //inputBufferY = av_frame->data[0];
    //uint8_t* inputBufferU = av_frame->data[1];
    //uint8_t* inputBufferV = av_frame->data[2];
    //size_t Ybuffer = sizeof(av_frame->data[0]);
    //size_t Ubuffer = sizeof(av_frame->data[1]);
    //size_t Vbuffer = sizeof(av_frame->data[2]);
    //memcpy(av_frame->data[0], inputBufferY, av_frame->linesize[0] * frame->height);
    //memcpy(av_frame->data[1], inputBufferU, av_frame->linesize[1] * frame->height / 2);
    //memcpy(av_frame->data[2], inputBufferV, av_frame->linesize[2] * frame->height / 2);

    ////Simd::Base::Yuv420pToRgb(inputBufferY, width, inputBufferU, width / 2, inputBufferV, width / 2, width, height, frame_rgb_buffer, width * 3);
    //Simd::Base::Yuv420pToBgr(inputBufferY, width, inputBufferU, width / 2, inputBufferV, width / 2, width, height, frame_bgr_buffer, width * 3);


    //Simd::Base::BgrToYuv420p(frame_bgr_buffer, width,height,width*3, inputBufferY, width, inputBufferU, width / 2, inputBufferV, width / 2);
    //Simd::Base::BgrToYuv420p(inputBufferY, width, inputBufferU, width / 2, inputBufferV, width / 2, width, height, frame_bgr_buffer, width * 3);
    //SimdYuv420pToRgb(inputBufferY, width, inputBufferU, width / 2, inputBufferV, width / 2, width, height, frame_rgb_buffer, width * 3);
    //SimdYuv420pToBgr(inputBufferY, width, inputBufferU, width / 2, inputBufferV, width / 2, width, height, frame_bgr_buffer, width * 3);
    return true;
}
void video_reader_close(VideoReaderState* state) {
    sws_freeContext(state->sws_scaler_ctx);
    avformat_close_input(&state->av_format_ctx);
    avformat_free_context(state->av_format_ctx);
    av_frame_free(&state->av_frame);
    av_packet_free(&state->av_packet);
    avcodec_free_context(&state->av_codec_ctx);
}