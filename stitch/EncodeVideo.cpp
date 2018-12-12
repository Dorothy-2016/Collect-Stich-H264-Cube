#include "EncVideo.h"

extern bool flag_send;

//初始化操作(编码器，输入帧，输出包，原始一帧yuv图像缓存)
int x264_encoder_Init(AVCodecContext **c,  AVFrame **frame, AVPacket **pkt, uint8_t **pic_inbuff,int *seq_number)
{
	const AVCodec *codec;
	int ret;
	seq_number = 0;
	//查找需要的编码器
	codec = avcodec_find_encoder_by_name("libx264");
	if (!codec) {
		fprintf(stderr, "Codec libx264 not found\n");
		exit(1);
	}
	//按照编码器的默认值分配AVCodecContext内容以及默认值
	*c = avcodec_alloc_context3(codec);
	if (!(*c)) {
		fprintf(stderr, "Could not allocate video codec context\n");
		exit(1);
	}

	(*c)->bit_rate = 10000000;  //编码比特率
	(*c)->width = PICTURE_WIDTH;    //输入分辨率
	(*c)->height = PICTURE_HEIGHT;
	(*c)->time_base = AVRational{ 1, FRAME_PER_SECOND };
	(*c)->framerate = AVRational{ FRAME_PER_SECOND, 1 };
	(*c)->gop_size = 8;  //每GOP帧做一次帧内预测
	(*c)->max_b_frames = 3;
	(*c)->pix_fmt = AV_PIX_FMT_YUV420P;
	(*c)->qmin = 10;
	(*c)->qmax = 51;

	if (codec->id == AV_CODEC_ID_H264) {
		av_opt_set((*c)->priv_data, "preset", "medium", 0);
		//av_opt_set(c->priv_data, "tune", "zerolatency", 0);
	}

	//打开编码器
	ret = avcodec_open2(*c, codec, NULL);
	if (ret < 0) {
		fprintf(stderr, "Could not open codec: %s\n", av_err2str(ret));
		exit(1);
	}

	//申请帧操作方法,通过format/width/height三个参数来确定帧缓存大小
	*frame = av_frame_alloc();
	if (!frame) {
		fprintf(stderr, "Could not allocate video frame\n");
		exit(1);
	}
	(*frame)->format = (*c)->pix_fmt;
	(*frame)->width =(*c)->width;
	(*frame)->height = (*c)->height;
	ret = av_frame_get_buffer(*frame, 32);     //后一个参数需要根据电脑CPU型号选择对齐方式，0表示自适应
	if (ret < 0) {
		fprintf(stderr, "Could not allocate the video frame data\n");
		exit(1);
	}

	ret = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, PICTURE_WIDTH, PICTURE_HEIGHT, 32);  //32位对齐方式
	*pic_inbuff = (uint8_t *)malloc(ret * sizeof(uint8_t)); 
	//编码后负载数据包的大小和分配
	*pkt = av_packet_alloc();
	if (!pkt)
		exit(1);
	av_new_packet(*pkt, ret);   //data_buffer,需要包括每帧头的大小
	return 1;
}

//销毁申请内存
void x264_encoder_Destroy(AVCodecContext **c, AVFrame **frame, AVPacket **pkt,uint8_t **pic_inbuff,FILE*fout)
{
#ifdef DEBUG
	fclose(fout);
#endif // DEBUG*/
	avcodec_free_context(c);
	av_frame_free(frame);
	av_packet_free(pkt);
	free(*pic_inbuff);
}



//从左至右：输出文件指针  输入缓存 原始yuv图像每帧的大小 每帧图像编号
void x264_encodeVideo(SOCKET sockfd, AVCodecContext *c, AVFrame *frame, AVPacket *pkt, uint8_t *pic_inbuff,int seq_number,FILE*fout)
{	
	int ret;
	frame->data[0] = pic_inbuff;  //Y分量
	frame->data[1] = pic_inbuff + PIC_SIZE; //U分量
	frame->data[2] = pic_inbuff + PIC_SIZE * 5 / 4; //V分量
	frame->pts = seq_number;   //编码一帧输出一帧
	
	//将一帧图像送入编码器
#ifdef FRAME_INFO
	if (frame)
		printf("Send frame %3" PRId64"\n", frame->pts);
#endif
	ret = avcodec_send_frame(c, frame);
	if (ret < 0) {
		fprintf(stderr, "Error sending a frame for encoding\n");
		exit(1);
	}

	while (ret >= 0) {
		ret = avcodec_receive_packet(c, pkt);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) //现在还不能取出编码后的码流
			return;
		else if (ret < 0) {    //编码错误
			fprintf(stderr, "Error during encoding\n");
			exit(1);
		}
#ifdef FRAME_INFO
		printf("Write packet %3" PRId64" (size=%5d)\n", pkt->pts, pkt->size);
#endif
#ifdef DEBUG
		fwrite(pkt->data, 1, pkt->size, fout);
#endif // DEBUG
		if (send_non_Block(sockfd, (char*)pkt->data, pkt->size, 0)!=pkt->size) {
			printf("data loss!\n");
		}		
		av_packet_unref(pkt);
	}
}

//清出缓冲区，把编码器里的内容
void x264_encoder_Flush(SOCKET sockfd,AVCodecContext *c, AVPacket *pkt,FILE *fout)
{
	int ret;
	//读出编码器缓存
	ret = avcodec_send_frame(c, NULL);
	if (ret < 0) {
		fprintf(stderr, "Error sending a frame for encoding\n");
		exit(1);
	}

	while (ret >= 0) {
		ret = avcodec_receive_packet(c, pkt);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) //现在还不能取出编码后的码流
			return;
		else if (ret < 0) {    //编码错误
			fprintf(stderr, "Error during encoding\n");
			exit(1);
		}
#ifdef FRAME_INFO
		printf("Write packet %3" PRId64" (size=%5d)\n", pkt->pts, pkt->size);
#endif
#ifdef DEBUG
		fwrite(pkt->data, 1, pkt->size, fout);
#endif // DEBUG
		if (send_non_Block(sockfd, (char*)pkt->data, pkt->size, 0) != pkt->size) {
			printf("data loss!\n");
		}
		av_packet_unref(pkt);
	}
	// 添加h264文件结束标志0,0,0,0xb7
	uint8_t endcode[] = { 0, 0, 1, 0xb7 };
#ifdef FRAME_INFO
	printf("Write H.264 end symbol!\n");
#endif // FRAME_INFO

#ifdef DEBUG
	fwrite(endcode, 1, sizeof(endcode), fout);
#endif // DEBUG*/
	send_non_Block(sockfd, (char*)endcode, 4, 0);
}


