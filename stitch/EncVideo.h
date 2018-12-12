#pragma once

#include <stdio.h>
#include <WinSock2.h>
#include <windows.h>
#include<time.h>
#include <WS2tcpip.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavformat/avformat.h>
}

/*定义最大监听队列长度*/
#define MAX_LIS_NUM 10
#define PORT_NUMBER 8088
#define FRAME_PER_SECOND 30     //帧率
#define PICTURE_WIDTH 2048		//图片宽度
#define PICTURE_HEIGHT 1024     //图片高度
#define PIC_SIZE  PICTURE_WIDTH*PICTURE_HEIGHT

//调试选项
//#define DEBUG    //读写文件
//#define FRAME_INFO //每帧输出信息

//设置socket非阻塞
int set_non_Block(SOCKET socket);	
//非阻塞发送数据
int send_non_Block(SOCKET socket, char *buffer, int length, int flags);
//非阻塞接收数据
int recv_non_Block(SOCKET socket, char *buffer, int length, int flags);
//初始化socket操作
int server_transfer_Init(SOCKET *listenfd, SOCKET *connfd, struct WSAData *wsaData);
//销毁服务端传输socket
void server_transfer_Destroy(SOCKET *listenfd, SOCKET *connfd);

//初始化x264编码器
int x264_encoder_Init(AVCodecContext **c, AVFrame **frame, AVPacket **pkt, uint8_t **pic_inbuff,int *seq_number);
//销毁x264编码器
void x264_encoder_Destroy(AVCodecContext **c, AVFrame **frame, AVPacket **pkt,uint8_t **pic_inbuff,FILE *fout);
//从左至右：输出文件指针  输入缓存 原始yuv图像每帧的大小 每帧图像编号
void x264_encodeVideo(SOCKET sockfd,AVCodecContext *c, AVFrame *frame, AVPacket *pkt, uint8_t *pic_inbuff ,int seq_number,FILE *fout);
//清出缓冲区，把编码器里的内容
void x264_encoder_Flush(SOCKET sockfd,AVCodecContext *c, AVPacket *pkt,FILE *fout);

