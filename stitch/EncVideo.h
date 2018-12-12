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

/*�������������г���*/
#define MAX_LIS_NUM 10
#define PORT_NUMBER 8088
#define FRAME_PER_SECOND 30     //֡��
#define PICTURE_WIDTH 2048		//ͼƬ���
#define PICTURE_HEIGHT 1024     //ͼƬ�߶�
#define PIC_SIZE  PICTURE_WIDTH*PICTURE_HEIGHT

//����ѡ��
//#define DEBUG    //��д�ļ�
//#define FRAME_INFO //ÿ֡�����Ϣ

//����socket������
int set_non_Block(SOCKET socket);	
//��������������
int send_non_Block(SOCKET socket, char *buffer, int length, int flags);
//��������������
int recv_non_Block(SOCKET socket, char *buffer, int length, int flags);
//��ʼ��socket����
int server_transfer_Init(SOCKET *listenfd, SOCKET *connfd, struct WSAData *wsaData);
//���ٷ���˴���socket
void server_transfer_Destroy(SOCKET *listenfd, SOCKET *connfd);

//��ʼ��x264������
int x264_encoder_Init(AVCodecContext **c, AVFrame **frame, AVPacket **pkt, uint8_t **pic_inbuff,int *seq_number);
//����x264������
void x264_encoder_Destroy(AVCodecContext **c, AVFrame **frame, AVPacket **pkt,uint8_t **pic_inbuff,FILE *fout);
//�������ң�����ļ�ָ��  ���뻺�� ԭʼyuvͼ��ÿ֡�Ĵ�С ÿ֡ͼ����
void x264_encodeVideo(SOCKET sockfd,AVCodecContext *c, AVFrame *frame, AVPacket *pkt, uint8_t *pic_inbuff ,int seq_number,FILE *fout);
//������������ѱ������������
void x264_encoder_Flush(SOCKET sockfd,AVCodecContext *c, AVPacket *pkt,FILE *fout);

