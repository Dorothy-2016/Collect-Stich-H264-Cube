#include"EncVideo.h"
/*����windows socket�Ŀ⺯��*/
#pragma comment(lib,"ws2_32.lib")

extern bool flag_send;

/*�������׽��ֵ�recv*/
int recv_non_Block(SOCKET socket, char *buffer, int length, int flags)
{
	int recv_len, ret_val, sel;
	struct timeval tm;

	for (recv_len = 0; recv_len < length;)
	{
		/*�ö���*/
		fd_set read_fd;
		FD_ZERO(&read_fd);
		FD_SET(socket, &read_fd);
		//�ȴ�1s���ղ����ͷ���
		tm.tv_sec = 1;    //1��
		tm.tv_usec = 1;    //1u��

						   /*����select*/
		sel = select(socket + 1, &read_fd, NULL, NULL, &tm);
		if (sel < 0) {   //����ʧ��
			printf("select socket error: (errno: %d)\n", WSAGetLastError());
			return -1;
		}
		else if (sel == 0) {//��ʱ���ؽ��յ�����
			printf("Recv timout!: (errno: %d)\n", WSAGetLastError());
			return recv_len;
		}
		else {
			if (FD_ISSET(socket, &read_fd)) { //���������д
				ret_val = recv(socket, buffer + recv_len, length - recv_len, flags);
				if (ret_val < 0) {
					printf("recv error\n");
					return -2;
				}
				else if (ret_val == 0) {
					printf("connection closed\n");
					return 0;
				}
				else
					recv_len += ret_val;
			}
		}
	}
	return recv_len;
}

/*�������׽��ֵ�send*/
int send_non_Block(SOCKET socket, char *buffer, int length, int flags)
{
	flag_send = false;
	int send_len, ret_val, sel;
	struct timeval tm;

	for (send_len = 0; send_len < length;)
	{
		/*��д��*/
		fd_set write_fd;
		FD_ZERO(&write_fd);
		FD_SET(socket, &write_fd);
		//�����������ϴ󣬵�1s���Ͳ��˾ͷ���
		tm.tv_sec =1;    
		tm.tv_usec = 1;    

		sel = select(socket + 1, NULL, &write_fd, NULL, &tm);/*����select*/
		if (sel <0) {   //����ʧ��
			printf("select socket error: (errno: %d)\n", WSAGetLastError());
			return -1;
		}
		else if (sel == 0) {
			printf("Send time out(2s)\n");
			return send_len;
		}
		else {
			if (FD_ISSET(socket, &write_fd)) { //���������д
				ret_val = send(socket, buffer + send_len, length - send_len, flags);
				if (ret_val < 0) {
					printf("send error%d\n", ret_val);
					return -2;
				}
				else if (ret_val == 0) {
					printf("connection closed\n");
					return 0;
				}
				else
					send_len += ret_val;

			}
		}
	}
	flag_send = true;
	return send_len;
}


/*�����׽���Ϊ������ģʽ*/
int set_non_Block(SOCKET socket)
{
	/*��ʶ����0���������ģʽ*/
	int ret;
	unsigned long flag = 1;
	ret = ioctlsocket(socket, FIONBIO, &flag);
	if (ret)
		printf("set nonblock error: (errno: %d)\n", WSAGetLastError());
	return ret;
}


//��ʼ��socket����
int server_transfer_Init(SOCKET*connfd, SOCKET *listenfd, WSAData *wsaData)
{
	int clientaddr_len, sel;
	int flag = 1;
	SOCKADDR_IN servaddr, clientaddr;
	//�ö���
	fd_set read_fd;
	timeval tt;

	if (WSAStartup(MAKEWORD(2, 2), wsaData)) {  //�汾2
		printf("Fail to initialize windows socket!\n");
		return -1;
	}
	//����һ���׽���
	if ((*listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
		printf("create socket error: (errno: %d)\n", WSAGetLastError());
		return -1;
	}
	//�����׽���Ϊ������ģʽ
	if (set_non_Block(*listenfd)) {
		closesocket(*listenfd);
		return -1;
	}
	//��ʼ���׽���*/
	memset(&servaddr, 0, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(PORT_NUMBER);
	servaddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);

	/*��������˿��ظ���*/
	if (setsockopt(*listenfd, SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(int)) == -1) {
		printf("set socket option error: (errno: %d)\n", WSAGetLastError());
		closesocket(*listenfd);
		return -1;
	}

	/*�󶨶˿�*/
	if (bind(*listenfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
		printf("bind socket error: (errno: %d)\n", WSAGetLastError());
		closesocket(*listenfd);
		return -1;
	}

	/*�����˿�*/
	if (listen(*listenfd, MAX_LIS_NUM) == -1) {
		printf("listen socket error: (errno: %d)\n", WSAGetLastError());
		closesocket(*listenfd);
		return -1;
	}

//#ifdef DEBUG
	printf("======waiting for client's request======\n");
//#endif

	//���ܶ˿�����
	clientaddr_len = sizeof(SOCKADDR_IN);

	//����select�ȴ�accept
	FD_ZERO(&read_fd);
	FD_SET(*listenfd, &read_fd);
	tt.tv_sec = 600;    //50�볬ʱ,�ȴ��ͻ�������
	tt.tv_usec = 1;    //1u��

	sel = select(*listenfd + 1, &read_fd, NULL, NULL, &tt);
	if (sel <= 0) {   //����ʧ��
		printf("select socket error: (errno: %d)\n", WSAGetLastError());
		return -1;
	}

	if ((*connfd = accept(*listenfd, (struct sockaddr*)&clientaddr, &clientaddr_len)) == -1) {
		printf("accept socket error: (errno: %d)\n", WSAGetLastError());
		closesocket(*listenfd);
		return -1;
	}
//#ifdef DEBUG
	char buffer[512];
	inet_ntop(AF_INET, &clientaddr.sin_addr, buffer, 512);
	printf("client IP:%s, port:%d, connected\n", buffer, ntohs(clientaddr.sin_port));
//#endif
	return 0;
}

//���ٷ���˴���socket
void server_transfer_Destroy(SOCKET *listenfd, SOCKET *connfd)
{
	/* �ȴ��ͻ��˹ر����� */
	char buffer[256];
	int ret;
	while (1)
	{
		ret = recv(*connfd, buffer, 256, 0);
		if (ret <= 0) {
			closesocket(*connfd);
			printf("Client close\n");
			break;
		}
		Sleep(1000);  //˯��1s�ȴ��ر�
	}
	closesocket(*listenfd);
	WSACleanup();
}
