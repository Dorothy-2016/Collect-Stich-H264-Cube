#include"EncVideo.h"
/*调用windows socket的库函数*/
#pragma comment(lib,"ws2_32.lib")

extern bool flag_send;

/*非阻塞套接字的recv*/
int recv_non_Block(SOCKET socket, char *buffer, int length, int flags)
{
	int recv_len, ret_val, sel;
	struct timeval tm;

	for (recv_len = 0; recv_len < length;)
	{
		/*置读集*/
		fd_set read_fd;
		FD_ZERO(&read_fd);
		FD_SET(socket, &read_fd);
		//等待1s接收不到就返回
		tm.tv_sec = 1;    //1秒
		tm.tv_usec = 1;    //1u秒

						   /*调用select*/
		sel = select(socket + 1, &read_fd, NULL, NULL, &tm);
		if (sel < 0) {   //连接失败
			printf("select socket error: (errno: %d)\n", WSAGetLastError());
			return -1;
		}
		else if (sel == 0) {//超时返回接收的数据
			printf("Recv timout!: (errno: %d)\n", WSAGetLastError());
			return recv_len;
		}
		else {
			if (FD_ISSET(socket, &read_fd)) { //如果真正可写
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

/*非阻塞套接字的send*/
int send_non_Block(SOCKET socket, char *buffer, int length, int flags)
{
	flag_send = false;
	int send_len, ret_val, sel;
	struct timeval tm;

	for (send_len = 0; send_len < length;)
	{
		/*置写集*/
		fd_set write_fd;
		FD_ZERO(&write_fd);
		FD_SET(socket, &write_fd);
		//发送数据量较大，等1s发送不了就返回
		tm.tv_sec =1;    
		tm.tv_usec = 1;    

		sel = select(socket + 1, NULL, &write_fd, NULL, &tm);/*调用select*/
		if (sel <0) {   //连接失败
			printf("select socket error: (errno: %d)\n", WSAGetLastError());
			return -1;
		}
		else if (sel == 0) {
			printf("Send time out(2s)\n");
			return send_len;
		}
		else {
			if (FD_ISSET(socket, &write_fd)) { //如果真正可写
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


/*设置套接字为非阻塞模式*/
int set_non_Block(SOCKET socket)
{
	/*标识符非0允许非阻塞模式*/
	int ret;
	unsigned long flag = 1;
	ret = ioctlsocket(socket, FIONBIO, &flag);
	if (ret)
		printf("set nonblock error: (errno: %d)\n", WSAGetLastError());
	return ret;
}


//初始化socket操作
int server_transfer_Init(SOCKET*connfd, SOCKET *listenfd, WSAData *wsaData)
{
	int clientaddr_len, sel;
	int flag = 1;
	SOCKADDR_IN servaddr, clientaddr;
	//置读集
	fd_set read_fd;
	timeval tt;

	if (WSAStartup(MAKEWORD(2, 2), wsaData)) {  //版本2
		printf("Fail to initialize windows socket!\n");
		return -1;
	}
	//创建一个套接字
	if ((*listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
		printf("create socket error: (errno: %d)\n", WSAGetLastError());
		return -1;
	}
	//设置套接字为非阻塞模式
	if (set_non_Block(*listenfd)) {
		closesocket(*listenfd);
		return -1;
	}
	//初始化套接字*/
	memset(&servaddr, 0, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(PORT_NUMBER);
	servaddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);

	/*设置允许端口重复绑定*/
	if (setsockopt(*listenfd, SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(int)) == -1) {
		printf("set socket option error: (errno: %d)\n", WSAGetLastError());
		closesocket(*listenfd);
		return -1;
	}

	/*绑定端口*/
	if (bind(*listenfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
		printf("bind socket error: (errno: %d)\n", WSAGetLastError());
		closesocket(*listenfd);
		return -1;
	}

	/*监听端口*/
	if (listen(*listenfd, MAX_LIS_NUM) == -1) {
		printf("listen socket error: (errno: %d)\n", WSAGetLastError());
		closesocket(*listenfd);
		return -1;
	}

//#ifdef DEBUG
	printf("======waiting for client's request======\n");
//#endif

	//接受端口连接
	clientaddr_len = sizeof(SOCKADDR_IN);

	//调用select等待accept
	FD_ZERO(&read_fd);
	FD_SET(*listenfd, &read_fd);
	tt.tv_sec = 600;    //50秒超时,等待客户端连接
	tt.tv_usec = 1;    //1u秒

	sel = select(*listenfd + 1, &read_fd, NULL, NULL, &tt);
	if (sel <= 0) {   //连接失败
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

//销毁服务端传输socket
void server_transfer_Destroy(SOCKET *listenfd, SOCKET *connfd)
{
	/* 等待客户端关闭连接 */
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
		Sleep(1000);  //睡眠1s等待关闭
	}
	closesocket(*listenfd);
	WSACleanup();
}
