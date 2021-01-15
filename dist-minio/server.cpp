#include <iostream>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <thread> 
#include <cassert>
#include <algorithm>
#include "commands.h"
#include <chrono>
#include <mutex>
using namespace std;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::microseconds us;
typedef std::chrono::duration<float> fsec;

//std::mutex safecout::_mutexPrint{};

#define PORT 5555
#define BACKLOG 5
#define BUFSIZE 4096 

/* Creates a socket at a given port and starts
 * listening on the port for connections
 * This function must be called, once per new 
 * client, so that there exists an independent
 * connection between the server and each clien
 * The client here is the dataloader
 * Only returns valid fd
 * */

int initialize_socket(int port){
	int server_fd;
	int bnd, lstn;
	struct sockaddr_in server;
	
	server_fd = socket(AF_INET, SOCK_STREAM, 0);
	if(server_fd < 0){
		cerr << "Error creating socket : " << strerror(errno) << endl;
		return -1;
	}
	server.sin_family = AF_INET;
	server.sin_port = htons(port);
	server.sin_addr.s_addr = INADDR_ANY;

	set_tcp_nodelay(server_fd);
	set_recv_window(server_fd, 1500000);
	set_send_window(server_fd, 1500000);
	set_mss(server_fd, 1460);
	
	bnd = bind(server_fd, (struct sockaddr *)&server, sizeof(server));
	if(bnd < 0) {
		safecout{} <<"Error binding : " << strerror(errno) << endl;
		return -1;
	}
	safecout{}	<< "Socket created at " << port << endl;

	lstn = listen(server_fd, BACKLOG);
	if(lstn < 0){
		safecout{} <<"Error listening at " << port << " : " << strerror(errno) << endl;
		return -1;
	}
	safecout{}	<< "Server is listening on port " << port << endl;
	//safecout{}	<< "-----------------------" << endl;
	//print_socket_options(server_fd);
	//safecout{}	<< "-----------------------" << endl;
	return server_fd;
}

int send_header(long filesize, int client_fd) {
	unsigned char * p = (unsigned char*)&filesize;
	int s = send(client_fd, p, HEADER_SIZE, 0);
	if (s < 0){
		cerr << "Error sending header : " << strerror(errno) << endl;
		return -1;
	}
	//safecout{}	<< "Sent header : " << s << " bytes. Value = " <<  *(reinterpret_cast<unsigned long *>(p)) << endl;
	return s;
}



/* Given a sample in cache, sends it to the
 * requested client over the network
 * Returns the number of buytes written or 
 * -1 for failure
 */
int send_sample(string fname, int client_fd, us &header_time, us &body_time, int id,	bool must_send_header){

	struct stat filestatus; 
	stat(fname.c_str(), &filestatus );		
	long filesize = filestatus.st_size;		

	if (filesize <= 0) {
		cerr << "Couldn't get filesize for " << fname << endl;
		return -1;
	}


	char* buffer;
	buffer = (char*) malloc(filesize*sizeof(char));
	if (buffer == NULL){
		cerr << "Cannot allocate buffer " << strerror(errno) << endl;
		return -1;
	}

	int sent_bytes = 0, s = 0, h = 0;
	//safecout{}	<< "Sending " << filesize << " to " << client_fd << endl;
	auto start = Time::now();
	
	if (must_send_header) { 
		h = send_header(filesize, client_fd);
		if (h < 0){
			cerr << "Filename : " << fname << ", header = " << filesize << " size " << h << endl;
			return -1;
		}
	}
	auto finish = Time::now();
	auto diff = (finish-start);
	header_time = std::chrono::duration_cast<us>(diff);
	auto start1 = Time::now();

	FILE * filp = fopen(fname.c_str(), "rb"); 
	if(!filp) {
		cerr << "File opening failed for " << fname << ": " << strerror(errno) << endl;
		return -1;
	} 
	int bytes_read = fread(buffer, sizeof(char), filesize, filp);  
	fclose (filp);

	if (bytes_read != filesize){
		cerr << "Bytes read mismatch for " << fname << " : " << strerror(errno) << endl;
		cerr << "Expected " << filesize << " , read " << bytes_read << endl;
		return -1;
	}

	int SEND_SIZE = BUFSIZE;
	do {
		int remaining = filesize - sent_bytes;
		SEND_SIZE = (remaining > BUFSIZE ? BUFSIZE : remaining );
		//s = send(client_fd, buffer + s, filesize, 0);  
		s = send(client_fd, buffer + sent_bytes, SEND_SIZE, 0);  
		sent_bytes += s;		
		//safecout{}	<< "\tsent " << sent_bytes << endl;  
		if ( s < 0 ){			
			cerr << "error sending " << fname << " : " << strerror(errno) << endl;	
			return -1;			
		}
	} while (sent_bytes < filesize);

	free(buffer);

	//safecout{}	<< "Sent " << sent_bytes << " for file of size " << filesize << " : " << fname << endl;
	auto finish1 = Time::now();
	auto diff1 = finish1 - start1;
	body_time = std::chrono::duration_cast<us>(diff1);
	return sent_bytes;
}


int send_notfound(int client_fd){
	char not_found[HEADER_SIZE] = "NOFOUND";
	safecout{}	<< "Sending NOTFOUND to " << client_fd << " : " << HEADER_SIZE << endl;
	int s = send(client_fd, not_found, HEADER_SIZE, 0); 
	if ( s < 0 ){			
		cerr << "error sending NOTFOUND : " << strerror(errno) << endl;  
		return -1;		 
	}
}



/* In an infinite loop, waits for requets from client
 * and processes the requests.
 * Terminates when client sends an TERMINATE command
 * Assumes max get request size of 100 bytes - filename
 */
int wait_for_requests(int server_fd, int client_fd, us &header_time, us &body_time, int id, bool must_send_header){
	char request[REQUEST_SIZE];
	int retval = recv(client_fd, request, REQUEST_SIZE ,0);
	
	if (retval < 0) {
		// We could be reading from a closed socket
		safecout{}	<< "No request from " << client_fd << endl;
		return SOCK_ERROR;
	}
	else if (retval == 0){
		//safecout{}	<< "Client conn is closed " << client_fd << endl;
		return SOCK_CLOSED;
	}

	string fname = filename(request);
	//safecout{}	<< "Received request " << request << ", fname = " << fname << endl;
		
	//If fname is cached, return it. Else return NOTFOUND
	if (is_cached(fname)){

		string full_path = file_path(fname);
		//safecout{}	<< "Sending sample for " << full_path << endl;
		int bytes_sent = send_sample(full_path, client_fd, header_time, body_time, id, must_send_header);
		if (bytes_sent < 0) {
			cerr << "Error sending file" << endl;
			return -1;
		}
			return bytes_sent;
	}

	else{
		int ret =  send_notfound(client_fd);
		if (ret < 0) {
			cerr << "Error sending file " << strerror(errno) << endl;
			return -1;
		}
		return 1;
	}
	//safecout{}	<< "request serviced for " << server_fd << " : " << client_fd << endl;
}



/* Takes an active socket and starts listening for 
 * new connections. Returns when one
 * connection is made
 * It is incorrect to have two connections on one
 * socket. Return sthe fd of the connected client
 */
int wait_for_connection(int server_fd, int client_fd, int id, bool must_send_header){
	//string ofname =  "outfile-" + to_string(server_fd) + ".log";
	//std::ofstream outfile;	
	//outfile.open(ofname);		
	safecout{}	<< "Waiting for connection" << endl;
	struct sockaddr_in client;
	socklen_t len=sizeof(client);

	while(client_fd = accept(server_fd, (struct sockaddr*)&client, &len)) {
		set_mss(server_fd, 1460);
		if( client_fd < 0){
			safecout{}	<< "Error accepting " << server_fd << " : " << strerror(errno) << endl;
			return -1;
		}
		safecout{}	<< "[" << id << "] Connection made to " << client_fd << endl;
		print_socket_options(server_fd, id=id);
		print_socket_options(client_fd);
		us header_time; 
		us body_time;
		float total_header_time = 0;
		float total_body_time = 0;
		float total_body_size = 0;
		float total_header_size = 0;
		while(1){
			//safecout{}	<< "starting one request for serv " << server_fd << " and client " << client_fd << endl;
			int retval = wait_for_requests(server_fd, client_fd, header_time, body_time, id, must_send_header);
			if (retval == SOCK_ERROR || retval == SOCK_CLOSED){
				break;
			}
			total_header_time += header_time.count();
			total_body_time += body_time.count();
			total_header_size += HEADER_SIZE;
			total_body_size += retval;	
			//safecout{}	<< "Done one request for serv " << server_fd << " and client " << client_fd << endl;
		}
		close(client_fd);
		if (must_send_header){
		 safecout{}	<< "[" << id << "]" <<  string(50, '-') << "\n"
								<< "[" << id << "] Header size = " << total_header_size << " B" 
								<< "		Time = " << total_header_time/1024/1024  << " s"
								<< "		SPEED = " << total_header_size/total_header_time << " MBps\n" 
				 				<< "[" << id << "]" <<  string(50, '-') << endl;
		}

		safecout{}	<< "[" << id << "]" <<	string(50, '-') << "\n"
				 				<< "[" << id << "] Closed conn for client " << client_fd << "\n"
				 				<< "[" << id << "] Payload size = " << total_body_size << " B" 
				 				<< "		 Time = " << total_body_time/1024/1024	<< " s"
				 				<< "		 SPEED = " << total_body_size/total_body_time << " MBps\n"
				 				<< "[" << id << "] Listening again \n"
				 				<< "[" << id << "]" <<  string(50, '-') << endl;

	}
	safecout{}	<< "[" << id << "] Returning from client " << client_fd << endl;
	return 0;
}



int main(int argc, char** argv) {

	if (argc < 3){
		safecout{}	<< "Entered " << argc << " args. Pass in num clients and start port" << endl;
		return -1;
	}

	bool must_send_header = false;
	if (argc > 3) {
		string action(argv[3]);
		if (action == "header")
			must_send_header = true;
	}

	int num_clients = atoi(argv[1]);
	int port_start = atoi(argv[2]);
	safecout{}	<< "Expecting num clients = " << num_clients << endl;

	vector<int> server_fd(num_clients, 0);
	vector<int> client_fd(num_clients, 0);

	for (int i =0; i < num_clients; i++){
		int port = port_start + i;
		int fd = initialize_socket(port);
		if (fd < 0){
			cerr << "Exiting due to failure" << endl;
			return -1;
		}
		server_fd[i] = fd;
	}
	assert(server_fd.size() == num_clients);
	
	// The server was all serial so far
	// Now spawn one thread each to handle every connection
	// In each thread, wait forever to receive requests
	vector<thread> thread_list(num_clients);

	for (int i = 0; i < num_clients; i ++){
		thread_list[i] = thread(wait_for_connection, server_fd[i], client_fd[i], i, must_send_header);
		safecout{}	<< "Created thread "	<< i << endl;
	}


	safecout{}	<< "Must wait to join now" << endl;
	// Wait for all threads to finish
	for (int i = 0; i < num_clients; i ++) {
		safecout{}	<< "Wait " << i << endl;
		thread_list[i].join();
	}
	return 0;
}
		
