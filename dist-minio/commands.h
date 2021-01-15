#ifndef COMMANDS_H
#define COMMANDS_H
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/uio.h>
#include <sys/syscall.h>
#include <sys/syscall.h>
#include <netinet/tcp.h>
#include <cstring>
#include <mutex>
#include <sstream>
#define REQUEST_SIZE 100
#define GET_SIZE 4
#define HEADER_SIZE 8
#define SOCK_CLOSED -2
#define SOCK_ERROR -3
#define SUCCESS 2
#define NOT_FOUND "NOTFOUND"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;
typedef std::chrono::milliseconds ms;
typedef std::chrono::microseconds us;
typedef std::chrono::seconds s;
typedef std::chrono::duration<float> fsec;

std::string prefix = "/dev/shm/cache/";



/** Thread safe cout class
	* Exemple of use:
	*		 safecout{} << "Hello world!" << std::endl;
	*/
class safecout: public std::ostringstream {
public:
	safecout() = default;

	~safecout(){
		std::lock_guard<std::mutex> guard(_mutexPrint);
		std::cout << this->str();
	}

private:
	static std::mutex _mutexPrint;
};

std::mutex safecout::_mutexPrint{};


 
/* Gets a request of length 100 bytes
 * and strips the filename from the request
 * Request of the type GET filename.jpeg
 */
std::string filename(const char* request) {
	int ret = strncmp(request, "GET ", GET_SIZE);
	if (ret != 0){
		std::cerr << "Received unknown request " << request << std::endl;
		return NULL;
	}
	std::string fname(request + 4);
	return	fname;
}


bool is_cached(std::string filename){
	std::string path = prefix + filename;
	struct stat		buffer;
	return (stat (path.c_str(), &buffer) == 0);
}

std::string file_path(std::string filename){
	return prefix + filename;
}

int compare_files(const char* origfile, const char* newfile){
	FILE *forig, *fnew;
	long lorig, lnew;
	char corig, cnew;

	forig = fopen(origfile, "r");
	fnew = fopen(newfile, "r");

	struct stat fstat;
	stat(origfile, &fstat);
	lorig = fstat.st_size;

	stat(newfile, &fstat);
	lnew = fstat.st_size;

	if (lnew != lorig){
		std::cerr <<"File lengths mismatch" <<
								"\tOrig : " << lorig <<
								"\tNew : " << lnew << std::endl;
		return -1;
	}

	for (int i = 0; i < lorig; i++){
		fread(&corig, sizeof(char), 1, forig);
		fread(&cnew, sizeof(char), 1, fnew);
		if ( corig != cnew ){
			std::cerr << corig << " mismatch with " << cnew << " at pos " << i << std::endl;
			return -1;
		}
	}
	return 0;
}


int compare_buffers(uint8_t* recv_buffer, std::string full_path, int filesize) {
	uint8_t* orig_buffer = (uint8_t*)malloc(filesize*sizeof(uint8_t));
	if (orig_buffer == NULL){
		std::cerr << "Cannot allocate buffer " << strerror(errno) << std::endl;
		return -1;
	}


	FILE * filp = fopen(full_path.c_str(), "rb"); 
	if(!filp) { 
		 std::cerr << "File opening failed for " << full_path << ": " << strerror(errno) << std::endl;		 
		return -1;
	}

	int bytes_read = fread(orig_buffer, sizeof(uint8_t), filesize, filp);
	fclose (filp);	
	if (bytes_read != filesize){		
		std::cerr << "Expected " << filesize << " , read " << bytes_read << std::endl; 
		return -1;
	}

	int ret = memcmp(orig_buffer, recv_buffer, filesize);
	free(orig_buffer);
	if (ret!=0){
		std::cout << "Mismatch for " << full_path << std::endl;
		return -1;
	}

	return 0;
}
	

bool set_recv_window(int sockfd, int len_bytes) {
	socklen_t i;	
	i =  sizeof(int); 
	if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &len_bytes, i) < 0) {
		std::cerr << "Error setting recvbuf size" << strerror(errno) << std::endl;
		return false;
	}
	return true;
}


bool set_send_window(int sockfd, int len_bytes) {
	socklen_t i;	
	i =  sizeof(int); 
	if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &len_bytes, i) < 0) {
		std::cerr << "Error setting sendbuf size" << strerror(errno) << std::endl;
		return false;
	}
	return true;
}


bool set_tcp_nodelay(int sockfd) {
	int yes = 1;

	if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char*) &yes, sizeof(int)) < 0) {
		std::cerr << "Error setting tcp nodel" << strerror(errno) << std::endl;
		return false;
	}
	return true;
}


bool set_mss(int sockfd, int len_bytes) {
	socklen_t i;	
	i =  sizeof(int); 
	if (setsockopt(sockfd, IPPROTO_TCP, TCP_MAXSEG, &len_bytes, i) < 0) {
		std::cerr << "Error setting MSS" << strerror(errno) << std::endl;
		return false;
	}
	return true;
}



void print_socket_options (int sockfd, int id=-1) {		
	socklen_t i; 
	size_t len;
	int len1, len2, len3, len4;  

	i = sizeof(len);			

	if (getsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &len, &i) < 0) {		 
		std::cerr << "Error getting recvbuf size" << strerror(errno) << std::endl;	
		return;  
	}
	len1 = len;

	if (getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &len, &i) < 0) {	 
		std::cerr << "Error getting sendbuf size" << strerror(errno) << std::endl; 
		return; 
	}
	len2 = len;

	if (getsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &len, &i) < 0) {  
		std::cerr << "Error getting TCP nodelay size" << strerror(errno) << std::endl;	
		return;  
	}
	len3 =len;

	if (getsockopt(sockfd, IPPROTO_TCP, TCP_MAXSEG, &len, &i) < 0) {		
		std::cerr << "Error getting TCP MSS size" << strerror(errno) << std::endl;	
		return;  
	}
	len4 = len;

	if (id != -1)
		sockfd = id;

	safecout{} 	<< "[" << sockfd << "] Receive Buf size : " << len1 << "\n" 
							<< "[" << sockfd << "] Send Buf size for : " << len2 << "\n"	
							<< "[" << sockfd << "] TCP nodelay for : " << len3 << "\n"		 
							<< "[" << sockfd << "] TCP MSS for	: " << len4 << "\n"		 
							<< "--------------------------------- "  << std::endl;		
	
}


void print_socket_options_unsafe(int sockfd) {	 
	socklen_t i; 
	size_t len;  

	i = sizeof(len);			

	if (getsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &len, &i) < 0) {		 
		std::cerr << "Error getting recvbuf size" << strerror(errno) << std::endl;	
		return;  
	}
	std::cout << "Receive Buf size for " << sockfd << " : " << len << "\n" ;

	if (getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &len, &i) < 0) {	 
		std::cerr << "Error getting sendbuf size" << strerror(errno) << std::endl; 
		return; 
	}
	std::cout << "Send Buf size for " << sockfd << " : " << len << "\n"; 

	if (getsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &len, &i) < 0) {  
		std::cerr << "Error getting TCP nodelay size" << strerror(errno) << std::endl;	
		return;  
	}
	std::cout << "TCP nodelay for " << sockfd << " : " << len << "\n"; 

	if (getsockopt(sockfd, IPPROTO_TCP, TCP_MAXSEG, &len, &i) < 0) {		
		std::cerr << "Error getting TCP MSS size" << strerror(errno) << std::endl;	
		return;  
	}
	std::cout << "TCP MSS for " << sockfd << " : " << len << "\n" ; 
}


#endif
