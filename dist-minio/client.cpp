#include<iostream>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<netdb.h>
#include<sys/uio.h>
#include<unistd.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/syscall.h>
#include<string.h>
#include<stdio.h>
#include <chrono>
#include <fstream>
#include "commands.h"
#include <vector>
#include <dirent.h>
#include <algorithm>
#include <random>
#include <cmath>
#include <thread>
//#include<arpanet.h>

using namespace std;

#define PORT 5555

void print_time(chrono::nanoseconds nanoseconds){
	cout << "Total read time = " << nanoseconds.count() << " ns" << endl;
	auto microseconds = std::chrono::duration_cast< std::chrono::microseconds >(nanoseconds);
	cout << "Total read time = " << microseconds.count() << " us" << endl;
	auto milliseconds = std::chrono::duration_cast< std::chrono::milliseconds >(nanoseconds);
	cout << "Total read time = " << milliseconds.count() << " ms" << endl;
}

void print_time_us(chrono::nanoseconds nanoseconds){
	auto microseconds = std::chrono::duration_cast< std::chrono::microseconds >(nanoseconds);
	cout	<< microseconds.count() << endl;
}

int read_header_from_other_node(int server_fd, string fname) {
	cout << "Reading file " << fname << endl;
	char request[REQUEST_SIZE] = "GET ";
	strncpy(request + GET_SIZE, fname.c_str(), fname.length());
	int ret = send(server_fd, request, REQUEST_SIZE, 0);
	if (ret < 0){
		cerr << "Error sending request" << endl;
		return -1;
	}
	unsigned char header[HEADER_SIZE];
	int r = recv(server_fd, header, HEADER_SIZE, 0);
	if (strcmp(reinterpret_cast<const char*>(header), "NOFOUND") == 0) {
		cout << "File " << fname	<< " requested was notfound on server cache" << endl;
		return -1;
	}
	unsigned long msg_size = *(reinterpret_cast<unsigned long *>(header));
	cout << "Header is " << msg_size << " and size " << sizeof(header) << endl;
	return msg_size;
}


int read_file_from_other_node(int server_fd, string fname, uint8_t * buf, unsigned long msg_size) {
	int msg_read = 0;
	int rec = 0;
	do {
		rec = recv(server_fd, buf + msg_read, msg_size, 0);
		msg_read += rec;
		if(rec < 0){
			cout<<"Error receiving\n";
			return -1;
		}
	}while (rec!=0 && (msg_read < msg_size));
	return msg_read;
}


int read_from_other_node(int server_fd, string fname, uint8_t * buf, unsigned long msg_size) {
	char request[REQUEST_SIZE] = "GET ";
	strncpy(request + GET_SIZE, fname.c_str(), fname.length());
	int ret = send(server_fd, request, REQUEST_SIZE, 0);
	if (ret < 0){
		cerr << "Error sending request" << endl;
		return -1;
	}

	int msg_read = 0;
	int rec = 0;
	do {
		rec = recv(server_fd, buf + msg_read, msg_size, 0);
		msg_read += rec;
		if(rec < 0){
			cout<<"Error receiving\n";
			return -1;
		}
	}while (rec!=0 && (msg_read < msg_size));
	return msg_read;
}


size_t getFilesize(const char* filename) {
	struct stat st;
	if(stat(filename, &st) != 0) {
		return 0;
	}
	return st.st_size;
}



string full_path(string fname, string root){
	return root + '/' + fname;
}



int initialize_socket(int port, char* ip_addr){
	int cnct, server_fd;
	struct sockaddr_in server;
	struct hostent *hp;
	
	server.sin_family=AF_INET;
	server.sin_port=htons(port);
	server.sin_addr.s_addr=INADDR_ANY;
	server_fd =socket(AF_INET,SOCK_STREAM,0);
	if(	server_fd	<	0){
		cout<<"Error creating socket\n";
		return -1;
	}

	if (!set_tcp_nodelay(server_fd))
		return -1;		
	set_recv_window(server_fd, 1500000);  
	set_send_window(server_fd, 1500000); 
	if (!set_mss(server_fd, 1460))
		return -1;

	cout<<"Socket created" << endl;
	hp=gethostbyname(ip_addr);
	bcopy((char *)hp->h_addr,(char *)&server.sin_addr.s_addr,hp->h_length);
	cnct=connect(server_fd,(struct sockaddr*)&server,sizeof(server));
	if(cnct<0){
		cout<<"Error connecting " << endl;
		return -1;
	}
	cout<<"Connection has been made\n";
	print_socket_options(server_fd);	
	cout << "-----------------------" << endl;	
	return server_fd;
}



// The file strings are relative paths in cache
int get_many_samples(vector<string> file_list,int start, int end, int server_fd, int id, string root, bool must_check){
	cout << "[" << id << "] : Check recv buffer integrity set to " << must_check << endl;
	int read = 0;
	long long total_bytes_read = 0;
	float total_elapsed_time = 0.0;
	ns batch_time = ns::zero();
	for (int i = start; i < end; i++){
		string fname = file_list[i];

		auto begin = Time::now();
		unsigned long image_size = getFilesize(full_path(fname, root).c_str());
		uint8_t * buffer;
		buffer = (uint8_t *) malloc(image_size);
		if (buffer == NULL){
			cerr << "Cannot allocate buffer " << strerror(errno) << endl;
			return -1;
		}
		//total_bytes_read += image_size;
		//cout << "Reading " << fname << endl;
		int bytes_read = read_from_other_node(server_fd, fname, buffer, image_size);
		if (bytes_read < 0){
			cerr << "Unsuccessful read for " << fname << endl;
			free(buffer);
			continue;
		}
		else{
			total_bytes_read += bytes_read;
		}
		if (must_check) {
			int equal = compare_buffers(buffer, full_path(fname, root), image_size);
			if(equal != 0){
				cout << "File " << fname << " mismatch" << endl;
				return -1;
			}
		}
		free(buffer);
		auto finish = Time::now();
		fsec diff = finish - begin;
		batch_time += std::chrono::duration_cast<ns>(diff);
		read ++;

		if (read % 50000 == 0){
			total_elapsed_time += ((batch_time).count()/ pow(10,9));
			//cout << total_bytes_read << " : " <<	total_elapsed_time << " : " << batch_time.count() << endl;
			batch_time = ns::zero();
			cout << "[" << id << " : " << read << "/" << (end - start) << "  SPEED] : " << total_bytes_read/total_elapsed_time/1024/1024 << " MBps" << endl;
		}
		
	}
	cout << "[" << id <<	"] Bytes read = " << total_bytes_read << "		Time(s) = " << total_elapsed_time << "		SPEED = " << total_bytes_read/total_elapsed_time/1024/1024 << " MBps" << endl;	
	return 0;

}

int get_sample_debug(int server_fd, string fname) {
	char request[REQUEST_SIZE] = "GET ";
	int BUFSIZE = 4096;
	//string fname = "sample2.txt";
	//char get[] = "GET ";
	//strncpy(request, get, strlen(get));
	strncpy(request + GET_SIZE, fname.c_str(), fname.length());
	cout << "Request is " << request << " of size " << REQUEST_SIZE << endl;
	int ret = send(server_fd, request, REQUEST_SIZE, 0);
	//int ret = write(server_fd, request, REQUEST_SIZE);
	if (ret < 0){
		cerr << "Error sending request" << endl;
		return -1;
	}
	unsigned char header[HEADER_SIZE];	
	int r = recv(server_fd, header, HEADER_SIZE, 0); 
	cout << reinterpret_cast<const char*>(header) << endl;		 
	//string not_found = "NOTFOUND"; 
	if (strcmp(reinterpret_cast<const char*>(header), "NOFOUND") == 0) {
		cout << "File " << fname	<< " requested was not found on server cache" << endl;
		return -1;
	}
	unsigned long msg_size = *(reinterpret_cast<unsigned long *>(header));	 
	cout << "Header is " << msg_size << " and size " << sizeof(header) << endl;
	int msg_read = 0;		 
	int rec = 0;		
	char buf[msg_size];
	cout << "Trying to recv" << endl;
	do{
		rec = recv(server_fd, buf + msg_read, msg_size, 0);
		msg_read += rec;
		//cout << "\trecv " << msg_read << endl;
		if(rec < 0){
			cout<<"Error receiving\n";
			return 0;
		}
	} while (rec!=0 && (msg_read < msg_size));

	const char* recvfile = "clientfile.txt";
	const char* origfile = "/dev/shm/cache/sample2.txt";
	cout << "Received file of size " << msg_read << endl; 
	ofstream outfile ("clientfile.txt", ofstream::binary);
	outfile.write (buf, msg_read);
	outfile.close();

	cout << "Comparing sample2.txt and clientfile.txt" << endl;
	int cmp = compare_files(origfile, recvfile); 
	if (cmp < 0) {
		cout << "Mismatched" << endl;
		return -1;
	}
	else
		cout << "Files matched" << endl;

	//close(server_fd);
	//shutdown(server_fd, 0);
	return rec;
	
}



void assemble_file_list(const std::string& path, const std::string& curr_entry, std::vector<string> *file_list) {
	std::string curr_dir_path = path + "/" + curr_entry;
	DIR *dir = opendir(curr_dir_path.c_str());
	struct dirent *entry;
	while ((entry = readdir(dir))) {
		std::string full_path = curr_dir_path + "/" + std::string{entry->d_name};
#ifdef _DIRENT_HAVE_D_TYPE
		if (entry->d_type != DT_REG && entry->d_type != DT_LNK &&
			entry->d_type != DT_UNKNOWN) {
			continue;
		}
#endif
		std::string rel_path = curr_entry + "/" + std::string{entry->d_name};
		file_list->push_back(rel_path);
		
	}
	closedir(dir);
}


vector<string> traverse_directories(const string & file_root) {
	DIR *dir = opendir(file_root.c_str());
	struct dirent *entry;
	vector<string> file_list;
	vector<string> entry_name_list;
	while ((entry = readdir(dir))) {
		struct stat s;
		std::string entry_name(entry->d_name);
		std::string full_path = file_root + "/" + entry_name;
		int ret = stat(full_path.c_str(), &s);
		if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
		if (S_ISDIR(s.st_mode)) {
			entry_name_list.push_back(entry_name);
		}
	}

	std::sort(entry_name_list.begin(), entry_name_list.end());
	for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
		assemble_file_list(file_root, entry_name_list[dir_count],  &file_list);
	}
	sort(file_list.begin(), file_list.end());
	shuffle (file_list.begin(), file_list.end(), std::default_random_engine(8));
	printf("read %lu files from %lu directories\n", file_list.size(), entry_name_list.size());
	closedir(dir);
	return file_list;
}




int main(int argc, char * argv[]){
	if (argc < 5){
		cout << "Entered " << argc << " args. Pass in num clients ,ip, and file root and start port" << endl;
		return -1;
	}

	bool must_check = false;
	if (argc > 5) {
		string action(argv[5]);
		if (action == "check")
			must_check = true;
	}

	int num_clients = atoi(argv[1]);
	int port_start = atoi(argv[4]);

	//create array of files to read
	string root = argv[3];
	vector<string> file_list = traverse_directories(argv[3]);
	//for (int i=0; i < file_list.size(); i++)
	//	cout << i << " : " << file_list[i] << endl;

	vector<int> server_fd(num_clients, 0);
	for (int i =0; i < num_clients; i++){  
		int port = port_start + i;	
		server_fd[i] = initialize_socket(port, argv[2]); 
		if (server_fd[i] < 0){
			cerr << "Error connecting to server " << strerror(errno) << endl;
			return -1;
		}
	}


	vector<thread> thread_list(num_clients);
	int num_samples = file_list.size();
	int samples_per_thread = ceil(num_samples/num_clients);
	cout << "Num samples = " << num_samples << ", samples/thread = " << samples_per_thread << endl;

	auto start_t = Time::now();
	for (int i = 0; i < num_clients; i ++){
		int start = i* samples_per_thread;
		int end = (i+1)* samples_per_thread;
		if ( i == num_clients -1)
			end = num_samples;

		cout << "Starting thread "<< i << " [FROM] " << start << "	[TO] " << end << endl;
		thread_list[i] = thread(get_many_samples, file_list, start, end, server_fd[i], i, root, must_check);
		cout << "Created thread "  << i << endl;
	}

	for (int i = 0; i < num_clients; i ++){
		cout << "Wait " << i << endl;
		thread_list[i].join();
		close(server_fd[i]);
	}

	auto stop_t = Time::now();
	fsec fs = stop_t - start_t;
	std::cout << "Duration =	" <<	fs.count() << " s" << endl;
	string command = "du -sh ";
	command.append(root);
	system(command.c_str());
	
	/* Debug - one sample
	for (int i =0; i < num_clients; i++){  
		//int port = PORT + num_clients;	
		int port = PORT + i;	
		int fd = initialize_socket(port, argv[2]); 
		if (fd < 0){
			cerr << "Error connecting to server " << strerror(errno) << endl;
			return -1;
		}
		int recd = read_header_from_other_node(fd, "sample3.txt");
		//int recd = get_sample(fd, fname);
		if (recd < 0){
			cout << "Recv failed from " << fd << endl;
			//return -1;
		}
		}
	cout << "--------------------------------------" << endl;
	*/
	return 0;
}
