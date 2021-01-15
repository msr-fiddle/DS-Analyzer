# A distributed minT cache server and client

Creates a server that handles cache requests from the client.

Client send a request of max size 100 bytes encoded as "GET <filename>"

Server returns data if the file is present in its local minT cache hosted at /dev/shm/cache/
	For the client request, server responds with a header of size 8 bytes which is either
	the length of the file if it exists in cache, else
	If the file is not in cache, it returns "NOTFOUND". 
	In the payload (if file found), the server sends the data of the file

## Running a test:

* Copy a binary file 'sample1.txt' to /dev/shm/cache on the server
	* ./build and copy the server on one node and client on other

* Server:
	* ./server <num_threads> <start_port> 
	* This opens up <num_threads parallel TCP flows that client can cannect to starting at port start_port and ending at start_port + num_threads


* Client:
	* ./client <num_threads> <server_ip> <dir_path whose files to read> <start_port>
	* The client shuffled the order of files read from dir_path and requests them in parallel (a partition handled in each thread) on num_threads consecutive ports starting at start_port
	* The start_port must match in server and client instances
	* All the files present in <dir_path> are assumed to be present in '/dev/shm/cache/' at the server.
	* Eg. If dir_path is /mnt/dataset and has tthe heirarchy:
	          /mnt/dataset/1/
			  				|--1.txt
							|--2.txt
							.
							|--100.txt
			  /mnt/dataset/2/
			  				|--1.txt
							|--2.txt
							.
							|--100.txt

		then the client creates a list of relative paths as follows :
		1/1.txt, 1/2.txt ... 1/100.txt, 2/1.txt, 2/2.txt .. and so on
		Therefore, at the server, the following files must exist :
		/dev/shm/cache/1/1.txt, /dev/shm/cache/1/2.txt ...


## Example:
> * Server (10.185.x.0)
> * cd dist-mint
> * ./build.sh
> * mkdir -p /dev/shm/cache
> * scp -r user@10.185.x.1:/home/data/\* /dev/shm/cache/
> * ./server 16 5555 


> * Client (10.185.x.1)
> * cd dist-mint
> * ./build.sh
> * ./client 16 10.185.x.0 /home/data/ 5555
> * To validate the received files against the original files in cache, pass an additional argument check:
> * ./client 16 10.185.x.0 /home/data/ 5555 check
