#!/bin/bash

g++ server.cpp  -std=c++11  -lpthread -o server
g++ client.cpp  -std=c++11  -lpthread -o client
