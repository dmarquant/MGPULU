HEADERS=tasks.hpp util.h Range.hpp MatrixView.hpp makefile
CXXFLAGS=-std=c++11 -O3 -g -Wall
INCLUDES=-I${CUDAHOME}/include -I${HOME}/openblas/include
LIBS=-L${CUDAHOME}/lib64 -L${HOME}/openblas/lib -lcudart -lcublas -lcurand -lpthread -lopenblas

all: single_lu lu task_lu task_lu_overlap

single_lu: single_lu.cpp $(HEADERS)
	g++ $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS)

lu: lu.cpp $(HEADERS)
	g++ $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS)

task_lu: task_lu.cpp $(HEADERS)
	g++ $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS) -DPROFILE_TASKS #-DDEBUG_TASKS

#task_lu_overlap: task_lu.cpp $(HEADERS)
	#g++ $< -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS) -DOVERLAP -DPROFILE_TASKS #-DDEBUG_TASKS
