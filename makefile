INCLUDES=-I${CUDAHOME}/include -I${HOME}/openblas/include
LIBS=-L${CUDAHOME}/lib64 -L${HOME}/openblas/lib -lcudart -lcublas -lcurand -lpthread -lmkl_rt



all:
	g++ lu.cpp      -o lu      -std=c++11 -g -Wall $(INCLUDES) $(LIBS) 
	g++ task_lu.cpp -o task_lu -std=c++11 -g -Wall $(INCLUDES) $(LIBS)
	g++ lu.cpp      -o lu_cpu  -std=c++11 -g -Wall $(INCLUDES) $(LIBS) -DTEST_CPU
