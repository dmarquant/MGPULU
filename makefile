INCLUDES=-I${CUDAHOME}/include
LIBS=-L${CUDAHOME}/lib64

all:
	g++ lu.cpp -o lu -lcudart -lcublas -lcurand -lmkl_rt -std=c++11 -g -Wall $(INCLUDES) $(LIBS) 
	g++ task_lu.cpp -o task_lu -lcudart -lcublas -lcurand -lmkl_rt -std=c++11 -g -Wall $(INCLUDES) $(LIBS) -lpthread
	g++ lu.cpp -o lu_cpu -lcudart -lcublas -lcurand -lmkl_rt -std=c++11 -g -Wall $(INCLUDES) $(LIBS) -DTEST_CPU
