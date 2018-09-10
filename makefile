INCLUDES=-I${CUDAHOME}/include
LIBS=-L${CUDAHOME}/lib64

all:
	g++ lu.cpp -o lu -lcudart -lcublas -lcurand -lopenblas -std=c++11 -g -Wall $(INCLUDES) $(LIBS) 
	g++ lu.cpp -o lu_cpu -lcudart -lcublas -lcurand -lopenblas -std=c++11 -g -Wall $(INCLUDES) $(LIBS) -DTEST_CPU
