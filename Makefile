

all:
	$(CXX) -O3 -march=skylake -std=c++17 -I./include -Wno-ignored-attributes ./src/main.cpp -o test
