

all:
	$(CXX) -O3 -march=skylake -std=c++17 -Wno-ignored-attributes main.cpp
