

all:
	$(CXX) -O3 -march=skylake -std=c++17 -I./ -Wno-ignored-attributes main.cpp
