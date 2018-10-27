

all:
	$(CXX) -O3 -march=skylake -std=c++17 -Wignored-attributes main.cpp
