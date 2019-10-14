#include <stdio.h>
#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <simd_vector.h>
#include <x86intrin.h>

using namespace std;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

__attribute__((noinline))
void RGB_To_C_SSE(uint16_t *nC, uint16_t *nR, uint16_t *nG, uint16_t *nB, size_t size) {
    __m128i *vR = (__m128i*)nR;
    __m128i *vG = (__m128i*)nG;
    __m128i *vB = (__m128i*)nB;
    __m128i *vC = (__m128i*)nC;

    for(int i=0; i < size/8; ++i) {
        __m128i R = _mm_loadu_si128(&vR[i]);
        __m128i G = _mm_loadu_si128(&vG[i]);
        __m128i B = _mm_loadu_si128(&vB[i]);

        __m128i res = _mm_sub_epi16( _mm_max_epi16(R, _mm_max_epi16(G, B)), _mm_min_epi16(R, _mm_min_epi16(G, B)));

        _mm_storeu_si128(&vC[i], res);
    }
}

typedef simd_vector_t<__m128i, __m256i, short> simd_vector;

__attribute__((noinline))
void RGB_To_C_vec(uint16_t *nC, uint16_t *nR, uint16_t *nG, uint16_t *nB, size_t size) {
    simd_vector vR((__m128i*)nR, size);
    simd_vector vG((__m128i*)nG, size);
    simd_vector vB((__m128i*)nB, size);
    simd_vector vC((__m128i*)nC, size);

    for(int i=0; i < size/simd_vector::elements_per_vector(); ++i)
    {
        auto R = _m_loadu_si(vR[i]);
        auto G = _m_loadu_si(vG[i]);
        auto B = _m_loadu_si(vB[i]);

        // Very similar to legacy intrinsics
        auto res = _m_sub_epi16( _m_max_epi16(R, _m_max_epi16(G, B)), _m_min_epi16(R, _m_min_epi16(G, B)));

        _m_storeu_si(vC[i], res);
    }
}

// TODO implement min/max operations
__attribute__((noinline))
void RGB_To_C_vec_operator(uint16_t *nC, uint16_t *nR, uint16_t *nG, uint16_t *nB, size_t size) {
    simd_vector vR((__m128i*)nR, size);
    simd_vector vG((__m128i*)nG, size);
    simd_vector vB((__m128i*)nB, size);
    simd_vector vC((__m128i*)nC, size);

    simd_vector::iterator iR=vR.begin();
    simd_vector::iterator iG=vG.begin();
    simd_vector::iterator iB=vB.begin();
    for(auto iC=vC.begin(); iC != vC.end(); iR++, iG++, iB++, iC++)
    {
        auto res = *iR * *iG + *iB;
        iC.store(res);
    }
}

#define SIZE 2048
#define TEST 10000000

int main(int argc, char const *argv[])
{
    uint16_t *nR = (uint16_t*)malloc(SIZE*sizeof(uint16_t));
    uint16_t *nG = (uint16_t*)malloc(SIZE*sizeof(uint16_t));
    uint16_t *nB = (uint16_t*)malloc(SIZE*sizeof(uint16_t));
    uint16_t *nC = (uint16_t*)malloc(SIZE*sizeof(uint16_t));


    for(int i=0; i<SIZE; ++i) {
        nR[i] = rand();
        nG[i] = rand();
        nB[i] = rand();
        nC[i] = rand();
    }

	auto t0 = Time::now();
	for (int i = 0; i < TEST; i++)
		RGB_To_C_SSE(nC, nR, nG, nB, SIZE);
    auto t1 = Time::now();

    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);
    std::cout << "SSE " << d.count() << "ms\n";

	t0 = Time::now();
    for(int i=0; i<TEST; i++)
        RGB_To_C_vec(nC, nR, nG, nB, SIZE);
    t1 = Time::now();

    fs = t1 - t0;
    d = std::chrono::duration_cast<ms>(fs);
    std::cout << "Vec " << d.count() << "ms\n";

    free(nR);
    free(nG);
    free(nB);
    free(nC);

	return 0;
}
