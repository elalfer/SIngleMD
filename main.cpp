#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <simd_vector.h>

using namespace std;

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
        // TODO: Same should be possible to write without specifying op size
        //      auto res = _m_sub( _m_max(R, _m_max(G, B)), _m_min(R, _m_min(G, B)));

        _m_storeu_si(vC[i], res);
    }
}

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

#define SIZE 10000
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

    for(int i=0; i<TEST; i++)
        RGB_To_C_vec(nC, nR, nG, nB, SIZE);

    /*
    short *a = (short*)malloc(sizeof(__m128i)*SIZE);
    short *b = (short*)malloc(sizeof(__m128i)*SIZE);
    short *c = (short*)malloc(sizeof(__m128i)*SIZE);

    simd_vector Va((__m128i*)a, SIZE);
    simd_vector Vb((__m128i*)b, SIZE);
    simd_vector Vc((__m128i*)c, SIZE);

    for(int i=0; i<TEST; i++) {
        auto ita=Va.begin();
        auto itb=Vb.begin();
        auto itc=Vc.begin();
        for(; ita!=Va.end(); ita++, itb++, itc++) {
            // ita.store(_m_add_i32(*itb, *itc));
            ita.store(*ita * *itb + *itc);
        }
    }*/

    short r = 0;
    for(int i=0; i<SIZE; ++i)
    {
        r += nC[i];
    }
    std::cout << r << std::endl;

    /*simd_wrapper<__m128i, int> sv1(a[0]);
    simd_wrapper<__m128i, int> sv2(a[0]);
    sv1 = sv1 + sv2;
    __m128i v = sv1;*/

    return 0;
}
