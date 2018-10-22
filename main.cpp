#include <immintrin.h>
#include <stdio.h>
#include <iostream>

using namespace std;

template<typename T> constexpr bool is_size(size_t size) { return sizeof(T) == size; }

template<typename T, typename V> class simd_iterator
{
private:
    T *buffer;
public:

    simd_iterator(T *buffer): buffer(buffer) {}
    ~simd_iterator() {}

    simd_iterator<T,V> operator++() {
        simd_iterator<T,V> r = *this;
        if constexpr( sizeof(V)==16 )
            buffer++;
        else if constexpr( sizeof(V)==32 )
            buffer+=2;
        return r;
    }
    simd_iterator<T,V> operator++(int j) {
        if constexpr( sizeof(V)==16 )
            buffer++;
        else if constexpr( sizeof(V)==32 )
            buffer+=2;
        return *this;
    }

    V operator*() {
        V res;
        if constexpr( sizeof(V)==16 ) {
            if constexpr( sizeof(T)==16 )
                res = _mm_loadu_si128((__m128i*)buffer);
            else if constexpr( sizeof(T)==8 )
                res = _mm_loadu_si64(buffer);
        }
        else if constexpr( sizeof(V)==32 )
        {
            if constexpr( sizeof(T) == 16 )
                res = _mm256_setr_m128i(_mm_loadu_si128((__m128i*)buffer), _mm_loadu_si128((__m128i*)(buffer+1)));
            else if constexpr( sizeof(T) == 8 )
                res = _mm256_setr_m128i(_mm_loadu_si64(buffer), _mm_loadu_si64(buffer+1));
        }

        return res;
    }

    bool operator!=(const simd_iterator<T,V> &other) {
        return this->buffer != other.buffer;
    }

    void store(V data) {
        if constexpr( is_size<V>(16) ) {
            if constexpr( is_size<T>(16) ) _mm_storeu_si128(buffer, data);
            if constexpr( is_size<T>(8) )  _mm_storel_epi64(buffer, data);
        }
        if constexpr( sizeof(V)==32 ) {
            if constexpr( sizeof(T)==16 ) { 
                _mm_storeu_si128(buffer,   _mm256_extracti128_si256(data, 0)); 
                _mm_storeu_si128(buffer+1, _mm256_extracti128_si256(data, 1)); 
            }
            if constexpr( sizeof(T)==8 ) { 
                _mm_storel_epi64(buffer,   _mm256_extracti128_si256(data, 0)); 
                _mm_storel_epi64(buffer+1, _mm256_extracti128_si256(data, 1)); 
            }
        }
    }
};

template<typename T, typename V> class simd_vector_t
{
private:
    T *buf;         // 
    size_t size;    // Number of elements
public:
    typedef simd_iterator<T, V> iterator;

    simd_vector_t(T *buffer, size_t size): buf(buffer), size(size) {}

    iterator begin() {
        return iterator(buf);
    }
    iterator end() {
        return iterator(buf+size);
    }
};

template<typename T> T _m_add32(T a, T b) {
    if constexpr(is_same<T,__m128i>::value)
        return _mm_add_epi32(a,b);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_add_epi32(a,b);
}

typedef simd_vector_t<__m128i, __m256i> simd_vector;

#define SIZE 1000
#define TEST 100000

int main(int argc, char const *argv[])
{
    __m128i *a = (__m128i*)malloc(sizeof(__m128i)*SIZE);
    __m128i *b = (__m128i*)malloc(sizeof(__m128i)*SIZE);;
    __m128i *c = (__m128i*)malloc(sizeof(__m128i)*SIZE);;

    simd_vector Va(a, SIZE);
    simd_vector Vb(b, SIZE);
    simd_vector Vc(c, SIZE);

    for(int i=0; i<TEST; i++) {
        auto ita=Va.begin();
        auto itb=Vb.begin();
        auto itc=Vc.begin();
        for(; ita!=Va.end(); ita++, itb++, itc++) {
            ita.store(_m_add32(*itb, *itc));
        }
    }


    return 0;
}
