#include <immintrin.h>
#include <stdio.h>
#include <iostream>

using namespace std;

#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)

#define _mm256_setr_m128i(lo, hi)   _mm256_set_m128i((hi), (lo))

/// Helper functions

template<typename T> constexpr bool is_size(size_t size) { return sizeof(T) == size; }

template<typename T> constexpr size_t get_num_lanes() {
    if constexpr(is_same<T,__m128i>::value)
        return 1;
    else
    if constexpr(is_same<T,__m256i>::value)
        return 2;
    /*else
    if constexpr(is_same<T,__m512i>::value)
        return 4;*/
    //else
    //    static_assert(false, "Data type not supported");

}

/// Intrinsics wrapper

// Add

template<typename T> T _m_add_i16(T a, T b) {
    if constexpr(is_same<T,__m128i>::value)
        return _mm_add_epi16(a,b);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_add_epi16(a,b);
}

template<typename T> T _m_add_i32(T a, T b) {
    if constexpr(is_same<T,__m128i>::value)
        return _mm_add_epi32(a,b);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_add_epi32(a,b);
}

template<typename T> T _m_add_f32(T a, T b) {
    if constexpr(is_same<T,__m128>::value)
        return _mm_add_ps(a,b);
    else
    if constexpr(is_same<T,__m256>::value)
        return _mm256_add_ps(a,b);
}

template<typename op_type, typename T> T _m_add(T a, T b) {
    if constexpr(is_same<op_type, short>::value)
        return _m_add_i16<T>(a, b);
    else if constexpr(is_same<op_type, int>::value)
        return _m_add_i32<T>(a, b);
    else if constexpr(is_same<op_type, float>::value)
        return _m_add_f32<T>(a, b);
}

// Mul

template<typename T> T _m_mul_i16(T a, T b) {
    if constexpr(is_same<T,__m128i>::value)
        return _mm_mullo_epi16(a,b);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_mullo_epi16(a,b);
}

template<typename T> T _m_mul_i32(T a, T b) {
    if constexpr(is_same<T,__m128i>::value)
        return _mm_mullo_epi32(a,b);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_mullo_epi32(a,b);
}

template<typename T> T _m_mul_f32(T a, T b) {
    if constexpr(is_same<T,__m128>::value)
        return _mm_mul_ps(a,b);
    else
    if constexpr(is_same<T,__m256>::value)
        return _mm256_mul_ps(a,b);
}

template<typename op_type, typename T> T _m_mul(T a, T b) {
    if constexpr(is_same<op_type, short>::value)
        return _m_mul_i16<T>(a, b);
    else if constexpr(is_same<op_type, int>::value)
        return _m_mul_i32<T>(a, b);
    else if constexpr(is_same<op_type, float>::value)
        return _m_mul_f32<T>(a, b);
}

/// Simd type wrapper
// supports int, float, short data types

template<typename simd_type, typename data_type> class simd_wrapper {
private:
    typedef simd_wrapper<simd_type, data_type> self_type;
    simd_type data;
public:

    simd_wrapper(simd_type value) {
        data = value;
    }

    operator simd_type() const {
        return data;
    }

    self_type operator+(const self_type &other) {
        return _m_add<data_type, simd_type>(this->data, other.data);
    }

    self_type operator*(const self_type &other) {
        return _m_mul<data_type, simd_type>(this->data, other.data);
    }
};

/// Simd vector implementation

// #define DOUBLE_PUMP_MEM

template<typename T, typename V, typename OP_TYPE> class simd_iterator
{
private:
    T *buffer;
    const size_t _lanes = get_num_lanes<V>();
public:

    simd_iterator(T *buffer): buffer(buffer) {}
    ~simd_iterator() {}

    inline simd_iterator<T,V,OP_TYPE> operator++() {
        simd_iterator<T,V,OP_TYPE> r = *this;
        buffer += _lanes;
        return r;
    }

    inline simd_iterator<T,V,OP_TYPE> operator++(int j) {
        buffer += _lanes;
        return *this;
    }

    inline simd_wrapper<V, OP_TYPE> operator*() {
        V res;
        if constexpr( is_same<V,__m128i>::value ) {
            if constexpr( sizeof(T)==16 )
                res = _mm_loadu_si128((__m128i*)buffer);
            else if constexpr( sizeof(T)==8 )
                res = _mm_loadu_si64(buffer);
        }
        else if constexpr( is_same<V,__m256i>::value )
        {
            if constexpr( sizeof(T) == 16 )
            #ifdef DOUBLE_PUMP_MEM
                res = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(buffer+1)), _mm_loadu_si128((__m128i*)(buffer)));
            #else
                res = _mm256_loadu_si256((__m256i*)(buffer));
            #endif
            else if constexpr( sizeof(T) == 8 )
                res = _mm256_set_m128i(_mm_loadu_si64(buffer+1), _mm_loadu_si64(buffer));
        }

        return res;
    }

    inline bool operator!=(const simd_iterator<T,V,OP_TYPE> &other) {
        return this->buffer != other.buffer;
    }

    inline void store(V data) {
        if constexpr( is_same<V,__m128i>::value ) {
            if constexpr( is_size<T>(16) ) _mm_storeu_si128(buffer, data);
            if constexpr( is_size<T>(8) )  _mm_storel_epi64(buffer, data);
        }
        if constexpr( is_same<V,__m256i>::value ) {
            if constexpr( sizeof(T)==16 ) {
            #ifdef DOUBLE_PUMP_MEM
                _mm_storeu_si128(buffer,   _mm256_extracti128_si256(data, 0));
                _mm_storeu_si128(buffer+1, _mm256_extracti128_si256(data, 1));
            #else
                _mm256_storeu_si256((__m256i*)buffer, data);
            #endif
            }
            if constexpr( sizeof(T)==8 ) { 
                _mm_storel_epi64(buffer,   _mm256_extracti128_si256(data, 0)); 
                _mm_storel_epi64(buffer+1, _mm256_extracti128_si256(data, 1)); 
            }
        }
    }
};

template<typename T, typename V, typename OP_TYPE> class simd_vector_t
{
private:
    T *buf;         // Source buffer
    size_t _size;    // Number of elements
public:
    typedef simd_iterator<T, V, OP_TYPE> iterator;

    simd_vector_t(T *buffer, size_t size): buf(buffer), _size(size) {}

    iterator begin() {
        return iterator(buf);
    }

    iterator end() {
        return iterator(buf+_size);
    }

    size_t size() {
        return _size;
    }
};

typedef simd_vector_t<__m128i, __m256i, short> simd_vector;

#define SIZE 1000
#define TEST 10000000

int main(int argc, char const *argv[])
{
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
    }

    short r = 0;
    for(int i=0; i<SIZE*16/2; ++i)
    {
        r += a[i];
    }
    std::cout << r << std::endl;

    /*simd_wrapper<__m128i, int> sv1(a[0]);
    simd_wrapper<__m128i, int> sv2(a[0]);
    sv1 = sv1 + sv2;
    __m128i v = sv1;*/

    return 0;
}
