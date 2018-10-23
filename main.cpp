#include <immintrin.h>
#include <stdio.h>
#include <iostream>

using namespace std;

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
        /*if constexpr( is_same<int, data_type>::value )
            return _m_add_i32(data, other.data);
        else
        if constexpr( is_same<short, data_type>::value )
            return _m_add_i16(data, other.data);
        else
        if constexpr( is_same<float, data_type>::value )
            return _m_add_ps(data, other.data);*/
        return _m_add<data_type, simd_type>(this->data, other.data);
    }

    self_type operator*(const self_type &other) {
        return _m_mul<data_type, simd_type>(this->data, other.data);
    }
};

/// Simd vector implementation

template<typename T, typename V, typename OP_TYPE> class simd_iterator
{
private:
    T *buffer;
    const size_t _lanes = get_num_lanes<V>();
public:

    simd_iterator(T *buffer): buffer(buffer) {}
    ~simd_iterator() {}

    simd_iterator<T,V,OP_TYPE> operator++() {
        simd_iterator<T,V,OP_TYPE> r = *this;
        buffer += _lanes;
        return r;
    }

    simd_iterator<T,V,OP_TYPE> operator++(int j) {
        buffer += _lanes;
        return *this;
    }

    V operator*() {
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
                res = _mm256_setr_m128i(_mm_loadu_si128((__m128i*)buffer), _mm_loadu_si128((__m128i*)(buffer+1)));
            else if constexpr( sizeof(T) == 8 )
                res = _mm256_setr_m128i(_mm_loadu_si64(buffer), _mm_loadu_si64(buffer+1));
        }

        return res;
    }

    bool operator!=(const simd_iterator<T,V,OP_TYPE> &other) {
        return this->buffer != other.buffer;
    }

    void store(V data) {
        if constexpr( is_same<V,__m128i>::value ) {
            if constexpr( is_size<T>(16) ) _mm_storeu_si128(buffer, data);
            if constexpr( is_size<T>(8) )  _mm_storel_epi64(buffer, data);
        }
        if constexpr( is_same<V,__m256i>::value ) {
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

typedef simd_vector_t<__m128i, __m256i, int> simd_vector;

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
            ita.store(*itb + *itc);
        }
    }

    simd_wrapper<__m128i, int> sv1(a[0]);
    simd_wrapper<__m128i, int> sv2(a[0]);
    sv1 = sv1 + sv2;
    __m128i v = sv1;


    return 0;
}
