#include "intrin_wrapper.h"

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

    simd_type get_value() {
    	return data;
    }

    self_type operator+(const self_type &other) {
        return _m_add<data_type, simd_type>(this->data, other.data);
    }

    self_type operator*(const self_type &other) {
        return _m_mullo<data_type, simd_type>(this->data, other.data);
    }
};

/// Simd vector implementation

// #define DOUBLE_PUMP_MEM

template<typename T, typename V, typename OP_TYPE> class simd_iterator
{
private:
    T *buffer;
    const size_t _lanes = get_num_lanes<V>();
    typedef simd_iterator<T,V,OP_TYPE> self_type;
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

    inline self_type operator+(int cnt) {
    	// buffer += _lanes*cnt;
    	return self_type(buffer + _lanes*cnt);
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

    static size_t elements_per_vector() {
    	return ( sizeof(V) / sizeof(OP_TYPE) );
    }

    iterator /*simd_wrapper<V, OP_TYPE>*/ operator[](size_t idx) {
    	return (begin()+idx);
    }
};
