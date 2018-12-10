#include <immintrin.h>

using namespace std;

#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)

#define _mm256_setr_m128i(lo, hi)   _mm256_set_m128i((hi), (lo))

// Load

#define _m_loadu_si(V) ( (*(V)).get_value() )

/*template<typename T> T _m_loadu_si(T* p) {
    if constexpr(is_same<T,__m128i>::value)
        return _mm_loadu_si128(p);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_loadu_si256(p);
}*/

// Store

template<typename T, typename V> _m_storeu_si(T p, V a) {
    /*if constexpr(is_same<T,__m128i>::value)
        return _mm_storeu_si128(p, a);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_storeu_si256(p, a);*/
    p.store(a);
}

#define DEFINE_VECOR_OPERATION(OP) \
template<typename T> T _m_ ## OP ## _i16(T a, T b) { \
    if constexpr(is_same<T,__m128i>::value) \
        return _mm_ ## OP ## _epi16(a,b); \
    else \
    if constexpr(is_same<T,__m256i>::value) \
        return _mm256_ ## OP ## _epi16(a,b); \
} \
template<typename T> T _m_ ## OP ## _epi16(T a, T b) { return _m_ ## OP ## _i16(a, b); } \
\
template<typename T> T _m_ ## OP ## _i32(T a, T b) { \
    if constexpr(is_same<T,__m128i>::value) \
        return _mm_ ## OP ## _epi32(a,b); \
    else \
    if constexpr(is_same<T,__m256i>::value) \
        return _mm256_ ## OP ## _epi32(a,b); \
} \
template<typename T> T _m_ ## OP ## _epi32(T a, T b) { return _m_ ## OP ## _i32(a, b); } \
 \
template<typename T> T _m_ ## OP ## _f32(T a, T b) { \
    if constexpr(is_same<T,__m128>::value) \
        return _mm_ ## OP ## _ps(a,b); \
    else \
    if constexpr(is_same<T,__m256>::value) \
        return _mm256_ ## OP ## _ps(a,b); \
} \
template<typename T> T _m_ ## OP ## _ps(T a, T b) { return _m_ ## OP ## _f32(a, b); } \
 \
template<typename op_type, typename T> T _m_ ## OP (T a, T b) { \
    if constexpr(is_same<op_type, short>::value) \
        return _m_ ## OP ## _i16<T>(a, b); \
    else if constexpr(is_same<op_type, int>::value) \
        return _m_ ## OP ## _i32<T>(a, b); \
    else if constexpr(is_same<op_type, float>::value) \
        return _m_ ## OP ## _f32<T>(a, b); \
}

DEFINE_VECOR_OPERATION(add)
DEFINE_VECOR_OPERATION(sub)
DEFINE_VECOR_OPERATION(mullo)
DEFINE_VECOR_OPERATION(div)

DEFINE_VECOR_OPERATION(min)
DEFINE_VECOR_OPERATION(max)

// Add

/*template<typename T> T _m_add_i16(T a, T b) {
    if constexpr(is_same<T,__m128i>::value)
        return _mm_add_epi16(a,b);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_add_epi16(a,b);
}
#define _m_add_epi16(a, b) _m_add_i16((a),(b))

template<typename T> T _m_add_i32(T a, T b) {
    if constexpr(is_same<T,__m128i>::value)
        return _mm_add_epi32(a,b);
    else
    if constexpr(is_same<T,__m256i>::value)
        return _mm256_add_epi32(a,b);
}
#define _m_add_epi32(a, b) _m_add_i32((a),(b))

template<typename T> T _m_add_f32(T a, T b) {
    if constexpr(is_same<T,__m128>::value)
        return _mm_add_ps(a,b);
    else
    if constexpr(is_same<T,__m256>::value)
        return _mm256_add_ps(a,b);
}
#define _m_add_ps(a, b) _m_add_f32((a),(b))

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
}*/
