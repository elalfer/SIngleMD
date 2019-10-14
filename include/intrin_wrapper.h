#include <immintrin.h>

using namespace std;

#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)

#define _mm256_setr_m128i(lo, hi)   _mm256_set_m128i((hi), (lo))

// Load intrinsic
#define _m_loadu_si(V) ( (*(V)).get_value() )

// Store intrinsic
template<typename T, typename V> void _m_storeu_si(T p, V a) {
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
