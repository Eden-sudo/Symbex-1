#include "littlebit_blob.h"
#include "tinylm_factor.h"
#include <string.h>

static uint32_t u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static int32_t i32(const uint8_t *p) {
    uint32_t v = u32(p);
    return v <= INT32_MAX ? (int32_t)v : -1 - (int32_t)(UINT32_MAX - v);
}
static int64_t rounded(int64_t x) {
    int64_t q = ((x < 0 ? -x : x) + (INT64_C(1) << 19)) >> 20;
    return x < 0 ? -q : q;
}
static int padding(const uint8_t *p, size_t rows, size_t cols) {
    size_t j, stride = (cols + 7) / 8;
    if (cols % 8)
        for (j = 0; j < rows; ++j)
            if (p[j * stride + stride - 1] >> (cols % 8)) return -1;
    return 0;
}
int littlebit_blob_matvec(const uint8_t *b, size_t size,
                         const int8_t *x, size_t nx, int32_t *y, size_t ny) {
    size_t n, r, o, count, vs, us, nv, nu, payload, k, j, t, i;
    uint64_t bytes;
    if (!b || !x || !y || size < 40 || memcmp(b, "LBT1", 4) ||
        u32(b+4) != 1 || u32(b+24) != 20 || u32(b+28) != 12) return -1;
    n=u32(b+8); r=u32(b+12); o=u32(b+16); count=u32(b+20);
    if (!n || !r || !o || r>n || r>o || count<1 || count>2 ||
        n>INT32_MAX/128 || r>INT32_MAX/127 || o>INT32_MAX || nx!=n || ny!=o) return -1;
    /* Positive finite IEEE754 binary64 input scale, inspected as bits only. */
    if ((b[39]&128) || ((u32(b+36)>>20)&2047)==2047 ||
        ((u32(b+36)&0x7fffffff)==0 && u32(b+32)==0)) return -1;
    vs=(n+7)/8; us=(r+7)/8;
    bytes=(uint64_t)r*vs+(uint64_t)o*us+4*((uint64_t)n+r+o);
    if (bytes>(SIZE_MAX-40)/count || size!=40+count*(size_t)bytes) return -1;
    payload=(size_t)bytes; nv=r*vs; nu=o*us;
    for (k=0; k<count; ++k) {
        const uint8_t *p=b+40+k*payload;
        if (padding(p,r,n) || padding(p+nv,o,r)) return -1;
    }
    for (j=0; j<o; ++j) {
        int64_t bound=0;
        for (k=0; k<count; ++k) {
            int64_t m=i32(b+40+k*payload+nv+nu+4*(n+r+j));
            bound+=rounded((int64_t)r*127*(m<0 ? -m : m));
        }
        if (bound>INT32_MAX) return -1;
    }
    for (j=0; j<o; ++j) {
        int64_t output=0;
        for (k=0; k<count; ++k) {
            const uint8_t *v=b+40+k*payload, *u=v+nv;
            const uint8_t *mg=u+nu, *m1=mg+4*n, *m2=m1+4*r;
            int32_t acc2=0;
            /* Allocation-free reference: recompute latent values per output.
             * A production kernel should cache r int8 values instead. */
            for (t=0; t<r; ++t) {
                int32_t acc=0;
                int8_t z;
                for (i=0; i<n; ++i) {
                    int32_t q=tinylm_factor_requantize(x[i],i32(mg+4*i),20);
                    acc+=((v[t*vs+i/8]>>(i%8))&1) ? q : -q;
                }
                z=tinylm_factor_requantize(acc,i32(m1+4*t),20);
                acc2+=((u[j*us+t/8]>>(t%8))&1) ? z : -z;
            }
            output+=rounded((int64_t)acc2*i32(m2+4*j));
        }
        y[j]=(int32_t)output;
    }
    return 0;
}
