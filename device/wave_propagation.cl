
union ufloat {
    float f;
    unsigned u;
};

#define GET_VALUE(dtdx2, vel, q, den, curr, prev, sdv, tv, td, nx, idx)  \
    ((union ufloat) {.f = (dtdx2*vel*vel*( \
        1.0/90.0*(curr[idx] * den[idx] + \
                  curr[idx+ 6*nx] * den[idx + 6*nx]) - \
        3.0/20.0*(curr[idx + nx] * den[idx + nx] + \
                  curr[idx + 5*nx] * den[idx + 5*nx]) + \
        3.0/2.0*(curr[idx + 2*nx] * den[idx + 2*nx] + \
                  curr[idx + 4*nx] * den[idx + 4*nx]) - \
        49.0/18.0*sdv[3] +\
        1.0/90.0*(sdv[0] + sdv[6]) - 3.0/20.0*(sdv[1] + sdv[5]) + \
        3.0/2.0*(sdv[2] + sdv[4]) - 49.0/18.0*sdv[3] \
    )/td+ (2-q*q)*tv-(1-q)*prev)/(1+q)}).u 


__kernel void wave_propagation(const int nx, 
                               const int nz,
                               const int dx,
                               const float dt,
                               __global const float *restrict velocity, 
                               __global const float *restrict density, 
                               __global const float *restrict abs_model, 
                               __global const float *restrict prev, 
                               __global const float *restrict curr, 
                               __global float* restrict next) 
{
    float shift_denvl[7];
    float shift_den[3];
    float shift_curr[3];

    float dtdx2 = pow(dt, 2)/pow(dx, 2);
    int out = 0;

    for (int cell = -6; cell < nx*(nz-6)-6; ++ cell)
    {
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            shift_denvl[i] = shift_denvl[i+1];
        }

        float this_value = shift_curr[0];
        float this_den = shift_den[0];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            shift_den[i] = shift_den[i+1];
            shift_curr[i] = shift_curr[i+1];
        }

        shift_den[2] = density[3*nx+cell+3];
        shift_curr[2] = curr[3*nx+cell+3];
        shift_denvl[6] = shift_den[2]*shift_curr[2];

        if (cell < 0) 
            continue;

        int mask = ~ ((cell%nx)/(nx-6));

        unsigned result_u = mask & GET_VALUE(dtdx2, velocity[cell], abs_model[cell], 
            density, curr, prev[cell], shift_denvl, this_value, this_den, nx, cell);
        next[cell] = *(float*) &result_u;
    }
}
