#define nx 571
#define BUFFER_SIZE 6*nx+1

union ufloat {
    float f;
    unsigned u;
};


#define GET_VALUE(dtdx2, vel, q, den, curr, prev, tv, td, nx)  \
    ((union ufloat) {.f = (dtdx2*vel*vel*( \
        1.0f/90.0f*(curr[0] * den[0] + \
                  curr[6*nx] * den[6*nx]) - \
        3.0f/20.0f*(curr[nx] * den[nx] + \
                  curr[5*nx] * den[5*nx]) + \
        3.0f/2.0f*(curr[2*nx] * den[2*nx] + \
                  curr[4*nx] * den[4*nx]) - \
        49.0f/18.0f*td*tv +\
        1.0f/90.0f*(curr[3*nx-3]*den[3*nx-3] + \
                  curr[3*nx+3]*den[3*nx+3]) - \
        3.0f/20.0f*(curr[3*nx-2]*den[3*nx-2] + \
                  curr[3*nx+2]*den[3*nx+2]) + \
        3.0f/2.0f*(curr[3*nx-1]*den[3*nx-1] + \
                 curr[3*nx+1]*den[3*nx+1]) - \
        49.0f/18.0f*td*tv \
   )/td+ (2-q*q)*tv-(1-q)*prev)/(1+q)}).u 


__kernel void wave_propagation(const int nz,
                               const int dx,
                               const float dt,
                               __global const float *restrict velocity, 
                               __global const float *restrict density, 
                               __global const float *restrict abs_model, 
                               __global const float *restrict prev, 
                               __global const float *restrict curr, 
                               __global float* restrict next) 
{
    float den_buff[BUFFER_SIZE];
    float curr_buff[BUFFER_SIZE];

    float dtdx2 = pow(dt, 2)/pow(dx, 2);
    int total_size = nx * nz;

    for (int cell = 0; cell < total_size; ++cell)
    {
        #pragma unroll
        for (int i = 0; i < 6*nx; ++i) {
            curr_buff[i] = curr_buff[i+1];
            den_buff[i] = den_buff[i+1];
        }
        curr_buff[6*nx] = curr[cell];
        den_buff[6*nx] = density[cell];

        if (cell < 6*nx)
            continue;

        int mask = - (cell % nx >= 3 && cell % nx < nx-3);
        // calculate d2u/dx2 using 6th order accuracy
        float result;
        unsigned result_u = mask & GET_VALUE(dtdx2, velocity[cell-6*nx], abs_model[cell-6*nx], 
            den_buff, curr_buff, prev[cell-6*nx], curr_buff[3*nx], den_buff[3*nx], nx);
        result = *(float*) &result_u;
        next[cell-6*nx] = result;
    }
}
