#define nx 571
#define BUFFER_SIZE 6*nx+8

#define GET_VALUE(dtdx2, vel, q, den, curr, prev, tv, td, nx, idx)  \
    ((union ufloat) {.f = (dtdx2*vel*vel*( \
        1.0/90.0*(curr[idx-3*nx] * den[idx-3*nx] + \
                  curr[idx+3*nx] * den[idx+3*nx]) - \
        3.0/20.0*(curr[idx-2*nx] * den[idx-2*nx] + \
                  curr[idx+2*nx] * den[idx+2*nx]) + \
        3.0/2.0*(curr[idx-nx] * den[idx-nx] + \
                  curr[idx+nx] * den[idx+nx]) - \
        49.0/18.0*td*tv +\
        1.0/90.0*(curr[idx-3]*den[idx-3] + \
                  curr[idx+3]*den[idx+3]) - \
        3.0/20.0*(curr[idx-2]*den[idx-2] + \
                  curr[idx+2]*den[idx+2]) + \
        3.0/2.0*(curr[idx-1]*den[idx-1] + \
                 curr[idx+1]*den[idx+1]) - \
        49.0/18.0*td*tv \
   )/td+ (2-q*q)*tv-(1-q)*prev)/(1+q)}).u 


union ufloat {
    float f;
    unsigned u;
};

__kernel void wave_propagation(const int nz,
                               const int dx,
                               const float dt,
                               __global const float8 *restrict velocity, 
                               __global const float8 *restrict density, 
                               __global const float8 *restrict abs_model, 
                               __global const float8 *restrict prev, 
                               __global const float8 *restrict curr, 
                               __global float8* restrict next) 
{
    float curr_buff[BUFFER_SIZE];
    float den_buff[BUFFER_SIZE];
    float den_remains[8];
    float curr_remains[8];
    int remainder = 8 - 6*nx%8;

    float dtdx2 = pow(dt, 2)/pow(dx, 2);
    int total_size = ceil(nx*nz/8);
    int offset = 6*nx/8;

    for (int cell = 0; cell < total_size; cell++)
    {
        int num_remains = remainder & (cell >= offset);

        #pragma unroll
        for (int i = 0; i < BUFFER_SIZE-8; ++i) {
            curr_buff[i] = curr_buff[i+8];
            den_buff[i] = den_buff[i+8];
        }

        float8 tmp_den = density[cell];
        float8 tmp_curr = curr[cell];

        
        #pragma unroll
        for (int i = 0; i < num_remains; ++i) {
            curr_buff[6*nx+i] = curr_remains[i];
            den_buff[6*nx+i] = den_remains[i];
            curr_remains[i] = tmp_curr[i+8-num_remains];
            den_remains[i] = tmp_den[i+8-num_remains];
        }

        #pragma unroll
        for (int i = num_remains; i < 8; ++i) {
            curr_buff[6*nx+i] = tmp_curr[i-num_remains];
            den_buff[6*nx+i] = tmp_den[i-num_remains];
        }

        if (cell < offset)  
            continue;

        const float8 vels = velocity[cell-offset];
        const float8 prevs = prev[cell-offset];
        const float8 abses = abs_model[cell-offset];

        float8 results;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            // calculate d2u/dx2 using 6th order accuracy
            int mask = ~ ((cell*8+i)% nx < 3 || (cell*8+i) % nx >= nx-3);

            unsigned result_u = mask & GET_VALUE(dtdx2, vels[i], abses[i], den_buff, 
                curr_buff, prevs[i], curr_buff[3*nx+i], den_buff[3*nx+i], nx, 3*nx+i);
            results[i] = *(float*) &result_u;
        }
        next[cell-offset] = results;
    }
}
