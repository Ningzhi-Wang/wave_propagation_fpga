#define nx 571
#define VECTOR_SIZE 16
#define BUFFER_SIZE 6*nx+VECTOR_SIZE

#define GET_VALUE(dtdx2, vel, q, den, curr, prev, tv, td, nx, idx)  \
    ((union ufloat) {.f = (dtdx2*vel*vel*( \
        1.0f/90.0f*(curr[idx-3*nx] * den[idx-3*nx] + \
                  curr[idx+3*nx] * den[idx+3*nx]) - \
        3.0f/20.0f*(curr[idx-2*nx] * den[idx-2*nx] + \
                  curr[idx+2*nx] * den[idx+2*nx]) + \
        3.0f/2.0f*(curr[idx-nx] * den[idx-nx] + \
                  curr[idx+nx] * den[idx+nx]) - \
        49.0f/18.0f*td*tv +\
        1.0f/90.0f*(curr[idx-3]*den[idx-3] + \
                  curr[idx+3]*den[idx+3]) - \
        3.0f/20.0f*(curr[idx-2]*den[idx-2] + \
                  curr[idx+2]*den[idx+2]) + \
        3.0f/2.0f*(curr[idx-1]*den[idx-1] + \
                 curr[idx+1]*den[idx+1]) - \
        49.0f/18.0f*td*tv \
   )/td+ (2-q*q)*tv-(1-q)*prev)/(1+q)}).u 


union ufloat {
    float f;
    unsigned u;
};

__kernel void wave_propagation(const int nz,
                               const int dx,
                               const float dt,
                               __global const float16 *restrict velocity, 
                               __global const float16 *restrict density, 
                               __global const float16 *restrict abs_model, 
                               __global const float16 *restrict prev, 
                               __global const float16 *restrict curr, 
                               __global float16* restrict next) 
{
    float curr_buff[BUFFER_SIZE];
    float den_buff[BUFFER_SIZE];
    float den_remains[VECTOR_SIZE];
    float curr_remains[VECTOR_SIZE];
    int remainder = VECTOR_SIZE - 6*nx%VECTOR_SIZE;

    float dtdx2 = pow(dt, 2)/pow(dx, 2);
    int total_size = ceil(nx*nz/VECTOR_SIZE);
    int offset = 6*nx/VECTOR_SIZE;

    for (int cell = 0; cell < total_size; cell++)
    {
        int num_remains = remainder & (cell >= offset);

        #pragma unroll
        for (int i = 0; i < BUFFER_SIZE-VECTOR_SIZE; ++i) {
            curr_buff[i] = curr_buff[i+VECTOR_SIZE];
            den_buff[i] = den_buff[i+VECTOR_SIZE];
        }

        float16 tmp_den = density[cell];
        float16 tmp_curr = curr[cell];

        
        #pragma unroll
        for (int i = 0; i < num_remains; ++i) {
            curr_buff[6*nx+i] = curr_remains[i];
            den_buff[6*nx+i] = den_remains[i];
            curr_remains[i] = tmp_curr[i+VECTOR_SIZE-num_remains];
            den_remains[i] = tmp_den[i+VECTOR_SIZE-num_remains];
        }

        #pragma unroll
        for (int i = num_remains; i < VECTOR_SIZE; ++i) {
            curr_buff[6*nx+i] = tmp_curr[i-num_remains];
            den_buff[6*nx+i] = tmp_den[i-num_remains];
        }

        if (cell < offset)  
            continue;

        const float16 vels = velocity[cell-offset];
        const float16 prevs = prev[cell-offset];
        const float16 abses = abs_model[cell-offset];

        float16 results;
        #pragma unroll
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            // calculate d2u/dx2 using 6th order accuracy
            int mask = ~ ((cell*VECTOR_SIZE+i)% nx < 3 || (cell*VECTOR_SIZE+i) % nx >= nx-3);

            unsigned result_u = mask & GET_VALUE(dtdx2, vels[i], abses[i], den_buff, 
                curr_buff, prevs[i], curr_buff[3*nx+i], den_buff[3*nx+i], nx, 3*nx+i);
            results[i] = *(float*) &result_u;
        }
        next[cell-offset] = results;
    }
}
