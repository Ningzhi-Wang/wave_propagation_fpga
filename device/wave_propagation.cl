#define nx 571
#define BUFFER_SIZE 6*nx+1

union ufloat {
    float f;
    unsigned u;
};


#define GET_VALUE(dtdx2, vel, q, den, curr, prev, tv, td, nx, o1, o2)  \
    ((union ufloat) {.f = (dtdx2*vel*vel*( \
        1.0f/90.0f*(curr[0+o1] * den[0+o1] + \
                  curr[6*nx] * den[6*nx]) - \
        3.0f/20.0f*(curr[nx+o2] * den[nx+o2] + \
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
                               const int src_loc,
                               const int receiver_depth,
                               const int num_steps,
                               const float dt,
                               __global const float *restrict velocity, 
                               __global const float *restrict density, 
                               __global const float *restrict abs_model, 
                               __global const float *restrict source, 
                               __global float* restrict prev, 
                               __global float* restrict curr, 
                               __global float* restrict next,
                               __global float* restrict output) 
{
    __global float* restrict fields[3] = {prev, curr, next};
    int pidx = 0;
    int cidx = 1;
    int nidx = 2;


    for (int i = 0; i < num_steps; ++i) {
        int tmp_idx = pidx;
        pidx = cidx;
        cidx = nidx;
        nidx = tmp_idx;

        fields[cidx][src_loc] = source[i];

        float den_buff[BUFFER_SIZE] = {0};
        float curr_buff[BUFFER_SIZE] = {0};

        float dtdx2 = pow(dt, 2)/pow(dx, 2);
        int total_size = nx * (nz-3);

        for (int cell = 0; cell < total_size; ++cell)
        {
            #pragma unroll
            for (int i = 0; i < 6*nx; ++i) {
                curr_buff[i] = curr_buff[i+1];
                den_buff[i] = den_buff[i+1];
            }
            curr_buff[6*nx] = fields[cidx][cell+3*nx];
            den_buff[6*nx] = density[cell+3*nx];

            int mask = - (cell > 3*nx && cell % nx >= 3 && cell % nx < nx-3);
            int idx_mask = - (cell < 5*nx);

            int offset_1 = (4*nx) & idx_mask;
            int offset_2 = (2*nx) & idx_mask;
            // calculate d2u/dx2 using 6th order accuracy
            float result;
            unsigned result_u = mask & GET_VALUE(dtdx2, velocity[cell], abs_model[cell], 
                den_buff, curr_buff, fields[pidx][cell], curr_buff[3*nx], den_buff[3*nx], nx, 
                offset_1, offset_2);
            result = *(float*) &result_u;
            fields[nidx][cell] = result;
        }
        //printf("source value: %0.3f, output value: %0.3f\n", source[i], fields[nidx][nx*73+185]);

        #pragma unroll
        for (int j = 0; j < nx; ++j) {
            output[nx*i+j] = fields[nidx][nx*receiver_depth+j];
        }

    }
}
