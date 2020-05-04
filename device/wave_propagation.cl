#define nx 471

__kernel void wave_propagation(const int nz,
                               const int dx,
                               const float dt,
                               __global const float *restrict velocity, 
                               __global const float *restrict density, 
                               __global const float *restrict abs_model, 
                               __global const float *restrict prev, 
                               __global const float *restrict curr, 
                               __global const float *restrict density_init, 
                               __global const float *restrict prev_init, 
                               __global const float *restrict curr_init, 
                               __global float* restrict next) 
{

    float density_buff[6*nx+1];
    float curr_buff[6*nx+1];

    int left_1 = 3 * nx - 3;
    int left_2 = 3 * nx - 3;
    int left_3 = 3 * nx - 1;
    int curr_0 = 3 * nx+1;
    int right_1 = 3 * nx + 1;
    int right_2 = 3 * nx + 2;
    int right_3 = 3 * nx + 3;
    int top_1 = 0;
    int top_2 = nx;
    int top_3 = 2*nx;
    int down_1 = 4*nx;
    int down_2 = 5*nx;
    int down_3 = 6*nx;

    // fill with initial values
    #pragma unroll
    for (int i = 0; i < 6*nx; ++i) {
        curr_buff[i+1] = curr_init[i];
    }

    #pragma unroll
    for (int i = 0; i < 6*nx; ++i) {
        density_buff[i+1] = density_init[i];
    }

    float dtdx2 = pow(dt, 2)/pow(dx, 2);
    int total_size = nx * (nz-6);

    for (int cell = 0; cell < total_size; ++cell)
    {
        #pragma unroll
        for (int i = 0; i < 6*nx; ++i){
            curr_buff[i] = curr_buff[i+1];
        }

        #pragma unroll
        for (int i = 0; i < 6*nx; ++i){
            density_buff[i] = density_buff[i+1];
        }

        density_buff[6*nx] = density[cell];
        curr_buff[6*nx] = curr[cell];

        // calculate d2u/dx2 using 6th order accuracy
        float result = 0;
        if (cell % nx >= 3 && cell % nx < nx-3) {
          float this_value = curr_buff[curr_0];
          float this_den = density_buff[curr_0];
          float d2z = 1.0/90.0*(curr_buff[top_1]*density_buff[top_1] + 
                                curr_buff[down_3]*density_buff[down_3]) -
                      3.0/20.0*(curr_buff[top_2]*density_buff[top_2] + 
                                curr_buff[down_2]*density_buff[down_2]) +
                      3.0/2.0*(curr_buff[top_3]*density_buff[top_3] + 
                               curr_buff[down_1]*density_buff[down_1]) -
                      49.0/18.0*this_value*this_den;

          float d2x = 1.0/90.0*(curr_buff[left_1]*density_buff[left_1] + 
                                curr_buff[right_3]*density_buff[right_3]) -
                      3.0/20.0*(curr_buff[left_2]*density_buff[left_2] + 
                                curr_buff[right_2]*density_buff[right_2]) +
                      3.0/2.0*(curr_buff[left_3]*density_buff[left_3] + 
                               curr_buff[right_1]*density_buff[right_1]) -
                      49.0/18.0*this_value*this_den;
  
          //perform update of wave pressure
          double q = abs_model[cell];
          result = (dtdx2*pow(velocity[cell], 2)*(d2x+d2z)/this_den +
                   (2-pow(q, 2))*this_value-(1-q)*prev[cell])/(1+q);
        }
        next[cell] = result;
    }
}
