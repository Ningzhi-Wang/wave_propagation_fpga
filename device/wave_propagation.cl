typedef struct wave_model_2d
{
    int nx;
    int nz;
    int dx;
    float dt;
} WAVE_MODLE_2D;

__kernel void wave_propagation(const int nx, 
                               const int nz,
                               const int dx,
                               const float dt,
                              __global const float* velocity, 
                              __global const float* density, 
                              __global const float* abs_model, 
                              __global const float* prev, 
                              __global const float* curr, 
                              __global const float* density_init, 
                              __global const float* prev_init, 
                              __global const float* curr_init, 
                              __global float* restrict next,
                              __local float* density_buff,
                              __local float* prev_buff,
                              __local float* curr_buff,
                              __local int* indices) 
{
    // The size of continuous data needed for calculate each new value
    int buffer_size = 6*nx+1;

    // The number of values/positions needed for new value calculation
    int num_positions = 13;

    // (i-3), j: 0; (i-2), j: 1; (i-1), j: 2; i, j: 3; (i+1), j: 4;
    // (i+2), j: 5; (i+3), j: 6;  i, j-3: 7; i, j-2: 8; i, j-1: 9;
    // i, j+1: 10; i, j+2: 11;  i, j+3: 12

    //set the ones on the vertical direction
    for (int i = 0; i < 7; ++i) {
        indices[i] = i * nx;
    }

    //set the ones on the horizontal direction
    for (int i = 7; i < 10; ++i) {
        indices[i] = 3 * nx - 10 + i;
    }
    for (int i = 10; i < num_positions; ++i) {
        indices[i] = 3*nx - 9 + i;
    }

    // fill with initial values
    for (int i = 0; i < buffer_size-1; ++i) {
        density_buff[i+1] = density_init[i];
        prev_buff[i+1] = prev_init[i];
        curr_buff[i+1] = curr_init[i];
    }

    float dtdx2 = pow(dt, 2)/pow(dx, 2);
    int total_size = nx * (nz-6);

    for (int cell = 0; cell < total_size; ++cell)
    {
        //Get next value and update value position/indices.
        density_buff[indices[0]] = density[cell];
        prev_buff[indices[0]] = prev[cell];
        curr_buff[indices[0]] = curr[cell];
        for (int i = 0; i <  num_positions; ++i) {
            indices[i] = (indices[i] + 1) % buffer_size;
        }

        // calculate d2u/dx2 using 6th order accuracy
        float result = 0;
        if (cell % nx >= 3 && cell % nx < nx-3) {
          float d2x = 1.0/90.0*(curr_buff[indices[7]]*density_buff[indices[7]] + 
                                curr_buff[indices[12]]*density_buff[indices[12]]) -
                      3.0/20.0*(curr_buff[indices[8]]*density_buff[indices[8]] + 
                                curr_buff[indices[11]]*density_buff[indices[11]]) +
                      3.0/2.0*(curr_buff[indices[9]]*density_buff[indices[9]] + 
                               curr_buff[indices[10]]*density_buff[indices[10]]) -
                      49.0/18.0*curr_buff[indices[3]]*density_buff[indices[3]];

          float d2z = 1.0/90.0*(curr_buff[indices[0]]*density_buff[indices[0]] + 
                                curr_buff[indices[6]]*density_buff[indices[6]]) -
                      3.0/20.0*(curr_buff[indices[1]]*density_buff[indices[1]] + 
                                curr_buff[indices[5]]*density_buff[indices[5]]) +
                      3.0/2.0*(curr_buff[indices[2]]*density_buff[indices[2]] + 
                               curr_buff[indices[4]]*density_buff[indices[4]]) -
                      49.0/18.0*curr_buff[indices[3]]*density_buff[indices[3]];
  
          //perform update of wave pressure
          double q = abs_model[cell];
          result = (dtdx2*pow(velocity[cell], 2)*(d2x+d2z)/density_buff[indices[3]] +
                   (2-pow(q, 2))*curr_buff[indices[3]]-(1-q)*prev_buff[indices[3]])/(1+q);
        }
        next[cell] = result;
    }
}