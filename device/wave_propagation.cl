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

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        shift_den[i] = density[3*nx+i]; 
        shift_curr[i] = curr[3*nx+i]; 
        shift_denvl[i+4] = shift_den[i] * shift_curr[i];
    }

    float dtdx2 = pow(dt, 2)/pow(dx, 2);

    #pragma ivdep array(next)
    for (int cell = 0; cell < nx*(nz-6); ++ cell)
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

        // calculate d2u/dx2 using 6th order accuracy
        float result = 0;
        if (cell % nx >= 3 && cell % nx < nx-3) {
            float d2z = 1.0/90.0*(curr[cell] * density[cell] + 
                                  curr[cell + 6*nx] * density[cell + 6*nx]) -
                        3.0/20.0*(curr[cell + nx] * density[cell + nx] + 
                                  curr[cell + 5*nx] * density[cell + 5*nx]) +
                        3.0/2.0*(curr[cell + 2*nx] * density[cell + 2*nx] + 
                                  curr[cell + 4*nx] * density[cell + 4*nx]) -
                        49.0/18.0*shift_denvl[3];

            float d2x = 1.0/90.0*(shift_denvl[0] + shift_denvl[6]) -
                        3.0/20.0*(shift_denvl[1] + shift_denvl[5]) +
                        3.0/2.0*(shift_denvl[2] + shift_denvl[4]) -
                        49.0/18.0*shift_denvl[3];
  
            //perform update of wave pressure
            float q = abs_model[cell];
            float vel = velocity[cell];

            result = (dtdx2*vel*vel*(d2x+d2z)/this_den + (2-q*q)*this_value-(1-q)*prev[cell])/(1+q);
        }
        next[cell] = result;
    }
}
