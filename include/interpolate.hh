#ifndef __INTERP_H__
#define __INTERP_H__

#include "global.h"

#ifdef __CUDACC__
#include "cusparse.h"

/*
CuSparse error checking
*/
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                            fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                            exit(-1);}} while(0)

#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

#endif

void interpolate_arrays(double *t_arr, double *interp_array, int ninterps, int length, double *B, double *upper_diag, double *diag, double *lower_diag);

void get_waveform(cmplx *d_waveform, double *interp_array,
              int *d_m, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
              double delta_t, double *h_t);

void get_waveform_fd(cmplx *waveform,
           double *interp_array,
           double *special_f_interp_array,
           double* special_f_seg_in,
            int *m_arr_in, int *n_arr_in, int num_teuk_modes, cmplx *Ylms_in,
            double* t_arr, int* start_ind_all, int* end_ind_all, int init_length,
            double start_freq, int* turnover_ind_all,
            double* turnover_freqs, int max_points, double df, double* f_data);

#endif // __INTERP_H__
