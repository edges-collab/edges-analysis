#include <math.h>
#include <stdio.h>
#include <stdlib.h> // For malloc and free

void qrd(long double a[], int n, double b[]) {
    /*
    QR-decomposition. This is included here to match the results of the original
    EDGES C-code pipeline. This is compiled and used in the edges.modeling module
    as one of the methods for fitting a linear model to data. While it gives similar
    results to QR-decomposition done with numpy (or lstsq for that matter), it differs
    in the numerical details.

    For instance, various dot-products are here written out as cumulative sums.
    In numpy, these sums use pairwise-summation, which is more accurate. However, the
    difference in precision can cause differences in the final fitted parameters due to
    the often poorly-conditioned nature of the fitting matrices. Implementing the
    non-pairwise-summation in numpy requires use of cumsum() which is very very slow,
    so we instead just include this small compiled function.
    */
    int i, j, k;
    int pi, pk;
    long double *c = (long double *)malloc(n * sizeof(long double)); // Allocate memory for 'n' long doubles
    long double *d = (long double *)malloc(n * sizeof(long double)); // Allocate memory for 'n' long doubles
    long double scale, sigma, sum, tau;
    long double **qt = (long double **)malloc(n * sizeof(long double *));
    long double **u = (long double **)malloc(n * sizeof(long double *));
    for (i = 0; i < n; i++) {
        qt[i] = (long double *)malloc(n * sizeof(long double));
        u[i] = (long double *)malloc(n * sizeof(long double));
    }

    for (k = 0; k < n - 1; k++) {
        scale = 0.0;
        for (i = k; i < n; i++)
        if (fabsl(a[k + i * n])) scale = fabsl(a[k + i * n]);
        if (scale == 0.0) {
        printf("singular!\n");
        c[k] = d[k] = 0.0;
        } else {
        for (i = k; i < n; i++) a[k + i * n] /= scale;
        for (sum = 0.0, i = k; i < n; i++) sum += a[k + i * n] * a[k + i * n];
        if (a[k + k * n] > 0)
            sigma = sqrtl(sum);
        else
            sigma = -sqrtl(sum);
        a[k + k * n] += sigma;
        c[k] = sigma * a[k + k * n];
        d[k] = -scale * sigma;
        for (j = k + 1; j < n; j++) {
            for (sum = 0.0, i = k; i < n; i++) sum += a[k + i * n] * a[j + i * n];
            tau = sum / c[k];
            for (i = k; i < n; i++) a[j + i * n] -= tau * a[k + i * n];
        }
        }
    }
    d[n - 1] = a[n - 1 + (n - 1) * n];
    if (d[n - 1] == 0.0) printf("singular2\n");
    for (i = 0; i < n; i++) {  // Form QT explicitly.
        for (j = 0; j < n; j++) qt[i][j] = 0.0;
        qt[i][i] = 1.0;
    }
    for (k = 0; k < n - 1; k++) {
        if (c[k] != 0.0) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (i = k; i < n; i++) sum += a[k + i * n] * qt[i][j];
            sum /= c[k];
            for (i = k; i < n; i++) qt[i][j] -= sum * a[k + i * n];
        }
        }
    }
    for (j = 0; j < n - 1; j++) {
        for (sum = 0, i = j; i < n; i++) sum += a[j + i * n] * b[i];
        tau = sum / c[j];
        for (i = j; i < n; i++) b[i] -= tau * a[j + i * n];
    }

    b[n - 1] /= d[n - 1];
    for (i = n - 2; i >= 0; i--) {
        for (sum = 0, j = i + 1; j < n; j++) sum += a[j + i * n] * b[j];
        b[i] = (b[i] - sum) / d[i];
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        if (j > i)
            u[i][j] = a[j + i * n];
        else
            u[i][j] = 0;
        if (i == j) u[i][i] = d[i];
        }
    }

    for (k = 0; k < n; k++) {
        if (u[k][k] == 0)
        return;
        else
        u[k][k] = 1.0 / u[k][k];
    }

    for (i = n - 2, pi = (n - 2); i >= 0; pi -= 1, i--) {
        for (j = n - 1; j > i; j--) {
        sum = 0.0;
        for (k = i + 1, pk = pi + 1; k <= j; pk += 1, k++) { sum += u[pi][k] * u[pk][j]; }
        u[pi][j] = -u[pi][i] * sum;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        sum = 0;
        for (k = 0; k < n; k++) sum += u[i][k] * qt[k][j];
        a[j + i * n] = sum;
        }
    }
}


void get_a_and_b(
    int npoly, int nfreq, double basis[], double ddata[], double wtt[],
    long double aarr[],
    double bbrr[]
) {
    /*
        Linear fit to data using basis functions provided in basis[].
        This is almost exactly the same as polyfit in the original EDGES C-code,
        but it can use an arbitrary set of basis functions.
    */
    int i, j, k, kk, m1, m2;
    double re;

    for (i = 0; i < npoly; i++) {
        re = 0.0;
        for (k = 0; k < nfreq; k++) {
            if (wtt[k] > 0) re += basis[k*npoly + i] * ddata[k] * wtt[k];
        }
        bbrr[i] = re;

        for (j = 0; j < npoly; j++) {
            re = 0.0;
            for (k = 0; k < nfreq; k++) {
                if (wtt[k] > 0) re += basis[k*npoly + i] * basis[k*npoly + j] * wtt[k];
            }
            k = j + i * npoly;
            aarr[k] = re;
        }
    }
}
