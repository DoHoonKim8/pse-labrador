// test_dachshund_quad.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "poly.h"
#include "polx.h"
#include "polz.h"
#include "data.h"
#include "labrador.h"
#include "dachshund.h"

// -------------------- triangular indexing (row-major upper) ------------------
// Pack (i,j) with i<=j in row-major order.
static inline size_t triangularidx(size_t i, size_t j, size_t r) {
    if (i > j) { size_t t = i; i = j; j = t; }
    // entries before row i: i*r - (i*(i+1))/2
    return i * r - (i * (i + 1)) / 2 + j;
}

// ----------------------------- helpers --------------------------------------

// Build deterministic witness + statement. We free betasq here because
// init_smplstmnt_raw copies it (LeakSanitizer previously showed a small leak).
static void make_witness_and_statement(smplstmnt *st,
                                       witness *wt,
                                       size_t r,
                                       const size_t n_full[r],
                                       size_t num_constraints) {
    // 1) Allocate witness with actual lengths
    init_witness_raw(wt, r, n_full);

    // 2) Deterministic seed ⇒ reproducible vectors
    __attribute__((aligned(16))) uint8_t seed[16];
    for (int i = 0; i < 16; ++i) seed[i] = (uint8_t)(7 + 13 * i);
    uint64_t nonce = 0;

    // 3) Fill witness (polynomial vectors) and compute beta^2 bounds
    uint64_t *betasq = (uint64_t *)malloc(sizeof(uint64_t) * r);

    for (size_t i = 0; i < r; ++i) {
        // wt->s[i] is a vector of polys of length wt->n[i]
        polyvec_ternary(wt->s[i], wt->n[i], seed, nonce++);
        wt->normsq[i] = polyvec_sprodz(wt->s[i], wt->s[i], wt->n[i]);
        betasq[i] = wt->normsq[i] + 1024;
    }

    // 4) Statement mirrors witness lengths / bounds
    int rc = init_smplstmnt_raw(st, r, n_full, betasq, num_constraints);
    if (rc != 0) {
        fprintf(stderr, "init_smplstmnt_raw failed (rc=%d)\n", rc);
    }
    free(betasq);
}

static int64_t zz_toint64(const zz *r) {
  // Start with the most-significant limb (carries the sign).
  int64_t a = (int64_t)r->limbs[L - 1];

  // Fold lower limbs back in, high to low, 14 bits at a time.
  // Each lower limb stores a 14-bit chunk (masked with 0x3FFF on write).
  for (size_t i = L - 1; i-- > 0; ) {
    a <<= 14;
    a |= (int64_t)(r->limbs[i] & 0x3FFF);
  }
  return a;
}

// Compute b(x) = sum_{i<=j} a_ij * < s_i, s_j >_ring as a polynomial,
// and write its N coefficients into out_b_ints.
// Uses only the provided polx API:
//   - convert poly -> polx:  polxvec_frompolyvec()
//   - inner product:         polxvec_sprod()
//   - scaling and add:       polx_scale(), polx_add()
//   - extract coeffs:        polx_getcoeff()
static void compute_b_poly_ints(const witness *wt,
                                size_t nz,
                                size_t deg,
                                const size_t idx[],  // which witness vectors
                                int64_t *A0,   // triangular scalars, row-major
                                int64_t out_b_ints[N]) {
    // Start with zero polynomial
    polx b_poly;
    polxvec_setzero(&b_poly, 1);

    for (size_t i = 0; i < nz; ++i) {
        for (size_t j = i; j < nz; ++j) {
            polz t;
            polzvec_fromint64vec(&t, 1, deg, A0);
            polx aij;
            polzvec_topolxvec(&aij, &t, deg);

            const size_t ii = idx[i], jj = idx[j];
            const size_t len = wt->n[ii]; // equals wt->n[jj] by construction

            // Convert witness slices from poly to polx (NTT representation)
            polx *Si = (polx *)aligned_alloc(64, len * sizeof(polx));
            polx *Sj = (polx *)aligned_alloc(64, len * sizeof(polx));
            assert(Si && Sj);
            polxvec_frompolyvec(Si, wt->s[ii], len);
            polxvec_frompolyvec(Sj, wt->s[jj], len);

            // acc = <Si, Sj> in the ring
            polx acc;
            polxvec_sprod(&acc, Si, Sj, len);

            // scaled = aij * acc
            polx scaled;
            polx_mul(&scaled, &acc, &aij);

            // b_poly += scaled
            polx_add(&b_poly, &b_poly, &scaled);

            free(Si);
            free(Sj);

            A0 += deg * N;
        }
    }
    polx_print(&b_poly);

    // Extract N coefficients of b_poly into out_b_ints
    for (int k = 0; k < (int)N; ++k) {
        zz c;
        polx_getcoeff(&c, &b_poly, k);
        // 'zz' is defined in data.h; typically an integer type
        out_b_ints[k] = zz_toint64(&c);
    }

    polx b_recon;
    polxvec_fromint64vec(&b_recon, 1, deg, out_b_ints);
    polx diff;
    polx_sub(&diff, &b_poly, &b_recon);
    if (!polx_iszero(&diff)) {
        fprintf(stderr, "Self-check failed: b_poly != recon(b_ints)\n");
        polx_print(&b_poly);
        polx_print(&b_recon);
    }
}

// Allocate and fill quad_coeffs in row‑major upper triangle.
// Each triangular entry consumes (deg * N) int64_t, matching set_quadcnst_raw:
//   polzvec_fromint64vec(t, /*len=*/1, deg, quad_coeffs);
// We encode a scalar a_ij by setting only (k=0, limb=0) to a_ij (others zero).
static int64_t *alloc_and_fill_quad_coeffs_row_major_scalars(size_t nz,
                                                             size_t deg,
                                                             const int64_t *A0) {
    const size_t tri_len = nz * (nz + 1) / 2;
    const size_t block   = deg * N;          // per (i,j)
    const size_t total   = tri_len * block;

    int64_t *A = (int64_t *)calloc(total, sizeof(int64_t));
    assert(A);

    size_t off = 0;
    for (size_t i = 0; i < nz; ++i) {
        for (size_t j = i; j < nz; ++j) {
            const int64_t aij = A0[triangularidx(i, j, nz)];
            A[off] = aij;  // constant polynomial
            off += block;
        }
    }
    assert(off == total);
    return A;
}

// ----------------------------- tests -----------------------------------------

// We keep deg = 1 for clarity and to match the test b(x) construction.
// (Once everything passes, you can experiment with deg=64 and decide per‑limb semantics.)
static int test_quadratic_single_vector(void) {
    const size_t deg = 1;

    // Witness actual length = n[0] * deg = 1
    const size_t r = 1;
    const size_t n_full[1] = { deg };

    smplstmnt st = (smplstmnt){0};
    witness   wt = (witness){0};
    make_witness_and_statement(&st, &wt, r, n_full, /*num_constraints=*/1);

    const size_t nz = 1;
    const size_t idx[1]   = { 0 };
    const size_t n_sub[1] = { 1 };  // base length
    for (size_t j = 0; j < nz; ++j) {
        assert(n_sub[j] * deg == st.n[idx[j]]);
    }

    // Triangular scalars: [a00]
    int64_t A0[1] = { 1 };

    // Quadratic coeffs buffer: tri_len * deg * N, row-major upper
    int64_t *A = alloc_and_fill_quad_coeffs_row_major_scalars(nz, deg, A0);

    // Compute b(x) in the ring and encode to int64 buffer (deg*N ints)
    int64_t *b = (int64_t *)calloc(deg * N, sizeof(int64_t));  // deg=1 → N
    assert(b);
    compute_b_poly_ints(&wt, nz, deg, idx, A, &b[0]);

    int rc = set_smplstmnt_quadcnst_raw(&st,
                                        0,
                                        nz,
                                        idx,
                                        n_sub,
                                        deg,
                                        A,
                                        b);
    if (rc != 0) {
        fprintf(stderr, "set_smplstmnt_quadcnst_raw failed (single, rc=%d)\n", rc);
        free(A); free(b);
        free_smplstmnt(&st);
        free_witness(&wt);
        return rc ? rc : -1;
    }

    int v = simple_verify(&st, &wt);
    if (v != 0) fprintf(stderr, "simple_verify failed (single) v=%d\n", v);

    free(A); free(b);
    free_smplstmnt(&st);
    free_witness(&wt);
    return v;
}

static int test_quadratic_two_vectors(void) {
    const size_t deg = 1;

    // Two vectors, each of actual length = 1
    const size_t r = 2;
    const size_t n_full[2] = { deg, deg };

    smplstmnt st = (smplstmnt){0};
    witness   wt = (witness){0};
    make_witness_and_statement(&st, &wt, r, n_full, /*num_constraints=*/1);

    const size_t nz = 2;
    const size_t idx[2]   = { 0, 1 };
    const size_t n_sub[2] = { 1, 1 };
    for (size_t j = 0; j < nz; ++j) {
        assert(n_sub[j] * deg == st.n[idx[j]]);
        assert(st.n[idx[j]] == st.n[idx[0]]);
    }

    // Row-major triangular A0: [a00, a01, a11]
    const int64_t a00 = 2, a01 = 5, a11 = 3;
    int64_t A0[3];
    A0[triangularidx(0,0,nz)] = a00;
    A0[triangularidx(0,1,nz)] = a01;
    A0[triangularidx(1,1,nz)] = a11;

    // quad_coeffs: tri_len * deg * N, row-major upper
    int64_t *A = alloc_and_fill_quad_coeffs_row_major_scalars(nz, deg, A0);

    // Compute b(x) in the ring and encode to int64 buffer (deg*N ints)
    int64_t *b = (int64_t *)calloc(deg * N, sizeof(int64_t));  // deg=1 → N
    assert(b);
    compute_b_poly_ints(&wt, nz, deg, idx, A, &b[0]);

    int rc = set_smplstmnt_quadcnst_raw(&st,
                                        /*i=*/0,
                                        nz,
                                        idx,
                                        n_sub,
                                        deg,
                                        /*quad_coeffs=*/A,
                                        /*b=*/b);
    if (rc != 0) {
        fprintf(stderr, "set_smplstmnt_quadcnst_raw failed (two, rc=%d)\n", rc);
        free(A); free(b);
        free_smplstmnt(&st);
        free_witness(&wt);
        return rc ? rc : -1;
    }

    int v = simple_verify(&st, &wt);
    if (v != 0) fprintf(stderr, "simple_verify failed (two) v=%d\n", v);

    free(A); free(b);
    free_smplstmnt(&st);
    free_witness(&wt);
    return v;
}

// -------------------------------- main ---------------------------------------

int main(void) {
    int rc;

    rc = test_quadratic_single_vector();
    if (rc) return rc;

    // rc = test_quadratic_two_vectors();
    // if (rc) return rc;

    free_comkey();
    return 0;
}
