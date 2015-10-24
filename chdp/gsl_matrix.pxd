from cython_gsl cimport *
from cpython.mem cimport PyMem_Malloc, PyMem_Free

ctypedef unsigned int uint

cdef inline gsl_vector** alloc_2d_gsl_matrix(uint A, uint B, double val):
    cdef gsl_vector **m = <gsl_vector**>PyMem_Malloc(A * sizeof(gsl_vector*))
    if not m:
        raise MemoryError()
    cdef size_t i
    for i in range(A):
        m[i] = gsl_vector_alloc(B)
        gsl_vector_set_all(m[i], val)
    return m

cdef inline gsl_vector*** alloc_3d_gsl_matrix(uint A, uint B, uint C, double val):
    cdef gsl_vector ***m = <gsl_vector***>PyMem_Malloc(A * sizeof(gsl_vector**))
    if not m:
        raise MemoryError()
    cdef size_t i
    for i in range(A):
        m[i] = alloc_2d_gsl_matrix(B, C, val)
    return m

cdef inline void free_2d_gsl_matrix(gsl_vector** m, uint A):
    cdef size_t i
    for i in range(A):
        gsl_vector_free(m[i])
    PyMem_Free(m)

cdef inline void free_3d_gsl_matrix(gsl_vector*** m, uint A, uint B):
    cdef size_t i
    for i in range(A):
        free_2d_gsl_matrix(m[i], B)
    PyMem_Free(m)

# warning: no range check below
cdef inline void gsl_1d_matrix_incr(gsl_vector* m, size_t i, double x):
    (gsl_vector_ptr(m, i))[0] += x

cdef inline void gsl_2d_matrix_incr(gsl_vector** m, size_t i, size_t j, double x):
    (gsl_vector_ptr(m[i], j))[0] += x

cdef inline void gsl_3d_matrix_incr(gsl_vector*** m, size_t i, size_t j, size_t k, double x):
    (gsl_vector_ptr(m[i][j], k))[0] += x

cdef inline void gsl_2d_matrix_add_constant(gsl_vector** m, size_t i, double x):
    cdef size_t k
    for k in range(i):
        gsl_vector_add_constant(m[k], x)

cdef inline void gsl_3d_matrix_add_constant(gsl_vector*** m, size_t i, size_t j, double x):
    cdef size_t k
    for k in range(i):
        gsl_2d_matrix_add_constant(m[k], j, x)

cdef inline void gsl_2d_matrix_set_all(gsl_vector** m, size_t i, double x):
    cdef size_t k
    for k in range(i):
        gsl_vector_set_all(m[k], x)

cdef inline void gsl_3d_matrix_set_all(gsl_vector*** m, size_t i, size_t j, double x):
    cdef size_t k
    for k in range(i):
        gsl_2d_matrix_set_all(m[k], j, x)

cdef inline void gsl_2d_matrix_scale(gsl_vector** m, size_t i, double x):
    cdef size_t k
    for k in range(i):
        gsl_vector_scale(m[k], x)

cdef inline void gsl_3d_matrix_scale(gsl_vector*** m, size_t i, size_t j, double x):
    cdef size_t k
    for k in range(i):
        gsl_2d_matrix_scale(m[k], j, x)
