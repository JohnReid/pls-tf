import numpy as np
import scipy.stats as st
import tensorflow as tf
import tensorflow_probability as tfp
from bluemyrtle import pls
la = tf.linalg


def test_polar_decomposition():
    _std_normal = tfp.distributions.Normal(loc=0, scale=1.)
    M, N = 5, 4
    B = _std_normal.sample((M, N))
    U = tf.constant(st.ortho_group.rvs(M), dtype=B.dtype)
    A = la.matmul(U, B)
    assert np.isclose(A.numpy(), tf.matmul(U, B).numpy()).all()  # Check assumptions
    Uhat = pls.polar_decomposition(A, B)
    assert np.isclose(A.numpy(), tf.matmul(Uhat, B).numpy(), atol=1e-5).all()  # Check result
    A_align, B_align = pls.align_transforms_modulo_orthonormal(A, B)
    assert np.isclose(A_align.numpy(), B_align.numpy(), atol=1e-5).all()
