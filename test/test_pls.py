import pytest
import numpy as np
import scipy.stats as st
import tensorflow as tf
import tensorflow_probability as tfp
from bluemyrtle import pls
la = tf.linalg


@pytest.fixture
def model():
    return pls.PLS.random_init(dx=11, dy=5, dzs=5, dzx=13)


def test_zs_alignment(model):
    U = tf.constant(st.ortho_group.rvs(model.dzs), dtype=model.Wy.dtype)
    Wx_other = tf.matmul(model.Wx, U)
    Wy_other = tf.matmul(model.Wy, U)
    Us = model.zs_alignment(Wy_other, Wx_other)
    print(model.Wy - tf.matmul(Wy_other, Us))
    assert np.isclose(model.Wy.numpy(), tf.matmul(Wy_other, Us).numpy(), atol=1e-6).all()
    assert np.isclose(model.Wx.numpy(), tf.matmul(Wx_other, Us).numpy(), atol=1e-6).all()
    assert np.isclose(tf.transpose(U).numpy(), Us.numpy(), atol=1e-6).all()


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
