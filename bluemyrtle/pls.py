import math
import tensorflow as tf
import tensorflow_probability as tfp
_std_normal = tfp.distributions.Normal(loc=0, scale=1.)
la = tf.linalg


def diff_transforms_modulo_orthonormal(A, B):
    """The difference between two transforms, A and B, modulo an arbitrary orthogonal transformation."""
    A, B = align_transforms_modulo_orthonormal(A, B)
    return B - A


def align_transforms_modulo_orthonormal(A, B):
    """Align two transforms, A and B, modulo an arbitrary orthogonal transformation."""
    return A, tf.matmul(polar_decomposition(A, B), B)


def polar_decomposition(A, B):
    """Find a unitary matrix that maps B as close to A as possible (in the
    least squares sense).

    Suppose a set of points B has been subjected to an unknown rotation and
    then jittered by white Gaussian noise to give a new set of points A. What
    is the most likely rotation? More generally, what unitary matrix minimizes
    f(U) = tr((A − UB)′(A − UB))?

    See Section 7 of Minka's [report on
    matrices](https://tminka.github.io/papers/matrix/minka-matrix.pdf).
    """
    s, V, W = la.svd(la.matmul(B, A, transpose_b=True))
    return la.matmul(W, V, transpose_b=True)


class PLS:

    def __init__(self, mu_y, mu_x, Wy, Wx, Bx, sigma):
        self.mu_y = mu_y
        self.mu_x = mu_x
        self.Wy = Wy
        self.Wx = Wx
        self.Bx = Bx
        self.sigma = sigma

        self.assert_dimensions_consistent()

    def assert_dimensions_consistent(self):
        """Check consistency of parameter dimensions."""
        assert self.mu_y.shape[-1] == self.dy
        assert self.mu_x.shape[-1] == self.dx
        assert self.Bx.shape[-2] == self.dx
        assert self.Bx.shape[-1] == self.dzx
        assert self.Wx.shape[-2] == self.dx
        assert self.Wx.shape[-1] == self.dzs
        assert self.Wy.shape[-2] == self.dy
        assert self.Wy.shape[-1] == self.dzs

    @staticmethod
    def random_init(dx, dy, dzs, dzx):
        return PLS(
            mu_y=_std_normal.sample([dy]),
            mu_x=_std_normal.sample([dx]),
            Wy=_std_normal.sample([dy, dzs]) / math.sqrt(dzs),
            Wx=_std_normal.sample([dx, dzs]) / math.sqrt(dzs) / math.sqrt(2),
            Bx=_std_normal.sample([dx, dzx]) / math.sqrt(dzx) / math.sqrt(2),
            sigma=tfp.distributions.LogNormal(loc=0, scale=1).sample())

    def zs_alignment(self, Wy_other, Wx_other):
        """Find an orthonormal matrix U, that rotates the shared z_s space such that
        Wx_other is as close to self.Wx and similarly with W_y.

        Wx_other @ U should be close to self.W
        """
        s, V, W = la.svd(la.matmul(Wy_other, self.Wy, transpose_a=True)
                         + la.matmul(Wx_other, self.Wx, transpose_a=True))
        return la.matmul(V, W, transpose_b=True)

    @property
    def parameters(self):
        """The parameters of the model."""
        return dict(
            # Means
            mu_y=self.mu_y,
            mu_x=self.mu_x,
            # Maps from latent spaces
            Wy=self.Wy,
            Wx=self.Wx,
            Bx=self.Bx,
            # Noise
            sigma=self.sigma)

    @property
    def dy(self):
        """Size of y."""
        return self.Wy.shape[-2]

    @property
    def dx(self):
        """Size of x."""
        return self.Wx.shape[-2]

    @property
    def dzs(self):
        """Size of shared latent space."""
        return self.Wy.shape[-1]

    @property
    def dzx(self):
        """Size of latent space dedicated to variation of x."""
        return self.Bx.shape[-1]

    def sample_z(self, N):
        zx = _std_normal.sample([N, self.dzx])
        zs = _std_normal.sample([N, self.dzs])
        return zs, zx

    def p_y_given_z(self, zs, zx):
        return tfp.distributions.MultivariateNormalDiag(
            loc=self.mu_y + la.matvec(self.Wy, zs),
            scale_identity_multiplier=self.sigma)

    def p_x_given_z(self, zs, zx):
        return tfp.distributions.MultivariateNormalDiag(
            loc=self.mu_x + la.matvec(self.Wx, zs) + la.matvec(self.Bx, zx),
            scale_identity_multiplier=self.sigma)

    def p_y_given_x(self, x):
        sigma2_Ix = self.sigma**2 * tf.eye(self.dx)
        sigma2_Iy = self.sigma**2 * tf.eye(self.dy)
        # Useful products
        BxBxT = tf.matmul(self.Bx, self.Bx, transpose_b=True)
        WxWxT = tf.matmul(self.Wx, self.Wx, transpose_b=True)
        WxWyT = tf.matmul(self.Wx, self.Wy, transpose_b=True)
        WyWyT = tf.matmul(self.Wy, self.Wy, transpose_b=True)
        # Calculate terms for posterior mean and covariance
        BxBxT_WxWxT_chol = la.cholesky(BxBxT + WxWxT + sigma2_Ix)
        WxWyT_term = la.triangular_solve(BxBxT_WxWxT_chol, WxWyT)
        x_mux_term = tf.squeeze(la.triangular_solve(BxBxT_WxWxT_chol,
                                                    tf.expand_dims(x - self.mu_x, axis=-1)),
                                axis=-1)
        # Create posterior
        likelihood_loc = self.mu_y + la.matvec(WxWyT_term, x_mux_term, transpose_a=True)
        likelihood_cov = WyWyT + sigma2_Iy - la.matmul(WxWyT_term, WxWyT_term, transpose_a=True)
        return tfp.distributions.MultivariateNormalTriL(loc=likelihood_loc,
                                                        scale_tril=la.cholesky(likelihood_cov))
