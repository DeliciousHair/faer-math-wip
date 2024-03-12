use faer::{mat, prelude::*, ComplexField, Entity};

trait ScalarOps: Entity {
    type Complex: ComplexField;

    fn faer_powi(self, rhs: i32) -> Self;
    fn faer_exp(self) -> Self;
    fn faer_cos(self) -> Self;
    fn faer_sin(self) -> Self;
    fn faer_cosh(self) -> Self;
    fn faer_sinh(self) -> Self;

    fn compute_two_d(a00: Self, a01: Self, a10: Self, a11: Self) -> (Self, Self, Self, Self);

    fn pade_theta() -> (Self, Self, Self, Self, Self);
    fn pade_coeff() -> (Self, Self, Self, Self, Self);
}

impl ScalarOps for c32 {
    type Complex = Self;

    #[inline(always)]
    fn faer_powi(self, rhs: i32) -> Self {
        Self::powi(self, rhs)
    }

    #[inline(always)]
    fn faer_exp(self) -> Self {
        Self::exp(self)
    }

    #[inline(always)]
    fn faer_cos(self) -> Self {
        Self::cos(self)
    }

    #[inline(always)]
    fn faer_sin(self) -> Self {
        Self::sin(self)
    }

    #[inline(always)]
    fn faer_cosh(self) -> Self {
        Self::cosh(self)
    }

    #[inline(always)]
    fn faer_sinh(self) -> Self {
        Self::sinh(self)
    }

    #[inline(always)]
    fn compute_two_d(a00: Self, a01: Self, a10: Self, a11: Self) -> (Self, Self, Self, Self) {
        let mu = ((a00.faer_sub(a11))
            .faer_powi(2)
            .faer_add(Self::faer_from_f64(4f64).faer_mul(a01).faer_mul(a10)))
        .faer_sqrt()
        .faer_mul(Self::faer_from_f64(0.5));

        let eap = ((a00.faer_add(a11)).faer_mul(Self::faer_from_f64(0.5))).faer_exp();
        let am = (a00.faer_sub(a11)).faer_mul(Self::faer_from_f64(0.5));
        let cosh_mu = mu.faer_cosh();
        let sinch_mu = match mu.eq(&Self::faer_zero()) {
            true => Self::faer_one(),
            false => mu.faer_sinh().faer_mul(mu.faer_inv()),
        };

        (
            eap.faer_mul(cosh_mu.faer_add(am.faer_mul(sinch_mu))),
            eap.faer_mul(a01).faer_mul(sinch_mu),
            eap.faer_mul(a10).faer_mul(sinch_mu),
            eap.faer_mul(cosh_mu.faer_sub(am.faer_mul(sinch_mu))),
        )
    }

    #[inline(always)]
    fn pade_theta() -> (Self, Self, Self, Self, Self) {
        (
            Self::faer_from_f64(1.495_585_217_958_292e-2),
            Self::faer_from_f64(2.539_398_330_063_23e-1),
            Self::faer_from_f64(9.504_178_996_162_932e-1),
            Self::faer_from_f64(2.097_847_961_257_068),
            Self::faer_from_f64(4.25),
        )
    }

    #[inline(always)]
    fn pade_coeff() -> (Self, Self, Self, Self, Self) {
        let u: Self = Self::faer_from_f64(2f64).faer_powi(24);
        (
            u.faer_mul(Self::faer_from_f64(100_800f64)),
            u.faer_mul(Self::faer_from_f64(10_059_033_600f64)),
            u.faer_mul(Self::faer_from_f64(4_487_938_430_976_000f64)),
            u.faer_mul(Self::faer_from_f64(5_914_384_781_877_411_840_000f64)),
            u.faer_mul(Self::faer_from_f64(
                113_250_775_606_021_113_483_283_660_800_000_000f64,
            )),
        )
    }
}

impl ScalarOps for c64 {
    type Complex = Self;

    #[inline(always)]
    fn faer_powi(self, rhs: i32) -> Self {
        Self::powi(self, rhs)
    }

    #[inline(always)]
    fn faer_exp(self) -> Self {
        Self::exp(self)
    }

    #[inline(always)]
    fn faer_cos(self) -> Self {
        Self::cos(self)
    }

    #[inline(always)]
    fn faer_sin(self) -> Self {
        Self::sin(self)
    }

    #[inline(always)]
    fn faer_cosh(self) -> Self {
        Self::cosh(self)
    }

    #[inline(always)]
    fn faer_sinh(self) -> Self {
        Self::sinh(self)
    }

    #[inline(always)]
    fn compute_two_d(a00: Self, a01: Self, a10: Self, a11: Self) -> (Self, Self, Self, Self) {
        let mu = ((a00.faer_sub(a11))
            .faer_powi(2)
            .faer_add(Self::faer_from_f64(4f64).faer_mul(a01).faer_mul(a10)))
        .faer_sqrt()
        .faer_mul(Self::faer_from_f64(0.5));

        let eap = ((a00.faer_add(a11)).faer_mul(Self::faer_from_f64(0.5))).faer_exp();
        let am = (a00.faer_sub(a11)).faer_mul(Self::faer_from_f64(0.5));
        let cosh_mu = mu.faer_cosh();
        let sinch_mu = match mu.eq(&Self::faer_zero()) {
            true => Self::faer_one(),
            false => mu.faer_sinh().faer_mul(mu.faer_inv()),
        };

        (
            eap.faer_mul(cosh_mu.faer_add(am.faer_mul(sinch_mu))),
            eap.faer_mul(a01).faer_mul(sinch_mu),
            eap.faer_mul(a10).faer_mul(sinch_mu),
            eap.faer_mul(cosh_mu.faer_sub(am.faer_mul(sinch_mu))),
        )
    }

    #[inline(always)]
    fn pade_theta() -> (Self, Self, Self, Self, Self) {
        (
            Self::faer_from_f64(1.495_585_217_958_292e-2),
            Self::faer_from_f64(2.539_398_330_063_23e-1),
            Self::faer_from_f64(9.504_178_996_162_932e-1),
            Self::faer_from_f64(2.097_847_961_257_068),
            Self::faer_from_f64(4.25),
        )
    }

    #[inline(always)]
    fn pade_coeff() -> (Self, Self, Self, Self, Self) {
        let u: Self = Self::faer_from_f64(2f64).faer_powi(53);
        (
            u.faer_mul(Self::faer_from_f64(100_800f64)),
            u.faer_mul(Self::faer_from_f64(10_059_033_600f64)),
            u.faer_mul(Self::faer_from_f64(4_487_938_430_976_000f64)),
            u.faer_mul(Self::faer_from_f64(5_914_384_781_877_411_840_000f64)),
            u.faer_mul(Self::faer_from_f64(
                113_250_775_606_021_113_483_283_660_800_000_000f64,
            )),
        )
    }
}

impl ScalarOps for f32 {
    type Complex = Self;

    #[inline(always)]
    fn faer_powi(self, rhs: i32) -> Self {
        Self::powi(self, rhs)
    }

    #[inline(always)]
    fn faer_exp(self) -> Self {
        Self::exp(self)
    }

    #[inline(always)]
    fn faer_cos(self) -> Self {
        Self::cos(self)
    }

    #[inline(always)]
    fn faer_sin(self) -> Self {
        Self::sin(self)
    }

    #[inline(always)]
    fn faer_cosh(self) -> Self {
        Self::cosh(self)
    }

    #[inline(always)]
    fn faer_sinh(self) -> Self {
        Self::sinh(self)
    }

    #[inline(always)]
    fn compute_two_d(a00: Self, a01: Self, a10: Self, a11: Self) -> (Self, Self, Self, Self) {
        let tmp = ((a00.faer_sub(a11))
            .faer_powi(2)
            .faer_add(Self::faer_from_f64(4f64).faer_mul(a01).faer_mul(a10)))
        .faer_mul(Self::faer_from_f64(0.25));

        match tmp.gt(&Self::faer_zero()) {
            true => {
                let mu = tmp.faer_sqrt();
                let eap = ((a00.faer_add(a11)).faer_mul(Self::faer_from_f64(0.5))).faer_exp();
                let am = (a00.faer_sub(a11)).faer_mul(Self::faer_from_f64(0.5));
                let cosh_mu = mu.faer_cosh();
                let sinch_mu = match mu.eq(&Self::faer_zero()) {
                    true => Self::faer_one(),
                    false => mu.faer_sinh().faer_mul(mu.faer_inv()),
                };
                (
                    eap.faer_mul(cosh_mu.faer_add(am.faer_mul(sinch_mu))),
                    eap.faer_mul(a01).faer_mul(sinch_mu),
                    eap.faer_mul(a10).faer_mul(sinch_mu),
                    eap.faer_mul(cosh_mu.faer_sub(am.faer_mul(sinch_mu))),
                )
            }
            false => {
                let mu = tmp.faer_neg().faer_sqrt();
                let eap = ((a00.faer_add(a11)).faer_mul(Self::faer_from_f64(0.5))).faer_exp();
                let am = (a00.faer_sub(a11)).faer_mul(Self::faer_from_f64(0.5));
                let cosh_mu = mu.faer_cos();
                let sinch_mu = match mu.eq(&Self::faer_zero()) {
                    true => Self::faer_one(),
                    false => mu.faer_sin().faer_mul(mu.faer_inv()),
                };
                (
                    eap.faer_mul(cosh_mu.faer_add(am.faer_mul(sinch_mu))),
                    eap.faer_mul(a01).faer_mul(sinch_mu),
                    eap.faer_mul(a10).faer_mul(sinch_mu),
                    eap.faer_mul(cosh_mu.faer_sub(am.faer_mul(sinch_mu))),
                )
            }
        }
    }

    #[inline(always)]
    fn pade_theta() -> (Self, Self, Self, Self, Self) {
        (
            Self::faer_from_f64(1.495_585_217_958_292e-2),
            Self::faer_from_f64(2.539_398_330_063_23e-1),
            Self::faer_from_f64(9.504_178_996_162_932e-1),
            Self::faer_from_f64(2.097_847_961_257_068),
            Self::faer_from_f64(4.25),
        )
    }

    #[inline(always)]
    fn pade_coeff() -> (Self, Self, Self, Self, Self) {
        let u: Self = Self::faer_from_f64(2f64).faer_powi(24);
        (
            u.faer_mul(Self::faer_from_f64(100_800f64)),
            u.faer_mul(Self::faer_from_f64(10_059_033_600f64)),
            u.faer_mul(Self::faer_from_f64(4_487_938_430_976_000f64)),
            u.faer_mul(Self::faer_from_f64(5_914_384_781_877_411_840_000f64)),
            u.faer_mul(Self::faer_from_f64(
                113_250_775_606_021_113_483_283_660_800_000_000f64,
            )),
        )
    }
}

impl ScalarOps for f64 {
    type Complex = Self;

    #[inline(always)]
    fn faer_powi(self, rhs: i32) -> Self {
        Self::powi(self, rhs)
    }

    #[inline(always)]
    fn faer_exp(self) -> Self {
        Self::exp(self)
    }

    #[inline(always)]
    fn faer_cos(self) -> Self {
        Self::cos(self)
    }

    #[inline(always)]
    fn faer_sin(self) -> Self {
        Self::sin(self)
    }

    #[inline(always)]
    fn faer_cosh(self) -> Self {
        Self::cosh(self)
    }

    #[inline(always)]
    fn faer_sinh(self) -> Self {
        Self::sinh(self)
    }

    #[inline(always)]
    fn compute_two_d(a00: Self, a01: Self, a10: Self, a11: Self) -> (Self, Self, Self, Self) {
        let tmp = ((a00.faer_sub(a11))
            .faer_powi(2)
            .faer_add(Self::faer_from_f64(4f64).faer_mul(a01).faer_mul(a10)))
        .faer_mul(Self::faer_from_f64(0.25));

        match tmp.gt(&Self::faer_zero()) {
            true => {
                let mu = tmp.faer_sqrt();
                let eap = ((a00.faer_add(a11)).faer_mul(Self::faer_from_f64(0.5))).faer_exp();
                let am = (a00.faer_sub(a11)).faer_mul(Self::faer_from_f64(0.5));
                let cosh_mu = mu.faer_cosh();
                let sinch_mu = match mu.eq(&Self::faer_zero()) {
                    true => Self::faer_one(),
                    false => mu.faer_sinh().faer_mul(mu.faer_inv()),
                };
                (
                    eap.faer_mul(cosh_mu.faer_add(am.faer_mul(sinch_mu))),
                    eap.faer_mul(a01).faer_mul(sinch_mu),
                    eap.faer_mul(a10).faer_mul(sinch_mu),
                    eap.faer_mul(cosh_mu.faer_sub(am.faer_mul(sinch_mu))),
                )
            }
            false => {
                let mu = tmp.faer_neg().faer_sqrt();
                let eap = ((a00.faer_add(a11)).faer_mul(Self::faer_from_f64(0.5))).faer_exp();
                let am = (a00.faer_sub(a11)).faer_mul(Self::faer_from_f64(0.5));
                let cosh_mu = mu.faer_cos();
                let sinch_mu = match mu.eq(&Self::faer_zero()) {
                    true => Self::faer_one(),
                    false => mu.faer_sin().faer_mul(mu.faer_inv()),
                };
                (
                    eap.faer_mul(cosh_mu.faer_add(am.faer_mul(sinch_mu))),
                    eap.faer_mul(a01).faer_mul(sinch_mu),
                    eap.faer_mul(a10).faer_mul(sinch_mu),
                    eap.faer_mul(cosh_mu.faer_sub(am.faer_mul(sinch_mu))),
                )
            }
        }
    }

    #[inline(always)]
    fn pade_theta() -> (Self, Self, Self, Self, Self) {
        (
            Self::faer_from_f64(1.495_585_217_958_292e-2),
            Self::faer_from_f64(2.539_398_330_063_23e-1),
            Self::faer_from_f64(9.504_178_996_162_932e-1),
            Self::faer_from_f64(2.097_847_961_257_068),
            Self::faer_from_f64(4.25),
        )
    }

    #[inline(always)]
    fn pade_coeff() -> (Self, Self, Self, Self, Self) {
        let u: Self = Self::faer_from_f64(2f64).faer_powi(53);
        (
            u.faer_mul(Self::faer_from_f64(100_800f64)),
            u.faer_mul(Self::faer_from_f64(10_059_033_600f64)),
            u.faer_mul(Self::faer_from_f64(4_487_938_430_976_000f64)),
            u.faer_mul(Self::faer_from_f64(5_914_384_781_877_411_840_000f64)),
            u.faer_mul(Self::faer_from_f64(
                113_250_775_606_021_113_483_283_660_800_000_000f64,
            )),
        )
    }
}

trait MathOps: ComplexField + ScalarOps {}
impl<E: ComplexField + ScalarOps> MathOps for E {}

#[allow(private_bounds)]
pub trait MatrixFunctions<E: MathOps> {
    /// Computes `exp(A)`` for matrix `A`. This is the matrix-valued function
    ///
    /// $$\text{exp}(A) = \sum_0^\infty \frac{A^k}{k!}$$
    ///
    /// as opposed to exponentiating the elements of `A`.
    fn expm(&self) -> Mat<E>;
    fn lower_bandwidth(&self) -> usize;
    fn upper_bandwidth(&self) -> usize;

    // #[doc(hidden)]
    // fn __pick_pade_structure(&self) -> usize;
}

impl<E: MathOps> MatrixFunctions<E> for MatRef<'_, E> {
    #[inline(always)]
    #[track_caller]
    fn expm(&self) -> Mat<E> {
        assert!(self.nrows().eq(&self.ncols()));

        match self.nrows() {
            0 => Mat::new(),
            1 => {
                let a: E = unsafe { self.read_unchecked(0, 0) };
                Mat::<E>::from_fn(1, 1, |_, _| a.faer_exp())
            }
            2 => {
                let out = unsafe {
                    E::compute_two_d(
                        self.read_unchecked(0, 0),
                        self.read_unchecked(0, 1),
                        self.read_unchecked(1, 0),
                        self.read_unchecked(1, 1),
                    )
                };
                mat![[out.0, out.1], [out.2, out.3],]
            }
            _ => todo!(),
        }
    }

    /// copied shamelessly from `scipy.linalg.bandwidth` but converted to
    /// column-major iterations which should be a little more efficient here.
    fn lower_bandwidth(&self) -> usize {
        assert!(self.nrows().eq(&self.ncols()));
        let n: usize = self.nrows();

        let mut lower_bw: usize = 0;
        let mut tmp: usize;

        for col in 0..(n - 1) {
            tmp = [0, col - lower_bw].into_iter().max().unwrap();
            for row in (tmp + 1..n).rev() {
                unsafe {
                    if self.read_unchecked(row, col).ne(&E::faer_zero()) {
                        lower_bw = row - col;
                        break;
                    }
                }
            }
            if lower_bw.eq(&(n - 1)) {
                break;
            }
        }

        lower_bw
    }

    fn upper_bandwidth(&self) -> usize {
        assert!(self.nrows().eq(&self.ncols()));
        let n: usize = self.nrows();

        let mut upper_bw: usize = 0;
        let mut tmp: usize;

        for col in (1..n).rev() {
            tmp = [n - 1, col - upper_bw].into_iter().min().unwrap();
            for row in 0..tmp {
                unsafe {
                    if self.read_unchecked(row, col).ne(&E::faer_zero()) {
                        upper_bw = col - row;
                        break;
                    }
                }
            }
            if upper_bw.eq(&(n - 1)) {
                break;
            }
        }

        upper_bw
    }
}

#[cfg(test)]
mod tests {
    use std::{
        f32::consts::E as E32, f32::consts::PI as PI32, f32::EPSILON as EPSILON32,
        f64::consts::E as E64, f64::consts::PI as PI64, f64::EPSILON as EPSILON64,
    };

    use super::*;

    #[test]
    fn test_0d() {
        let arr_c32: Mat<c32> = Mat::new();
        let arr_c64: Mat<c64> = Mat::new();
        let arr_f32: Mat<f32> = Mat::new();
        let arr_f64: Mat<f64> = Mat::new();

        assert!(arr_c32.as_ref().expm() == Mat::<c32>::new());
        assert!(arr_c64.as_ref().expm() == Mat::<c64>::new());
        assert!(arr_f32.as_ref().expm() == Mat::<f32>::new());
        assert!(arr_f64.as_ref().expm() == Mat::<f64>::new());
    }

    #[test]
    fn test_1d() {
        let arr_c32: Mat<c32> = Mat::from_fn(1, 1, |_, _| c32::new(0f32, PI32));
        let arr_c64: Mat<c64> = Mat::from_fn(1, 1, |_, _| c64::new(0f64, PI64));
        let arr_f32: Mat<f32> = Mat::from_fn(1, 1, |_, _| 1f32);
        let arr_f64: Mat<f64> = Mat::from_fn(1, 1, |_, _| 1f64);

        let res_c32: Mat<c32> = Mat::from_fn(1, 1, |_, _| c32::new(-1f32, 0f32));
        let res_c64: Mat<c64> = Mat::from_fn(1, 1, |_, _| c64::new(-1f64, 0f64));
        let res_f32: Mat<f32> = Mat::from_fn(1, 1, |_, _| E32);
        let res_f64: Mat<f64> = Mat::from_fn(1, 1, |_, _| E64);

        assert!(
            (arr_c32.as_ref().expm() - res_c32.as_ref())
                .read(0, 0)
                .re
                .faer_abs()
                < EPSILON32
        );
        assert!(
            (arr_c32.as_ref().expm() - res_c32.as_ref())
                .read(0, 0)
                .im
                .faer_abs()
                < EPSILON32
        );
        assert!(
            (arr_c64.as_ref().expm() - res_c64.as_ref())
                .read(0, 0)
                .re
                .faer_abs()
                < EPSILON64
        );
        assert!(
            (arr_c64.as_ref().expm() - res_c64.as_ref())
                .read(0, 0)
                .im
                .faer_abs()
                < EPSILON64
        );
        assert!(
            (arr_f32.as_ref().expm() - res_f32.as_ref())
                .read(0, 0)
                .faer_abs()
                < EPSILON32
        );
        assert!(
            (arr_f64.as_ref().expm() - res_f64.as_ref())
                .read(0, 0)
                .faer_abs()
                < EPSILON64
        );
    }

    #[test]
    /// The 2d case uses the square root of `(a00 - a11) ** 2 + 4 * a01 * a10`,
    /// so we need to test that values that would lead to casting to complex are
    /// handled accurately as no casting is used.
    fn test_2d_00() {
        let arr: Mat<f32> = mat![[0f32, PI32], [-PI32, 0f32],];
        let res: Mat<f32> = mat![[-1f32, 0f32], [0f32, -1f32],];
        let out: Mat<f32> = arr.as_ref().expm();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out.read(i, j) - res.read(i, j)).faer_abs() < EPSILON32);
            }
        }

        let arr: Mat<f64> = mat![[0f64, PI64], [-PI64, 0f64],];
        let res: Mat<f64> = mat![[-1f64, 0f64], [0f64, -1f64],];
        let out: Mat<f64> = arr.as_ref().expm();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out.read(i, j) - res.read(i, j)).faer_abs() < EPSILON64);
            }
        }
    }

    #[test]
    /// happy-path, no surprises
    fn test_2d_01() {
        let arr: Mat<c32> = mat![
            [c32::new(-3f32, 2f32), c32::new(-0.2f32, 1f32)],
            [c32::new(0f32, 0f32), c32::new(-5f32, -1f32)],
        ];
        let res: Mat<c32> = mat![
            [
                c32::new(-0.020718731, 0.045271255),
                c32::new(-0.015060061, 0.005316357)
            ],
            [c32::new(0f32, 0f32), c32::new(0.0036405267, -0.0056697857)],
        ];
        let out: Mat<c32> = arr.as_ref().expm();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out.read(i, j) - res.read(i, j)).re.faer_abs() < EPSILON32);
                assert!((out.read(i, j) - res.read(i, j)).im.faer_abs() < EPSILON32);
            }
        }

        // c64
        let arr: Mat<c64> = mat![
            [c64::new(-3f64, 2f64), c64::new(-0.2f64, 1f64)],
            [c64::new(0f64, 0f64), c64::new(-5f64, -1f64)],
        ];
        let res: Mat<c64> = mat![
            [
                c64::new(-0.020718731002242877, 0.04527125315609296),
                c64::new(-0.015060059871132576, 0.005316356150066151)
            ],
            [
                c64::new(0f64, 0f64),
                c64::new(0.003640528300423191, -0.005669786896903857)
            ],
        ];
        let out: Mat<c64> = arr.as_ref().expm();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out.read(i, j) - res.read(i, j)).re.faer_abs() < EPSILON64);
                assert!((out.read(i, j) - res.read(i, j)).im.faer_abs() < EPSILON64);
            }
        }

        // f32
        let arr: Mat<f32> = mat![[4f32, 0.2f32], [0f32, 3f32],];
        let res: Mat<f32> = mat![[4f32.exp(), 6.902522], [0f32, 3f32.exp()],];
        let out: Mat<f32> = arr.as_ref().expm();

        for i in 0..2 {
            for j in 0..2 {
                assert!((out.read(i, j) - res.read(i, j)).faer_abs() < EPSILON32);
            }
        }

        // f64
        let arr: Mat<f64> = mat![[3f64, 0.2f64], [0f64, 5f64],];
        let res: Mat<f64> = mat![[3f64.exp(), 12.832762217938892], [0f64, 5f64.exp()],];
        let out: Mat<f64> = arr.as_ref().expm();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out.read(i, j) - res.read(i, j)).faer_abs() < EPSILON64);
            }
        }
    }
}
