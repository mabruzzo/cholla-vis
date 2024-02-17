# here we define functions and classes related to the initial conditions
import numpy as np
try:
    import scipy.special
    import scipy.integrate
except:
    print("Unable to import scipy")
import unyt

def SQUARE(x): return x*x

_MU = 0.6
_KBOLTZ = 1.380658e-16
_MP = 1.672622e-24
_GN = 4.49451e-18  # gravitational constant, kpc^3 / M_sun / kyr^2

def standard_logistic_function(x):
    return 0.5 + 0.5 * np.tanh(0.5*x)

def dSigma_dR_full(r_kpc, Sigma_0, R_g, Rgas_truncation_radius, alpha = 0.005):
    sech2 = lambda x: SQUARE(1.0/np.cosh(x))
    a = R_g
    b = Rgas_truncation_radius
    c = alpha
    return (
        (0.5 * np.exp(-r_kpc/a)*np.tanh(0.5*(r_kpc - b)/c))/a - 
        (0.25 * np.exp(-r_kpc/a)*sech2((0.5 * (r_kpc - b))/c))/c - 
        (0.5 * np.exp(-r_kpc/a))/a
    )

class SurfaceDensityProfile:
    
    # think of alpha like a sharpness scale.
    # -> at Rgas_truncation_radius - 7*alpha, the taper-factor is ~(1 - 1e-3)
    # -> at Rgas_truncation_radius + 7*alpha, the taper-factor is ~1e-3

    def __init__(self, Sigma_0, R_g, Rgas_truncation_radius, alpha = 0.005):
        self.Sigma_0 = Sigma_0
        self.R_g = R_g
        self.Rgas_truncation_radius = Rgas_truncation_radius
        self.alpha = alpha

    def __call__(self, r_kpc):
        # nominal sigma (before tapering)
        Sigma = self.Sigma_0 * np.exp(-r_kpc / self.R_g)

        R_c = self.Rgas_truncation_radius;
        taper_factor = 1.0 - standard_logistic_function((r_kpc - R_c)
                                                        / self.alpha)
        return Sigma*taper_factor
    
    def dSigma_dR_div_Sigma(self, r_kpc):
        # this is the reverse of the tapering function, it goes from 0 to 1
        # -> yes, the only difference is changing one plus to a minus
        # -> it is exactly the formula of the standard logistic function
        tmp = 0.5 + 0.5 * np.tanh(0.5 * (r_kpc - self.Rgas_truncation_radius)
                                  / self.alpha)

        # compute factor, where dSigma_dR == factor * self(r_kpc)
        factor = -1.0 * ( (1.0 / self.R_g) + (1.0/self.alpha) * tmp)
        return factor
    
    def dSigma_dR(self, r_kpc):
        return self.dSigma_dR_div_Sigma(r_kpc) * self(r_kpc)

def get_sigma_gas_disk(Mgas, R_g, Rgas_truncation_radius,
                       alpha = 0.005, force_unity_Sigma0 = False):
    Sigma0 = Mgas/ (2 * np.pi * SQUARE(R_g))
    Sigma0_arg = Sigma0
    if force_unity_Sigma0:
        Sigma0_arg = 1.0
    if alpha is None:
        alpha = 0.005

    prof = SurfaceDensityProfile(
        Sigma_0 = Sigma0_arg, R_g = R_g, alpha = alpha,
        Rgas_truncation_radius = Rgas_truncation_radius)
    return Sigma0, prof

def invx_times_ln_1px(x):
    # computes ln(1+x)/x
    
    # for x <= 4e-4, we use a 3rd order taylor expansion
    # -> this ensures that the |relative error| < 3e-13 for 0 <= x < 1e4
    # -> above 1e4, the error starts to grow from using ln(1+x)/x
    
    # honestly, the main reason for using a taylor expansion is getting the 
    # right answer at x = 0 (previously we did something a lot more crude)
    
    if np.ndim(x) == 0:
        if x <= 5e-4:
            return 1 + x*(-0.5 + x*(1/3 - x*0.25))
        return np.log((1+x)**(1/x))
    
    out = np.empty_like(x)
    w = (x <= 5e-4)
    out[w] = 1 + x[w]*(-0.5 + x[w]*(1/3 - x[w]*0.25))
    out[~w] = np.log((1+x[~w])**(1/x[~w]))
    return out

class NFWHalo:
    def __init__(self, mass_msun, scale_radius_kpc, c_vir):
        self.mass_msun = mass_msun
        self.scale_radius_kpc = scale_radius_kpc
        self.c_vir = c_vir
        
    @staticmethod
    def _log_func(y):
        return np.log(1 + y) - y / (1 + y)

    def potential(self, r_kpc, z_kpc):
        rsphere = np.sqrt(r_kpc**2 + z_kpc**2) # spherical radius
        x = rsphere / self.scale_radius_kpc
        C = _GN * self.mass_msun / (self.scale_radius_kpc * self._log_func(self.c_vir))
        return -C * invx_times_ln_1px(x);

    def r_accel(self, r_kpc, z_kpc):
        rsphere = np.sqrt(r_kpc**2 + z_kpc**2) # spherical radius
        x = rsphere / self.scale_radius_kpc
        r_comp = r_kpc / rsphere
        A = self._log_func(x)
        B = 1.0 / (rsphere * rsphere)
        C = _GN * self.mass_msun / self._log_func(self.c_vir)
        return -C * A * B * r_comp

    def vcirc2_km2Pers2(self, r_kpc, z_kpc):
        return unyt.unyt_array(-r_kpc * self.r_accel(r_kpc, z_kpc),
                               'kpc**2/kyr**2').to('km**2/s**2').ndview

class MiyamotoNagaiDisk:

    def __init__(self, mass_msun, scale_radius_kpc, scale_height_kpc):
        self.scale_radius_kpc = scale_radius_kpc
        self.scale_height_kpc = scale_height_kpc
        self.mass_msun = mass_msun

    def potential(self, r_kpc, z_kpc):
        denom = np.sqrt(
            SQUARE(r_kpc) +
            SQUARE(self.scale_radius_kpc + 
                   np.sqrt(SQUARE(z_kpc) +
                           SQUARE(self.scale_height_kpc))
                  )
        )
        return -self.mass_msun * _GN / denom
    
    def r_accel(self, r_kpc, z_kpc):
        r_kpc, z_kpc = np.broadcast_arrays(r_kpc, z_kpc)
        A = self.scale_radius_kpc + np.sqrt(self.scale_height_kpc**2 + z_kpc**2)
        B = np.power(A * A + r_kpc * r_kpc, 1.5)

        return -_GN * self.mass_msun * r_kpc / B
    
    def vcirc2_km2Pers2(self, r_kpc, z_kpc):
        return unyt.unyt_array(-r_kpc * self.r_accel(r_kpc, z_kpc),
                               'kpc**2/kyr**2').to('km**2/s**2').ndview


class RazorThinDisk:
    # there's a chance that this is buggy!
    def __init__(self, mass_msun, scale_radius_kpc):
        self.sigma0 = mass_msun/ (2* np.pi * scale_radius_kpc**2)
        self.scale_radius_kpc = scale_radius_kpc

    def vcirc2_km2Pers2(self, r_kpc, z_kpc):
        # eqn 2.165 from Binney & Tremaine
        assert np.all(0.0 == z_kpc)

        y = (r_kpc / self.scale_radius_kpc)
        bessel_term = (scipy.special.i0(y) * scipy.special.kn(0,y) -
                       scipy.special.i1(y) * scipy.special.kn(1,y))

        tmp = (4 * np.pi * _GN * self.sigma0 * self.scale_radius_kpc *
               SQUARE(y) * bessel_term)
        
        return unyt.unyt_array(tmp, 'kpc**2/kyr**2').to('km**2/s**2').ndview

def get_defaults():
    mvir = 1.077e12
    mstellar_disk = 6.5e10
    return dict(
        halo = NFWHalo(mvir - mstellar_disk, 261/18, 18),
        stellar_disk = MiyamotoNagaiDisk(mstellar_disk, 2.5, 0.7),
        razorthin_gas_disk = RazorThinDisk(0.15 * 6.5e10, 3.5),
        starparticle_disk = RazorThinDisk(2e9, 2.5)
    )


def vcirc2_gas_builder(Mgas = 0.15 * 6.5e10,
                       R_g = 3.5,
                       Rgas_truncation_radius = 9.9,
                       alpha = 0.005):
    """
    This is an alternative to RazorThinDisk, that we know will work...
    """
    from galpy.potential import (
        MiyamotoNagaiPotential, MN3ExponentialDiskPotential,
        RazorThinExponentialDiskPotential,
        DoubleExponentialDiskPotential,
        AnyAxisymmetricRazorThinDiskPotential
    )
    from astropy import units

    nominal_gas_Sigma0, prof = get_sigma_gas_disk(
        Mgas = Mgas, R_g = R_g, Rgas_truncation_radius = Rgas_truncation_radius,
        alpha = alpha, force_unity_Sigma0 = True)

    ref_with_units = RazorThinExponentialDiskPotential(
        amp = nominal_gas_Sigma0 * units.Msun/units.kpc**2,
        hr = R_g*units.kpc)

    if Rgas_truncation_radius == 0.0:
        def vcirc2_km2pers2(r_kpc, z_kpc):
            assert np.all(z_kpc == 0.0)
            return_scalar = False
            if np.ndim(r_kpc) == 0:
                return_scalar = True
                r_kpc = [r_kpc]
            out = [ref_with_units.vcirc(r * units.kpc).to('km/s').value
                   for r in r_kpc]
            out = SQUARE(np.array(out))
            if return_scalar:
                return out[0]
            return out
    else:
        ref_no_units = RazorThinExponentialDiskPotential(hr = R_g)
        actual_no_units = AnyAxisymmetricRazorThinDiskPotential(surfdens=prof)
        
        def vcirc2_km2pers2(r_kpc, z_kpc):
            assert np.all(z_kpc == 0.0)
            return_scalar = False
            if np.ndim(r_kpc) == 0:
                return_scalar = True
                r_kpc = [r_kpc]
            out = []
            for r in r_kpc:
                tmp = actual_no_units.vcirc(r) / ref_no_units.vcirc(r)
                out.append(
                    (tmp*ref_with_units.vcirc(r * units.kpc)).to('km/s').value
                )
            out = SQUARE(np.array(out))
            if return_scalar:
                return out[0]
            return out
    return vcirc2_km2pers2

def calc_rho_midplane(R, cs_isothermal_kpc_per_kyr, static_components,
                      Sigma_gas_fn = None):
    """
    This computes the midplane mass-density assuming that there is no
    self-gravity.

    The gas disk's mass density can be written as 
        rho(R,z) = zeta(R,z) * Sigma(R)

    When `Sigma_gas_fn` is `None`, this effectively returns zeta(R,0)
    """

    scalar = (np.ndim(R) == 0)
    if scalar: R = [R]

    out = []
    cs2 = SQUARE(cs_isothermal_kpc_per_kyr)
    eval_phi = lambda R, z: sum(comp.potential(R,z) for comp in static_components)
    fn = lambda z, cur_R, phi_0: np.exp(-(eval_phi(cur_R,z) - phi_0) / cs2)
    if Sigma_gas_fn is None:
        Sigma_gas_fn = lambda R: 1.0
    for cur_R in R:
        phi_midplane = eval_phi(cur_R, 0.0)
        # integrate fn from z = 0 to z = inf
        integral, abserr = scipy.integrate.quad(fn,a = 0, b = np.inf,
                                                args = (cur_R, phi_midplane))

        rho_midplane = Sigma_gas_fn(cur_R)/ (2* integral)
        out.append(rho_midplane)
    if scalar:
        return out[0]
    return np.array(out)

def plot_vcirc_contributions(ax, comp_label_pairs = None,
                             static_label_pairs = None,
                             show_components = True, show_squared = False,
                             R_vals = None, total_label = 'total'):

    if R_vals is None:
        R_vals = np.linspace(0.0, 11, num = 1101)
        R_vals[0] = 1e-5
    
    vsq_total = np.zeros_like(R_vals)

    if static_label_pairs is None:
        dflts = get_defaults()
        pairs = [(dflts['stellar_disk'].vcirc2_km2Pers2, 'stellar-potential'),
                 (dflts['halo'].vcirc2_km2Pers2, 'halo-potential')
                ]
    else:
        pairs = [e for e in static_label_pairs]

    if comp_label_pairs is not None:
        pairs = pairs + comp_label_pairs

    if show_squared:
        fmt_y = lambda y: y
        ax.set_ylabel(r'$v_{\rm circ}^2\ ({\rm km}^2\ {\rm s}^{-2})$')
    else:
        fmt_y = lambda y: np.sqrt(y)
        ax.set_ylabel(r'$v_{\rm circ}\ ({\rm km}\ {\rm s}^{-1})$')

    for obj, label in pairs:
        contrib = obj(R_vals, 0.0)

        ls = '-'
        if np.any(contrib < 0).any():
            ls = '--'
            assert np.all(contrib <= 0)

        vsq_total += contrib
        if show_components:
            ax.plot(R_vals, fmt_y(np.abs(contrib)), label = label, ls = ls)

    ax.plot(R_vals, fmt_y(vsq_total), label = total_label)

def calc_vcirc2_presure_term(R_vals,cs_isothermal_kpc_per_kyr,
                             static_components,
                             Sigma_gas_fn, Rderiv_step_kpc = 1e-3,
                             zeta_func = None,):
    """
    that computes the contribution of pressure in the circular velocity
    calculation.

    Specifically, this returns `(R/rho) * (dP/dR)` in units of (km/s)^2. For a
    radially exponential disk, the values will be negative.
    """
    assert (R_vals[1] - 2 * Rderiv_step_kpc) >= 0.0

    # let's image that rho = zeta * Sigma(R)
    # rho_midplane = zeta_midplane(R) * Sigma(R)
    #  -> introduce abreviation
    # rhoM = zetaM(R) * Sigma(R)
    # (d/dR) rho_midplane = Sigma(R) * [ dzetaM/dR + 
    #                                    (zetaM/Sigma_gas_fn) * dSigma/dR]

    # currently, assume that vertical structure doesn't depend on self-grav
    # -> Thus, we can solve for dzetaM/dR
    kw = dict(cs_isothermal_kpc_per_kyr = cs_isothermal_kpc_per_kyr,
              static_components = static_components,
              Sigma_gas_fn = None)

    centered_zetaM_vals = calc_rho_midplane(R_vals, **kw)
    leftmost = calc_rho_midplane(R_vals - 2*Rderiv_step_kpc, **kw)
    left_zetaM_vals = calc_rho_midplane(R_vals - Rderiv_step_kpc, **kw)
    right_zetaM_vals = calc_rho_midplane(R_vals + Rderiv_step_kpc, **kw)
    rightmost = calc_rho_midplane(R_vals + 2*Rderiv_step_kpc, **kw)

    dzetaM_div_dR = ((-0.25 * leftmost - 2*left_zetaM_vals + 
                     2 * right_zetaM_vals + 0.25 * rightmost)/
                     (12*Rderiv_step_kpc))
    # overwrite the value at R = 0, use a (second-order) forward difference
    dzetaM_div_dR[0] = (-1.5 * centered_zetaM_vals[0] + 2 * right_zetaM_vals[0] 
                        - 0.5 * rightmost[0]) / Rderiv_step_kpc

    # (drho_midplane/dR) / Sigma(R) = [ dzetaM/dR +
    #                                   (zetaM/Sigma_gas_fn) * dSigma/dR ]
    drhoM_dR_div_Sigma = (
        dzetaM_div_dR +
        centered_zetaM_vals * Sigma_gas_fn.dSigma_dR_div_Sigma(R_vals))
    drhoM_dR_div_rhoM = drhoM_dR_div_Sigma / centered_zetaM_vals
    dP_dR_div_rhoM = SQUARE(cs_isothermal_kpc_per_kyr) * drhoM_dR_div_rhoM
    return unyt.unyt_array(dP_dR_div_rhoM, '(kpc/kyr)**2').to('km**2/s**2')


class HaloGasGenerator:
    
    def __init__(self, gamma, rcool, T_0h, rho_0h, phi_fn):
        """
        The gas is normalized at the cooling radius, rcool
        -> we provide T_0h and rho_0h at that location
        """
        self.T_0h = T_0h
        self.rho_0h = rho_0h
        self.gamma = gamma
        self.rcool = rcool
        
        # I really think the following should be an adiabatic sound
        # speed, rather than an isothermal sound-speed
        c_sh = np.sqrt(_KBOLTZ * T_0h / (_MU * _MP))
        self.c_sh = unyt.unyt_quantity(c_sh, 'cm/s').to('kpc/kyr').ndview
        self.phi_fn = phi_fn
        self.phi_0 = phi_fn(rcool)
        print(self.phi_0)

    def calc_rho(self, r_kpc, cgols_style = True):
        phi_fn = self.phi_fn
        gm1 = self.gamma - 1.0
        rho_rcool = self.rho_0h
        phi_rcool = self.phi_0
        
        c_sh2 = np.square(self.c_sh)

        if cgols_style:
            delta_phi = phi_fn(r_kpc) - phi_rcool

            # typo in the CGOLS paper used a '+', rather than a '-'
            return rho_rcool * np.power(1.0 - gm1 * (phi_fn(r_kpc) - phi_rcool)
                                        / c_sh2,
                                        1/gm1)

        else:
            rho_rcool = self.rho_0h
            phi_rcool = self.phi_0
            # compute Phi at r= 0
            phi_center = phi_fn(0.0)

            tmp_factor = 1.0 - gm1 * (phi_rcool - phi_center)/c_sh2
            print(phi_rcool, phi_center, phi_rcool - phi_center, tmp_factor)
            rho_center = (
                rho_rcool * np.power(tmp_factor, 1/gm1)
            )
            print(rho_center)

            return rho_center * np.power(1.0 - gm1 * (phi_fn(r_kpc) -
                                                      phi_center)/c_sh2,
                                         1/gm1)

    def calc_p_from_rho(self, rho_vals):
        K = (np.square(self.c_sh) * np.power(self.rho_0h, 1.0 - self.gamma)
             / self.gamma)
        return K * rho_vals**self.gamma
