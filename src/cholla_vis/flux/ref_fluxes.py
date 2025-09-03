"""This module defines machinery for computing reference fluxes
"""

from ..SNe_data import SNeData

import typing

import numpy as np
import unyt

if typing.TYPE_CHECKING:
    import sympy
    type unyt_dimension = sympy.core.symbol.Symbol
else:
    type unyt_dimension = typing.Any


def bicone_surface_area(
    spherical_radii: unyt.unyt_array,
    openning_angle_rad: float
) -> unyt.unyt_array:
    """Calculates the surface area of a bicone at each value of r_sph

    Parameters
    ----------
    r_sph
        The spherical radius from which we compute surface area, that
        we use to determine the fluxes.
    openning_angle_rad
        The openning angle of the bicone. This has units of radians (but
        should be passed in as a unitless `float`).

    Returns
    -------
    ``unyt.unyt_array``
        Specifies the surface area for each `r_sph`.
    """

    # 1. calculate solid angle subtended by a single cone
    #  -> Omega = 2*pi*(1 - cos(theta)),
    #     where theta is half of the openning angle
    #  -> https://en.wikipedia.org/wiki/Solid_angle#Cone,_spherical_cap,_hemisphere
    theta_rad = 0.5*openning_angle_rad
    solid_angle_single_cone = 2.0 * np.pi * (1.0 - np.cos(theta_rad))

    # 2. get solid angle of the bicone
    solid_angle = 2.0 * solid_angle_single_cone

    # 3. get surface area
    return spherical_radii * spherical_radii * solid_angle

def _has_dimensions(q: unyt.unyt_array, dim: unyt_dimension) -> bool:
    # check if a unyt_array has the appropriate unyt dimension
    if hasattr(q, "units"):
        return q.units.dimensions == dim
    return unyt.dimensionless == dim


class RefFluxCalculator:
    """
    Calculate "reference" fluxes expected from a supernova rate.

    In more detail, instances of this type compute fluxes from the provided
    supernova rate using the model defined in
        https://ui.adsabs.harvard.edu/abs/2020ApJ...900...61K/abstract
    We call these expected values the "reference fluxes". Instances can also
    be used to compute rate the globally averaged rate that a galactic wind is
    removing a quantity from a galaxy.

    Parameters
    ----------
    sn_dataset
        The supernova rate dataset
    use_lo
        Specifies whether we should use the lower estimate for the SNe rate
        or the higher estimate
    sn_e_inject
        An optional parameter to specify the amount of energy injected by a
        single supernova. The default value is 1e51 ergs.
    """

    def __init__(
        self,
        sn_dataset: SNeData,
        use_lo: bool,
        *,
        sn_e_inject: unyt.unyt_array | None = None
    ) -> typing.Self:

        # coerce sn_e_inject
        if sn_e_inject is None:
            sn_e_inject = unyt.unyt_quantity(1e51, 'erg')
        elif not _has_dimensions(sn_e_inject, unyt.dimensions.energy):
            raise ValueError("when specified, sn_e_inject have energy units")
        elif np.ndim(sn_e_inject) != 0:
            raise ValueError("when specified sn_e_inject must be a scalar")
        else:
            sn_e_inject = sn_e_inject.to("erg")

        # given below eqn 19
        v_cool = unyt.unyt_quantity(200.0, 'km/s')

        # equations: 16-18
        self._refquan = {
            # maybe we should pick a rounder number? like 100
            'density' : unyt.unyt_quantity(95.5, 'Msun'),
            # there is an alternative explanation suggested in the paper
            # (maybe consider that)
            'momentum' : 0.5 * sn_e_inject / v_cool,
            'energy' : sn_e_inject
        }

        assert sn_dataset.index.name == 't_kyr'
        self.t_Myr = sn_dataset.index.to_numpy() / 1000.0

        key = 'num_per_kyr_lo' if use_lo else 'num_per_kyr_hi'
        self._SN_rate = unyt.unyt_array(
            sn_dataset[key].to_numpy() /1000.0, units='year**-1'
        )


    def _ref_time_derive_it(self):
        for key, qref in self._refquan.items():
            yield (key, qref * self._SN_rate)

    def ref_time_deriv(self) -> dict[str, unyt.unyt_array]:
        """Returns a dictionary holding the globally averaged rate that a
        galactic wind is removing various quantities from a galaxy.

        Returns
        -------
        dict
            The keys hold the names of quantities and the associated values
            hold the galaxy-wide "reference" time derivatives of that quantity
            at each time given by `self.t_Myr`. In more detail:
            - `out["density"]` gives the rate at which mass is removed
            - `out["momentum"]` gives the rate at which momentum is removed.
               - this is a little confusing. You should think of this like the
                 component of the momentum moving with the outflow (e.g. think
                 of it like the momentum moving away from the disk)
               - if we were considering ordinary signed Cartesian components of
                 then this would be near zero.
            - `out["energy"]` gives the rate at which energy is removed
        """
        return dict(self._ref_time_derive_it())

    def _flux_helper(self, surface_area):
        assert isinstance(surface_area, np.ndarray)
        if surface_area.ndim == 0:
            surface_area = surface_area[None]
        # eqn 15:
        out = {
            key: t_deriv[None, :] / surface_area[:, None]
            for key, t_deriv in self._ref_time_derive_it()
        }
        return out

    # unyt.accepts raises an error if r_cyl doesn't have appropriate units
    @unyt.accepts(r_cyl=unyt.dimensions.length)
    def z_flux(self, r_cyl: unyt.unyt_array) -> dict[str, unyt.unyt_array]:
        """Returns the z fluxes (in various quantities) expected for the SNe
        rate if we assume a plane-parallel geometry.

        In more detail, imagine a cartesian coordinate system such that the
        disk lies at x=0 and y=0. Now imagine a cylinder of height, `h`, and
        radius, `r_cyl`, that is centered on (x=0,y=0,z=0). In this picture,
        the cylinder's circular faces are located at `h/2`, and `-h/2`. This
        function returns the fluxes through these cylindrical faces.

        Due to the assumption of a plane-parallel geometry, the value of `h`
        has no impact on the result.

        Parameters
        ----------
        r_cyl
            The cylindrical radius from which we compute surface area, that
            we use to determine the fluxes.

        Returns
        -------
        dict
            A dictionary, where each key is the name of a quantity. Each value
            holds the expected flux rate of the associated quantity computed
            at each specified cylindrical radius and at each time in
            `self.t_Myr`.
        """
        surface_area = np.pi * r_cyl**2
        # double surface area to include contribution from below
        surface_area *= 2
        return self._flux_helper(surface_area)

    # unyt.accept raises an error r_cyl have unexpected units
    @unyt.accepts(r_sph=unyt.dimensions.length, openning_angle_rad=unyt.dimensionless)
    def r_flux(
        self,
        spherical_radii: unyt.unyt_array,
        openning_angle_rad: float,
    ) -> dict[str, unyt.unyt_array]:
        """Returns the radial fluxes (in various quantities) expected for the
        SNe rate if we assume the outflow has a spherical geometry.

        In more detail, imagine a cartesian coordinate system such that the
        disk lies at x=0 and y=0. Now imagine a bicone, with radius `r_sph`,
        centered on (x=0,y=0,z=0), and openning angle `openning_angle_rad`.
        Furthermore, the bicone is defined such that `(x=0,y=0,z=r_sph)` and
        `(x=0,y=0,z=-r_sph)` always lies on the surface of the bicone.

        It's important to understand that the bicone doesn't have any faces
        parallel to the y-z plane. It has a "spherical cap."

        This function computes the radial flux through the bicone. Since the
        bicone is defined in terms of an openning angle. Consequently surface
        area depends on the spherical radius.

        Parameters
        ----------
        r_sph
            The spherical radius from which we compute surface area, that
            we use to determine the fluxes.
        openning_angle_rad
            The openning angle of the bicone. This has units of radians (but
            should be passed in as a unitless `float`).

        Returns
        -------
        dict
            A dictionary, where each key is the name of a quantity. Each value
            holds the expected radial flux of the associated quantity computed
            at for each `r_sph` and at each time in `self.t_Myr`.
        """
        surface_area = bicone_surface_area(r_sph, openning_angle_rad)
        return self._flux_helper(surface_area)

