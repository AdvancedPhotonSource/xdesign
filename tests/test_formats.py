from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from xdesign.formats import *


def test_get_NIST_table_element():
    table, density = get_NIST_table('Helium')

    energy = np.array([1.00E-03, 1.50E-03, 2.00E-03, 3.00E-03, 4.00E-03, 5.00E-03,
    6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02,
    5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01,
    4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00,
    2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01,
    1.50E+01, 2.00E+01]) * 1000

    mass_attenuation = [6.084E+01, 1.676E+01, 6.863E+00, 2.007E+00, 9.329E-01,
    5.766E-01, 4.195E-01, 2.933E-01, 2.476E-01, 2.092E-01, 1.960E-01,
    1.838E-01, 1.763E-01, 1.703E-01, 1.651E-01, 1.562E-01, 1.486E-01,
    1.336E-01, 1.224E-01, 1.064E-01, 9.535E-02, 8.707E-02, 8.054E-02,
    7.076E-02, 6.362E-02, 5.688E-02, 5.173E-02, 4.422E-02, 3.503E-02,
    2.949E-02, 2.577E-02, 2.307E-02, 1.940E-02, 1.703E-02, 1.363E-02,
    1.183E-02]

    mass_energy_absorption = [6.045E+01, 1.638E+01, 6.503E+00, 1.681E+00,
    6.379E-01, 3.061E-01, 1.671E-01, 6.446E-02, 3.260E-02, 1.246E-02,
    9.410E-03, 1.003E-02, 1.190E-02, 1.375E-02, 1.544E-02, 1.826E-02,
    2.047E-02, 2.424E-02, 2.647E-02, 2.868E-02, 2.951E-02, 2.971E-02,
    2.959E-02, 2.890E-02, 2.797E-02, 2.674E-02, 2.555E-02, 2.343E-02,
    2.019E-02, 1.790E-02, 1.622E-02, 1.493E-02, 1.308E-02, 1.183E-02,
    9.948E-03, 8.914E-03]

    assert density == 1.663E-04
    assert_array_equal(table['energy'], energy)
    assert_array_equal(table['mass_attenuation'], mass_attenuation)
    assert_array_equal(table['mass_energy_absorption'], mass_energy_absorption)


def test_get_NIST_table_compound():
    table, density = get_NIST_table('Bakelite')

    energy = np.array([1.00E-03, 1.50E-03, 2.00E-03, 3.00E-03, 4.00E-03, 5.00E-03,
    6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02,
    5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01,
    4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00,
    2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01,
    1.50E+01, 2.00E+01]) * 1000

    mass_attenuation = [2.484E+03, 8.027E+02, 3.512E+02, 1.065E+02, 4.494E+01,
    2.288E+01, 1.315E+01, 5.521E+00, 2.860E+00, 9.552E-01, 5.089E-01,
    2.824E-01, 2.241E-01, 2.000E-01, 1.866E-01, 1.707E-01, 1.602E-01,
    1.424E-01, 1.300E-01, 1.127E-01, 1.009E-01, 9.210E-02, 8.516E-02,
    7.478E-02, 6.723E-02, 6.013E-02, 5.472E-02, 4.694E-02, 3.761E-02,
    3.215E-02, 2.854E-02, 2.599E-02, 2.264E-02, 2.055E-02, 1.775E-02,
    1.641E-02]

    mass_energy_absorption = [2.480E+03, 8.010E+02, 3.500E+02, 1.057E+02,
    4.433E+01, 2.237E+01, 1.272E+01, 5.162E+00, 2.545E+00, 6.961E-01,
    2.779E-01, 8.135E-02, 3.988E-02, 2.754E-02, 2.340E-02, 2.200E-02,
    2.292E-02, 2.594E-02, 2.809E-02, 3.033E-02, 3.118E-02, 3.138E-02,
    3.124E-02, 3.049E-02, 2.951E-02, 2.821E-02, 2.696E-02, 2.479E-02,
    2.163E-02, 1.951E-02, 1.802E-02, 1.692E-02, 1.543E-02, 1.447E-02,
    1.315E-02, 1.249E-02]

    assert density == 1.250E+00
    assert_array_equal(table['energy'], energy)
    assert_array_equal(table['mass_attenuation'], mass_attenuation)
    assert_array_equal(table['mass_energy_absorption'], mass_energy_absorption)


def test_get_NIST_table_misspelled():
    try:
        table, density = get_NIST_table('ZZZZ')
    except ValueError:
        return 0

    assert False
