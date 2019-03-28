XDesign
#######

.. image:: https://readthedocs.org/projects/xdesign/badge/?version=latest
   :target: http://xdesign.readthedocs.io/en/latest/?badge=latest
   :alt: Read the Docs

.. image:: https://travis-ci.org/tomography/xdesign.svg?branch=master
   :target: https://travis-ci.org/tomography/xdesign
   :alt: Travis CI

.. image:: https://coveralls.io/repos/github/tomography/xdesign/badge.svg?branch=master
   :target: https://coveralls.io/github/tomography/xdesign?branch=master
   :alt: Coveralls

.. image:: https://codeclimate.com/github/tomography/xdesign/badges/gpa.svg
   :target: https://codeclimate.com/github/tomography/xdesign
   :alt: Code Climate

**XDesign** is an open-source Python package for generating configurable
x-ray imaging `phantoms <https://en.wikipedia.org/wiki/Imaging_phantom>`_,
simulating `data acquisition <https://en.wikipedia.org/wiki/Data_acquisition>`_,
and benchmarking x-ray `tomographic image reconstruction
<https://en.wikipedia.org/wiki/Tomography>`_.


Goals
=====
* Assist faster development of new generation tomographic reconstruction methods
* Allow quantitative comparison of different reconstruction methods
* Create a framework for designing x-ray imaging experiments


Current Scope
=============
* Customizable 2D phantoms constructed from circles and convex polygons
* Quantitative reconstruction quality and probe coverage metrics
* Attenuation interactions with X-ray probes of uniform flux
* Use of analytic (exact) solutions for algorithms and computation


Contribute
==========
* Issue Tracker: https://github.com/tomography/xdesign/issues
* Documentation: https://github.com/tomography/xdesign/tree/master/docs
* Source Code: https://github.com/tomography/xdesign/tree/master/xdesign
* Tests: https://github.com/tomography/xdesign/tree/master/tests


License
=======
The project is licensed under the
`BSD-3 <https://github.com/tomography/xdesign/blob/master/LICENSE.txt>`_ license.


Install
=======
Since version 0.5, XDesign is available on the conda-forge channel. Install it
in the usual way:

.. code-block:: bash

  $ conda install xdesign -c conda-forge
