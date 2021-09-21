.. currentmodule:: scatsol

API Reference
===============

The following section outlines the API of scatsol.


Solution PEC Cylinder 2D
--------------------------------

TM
~~~~~~~~~~~~~~~~~~

Incident field:

.. math::

    E_z = E_0 \exp{(-1j k x)}

    H_y = -\frac{E_0}{\eta} \exp{(-1j k x)}

Scattered field:

.. math::

    E_z = -E_0 \left(\frac{J_0(k a)}{H_0^{(2)}(k a)}H_0^{(2)}(k\rho)+2 \sum_{n=1}^{\infty}(-j)^n\frac{J_n(k a)}{H_n^{(2)}(k a)}H_n^{(2)}(k \rho)\cos(n\phi)\right )

.. automodule:: scatsol.pec_cylinder_2d
   :members:
   :undoc-members:
   :show-inheritance:

TE
~~~~~~~~~~~~~~~~~~

Utils
--------------------

.. automodule:: scatsol.utils
   :members:
   :undoc-members:
   :show-inheritance:

Constants
----------


.. automodule:: scatsol.constant
   :members:
   :undoc-members:
   :show-inheritance:
