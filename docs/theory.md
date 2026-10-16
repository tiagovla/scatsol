# Equations and conventions

ScatSol solves time-harmonic Maxwell scattering by concentric cylindrical and
spherical regions. The implementation uses the phasor convention
$e^{+j\omega t}$, where $j^2=-1$. With this convention, outgoing cylindrical
and spherical waves use the Hankel function of the second kind.

## Material parameters

For a material with relative permittivity $\epsilon_r$ and relative
permeability $\mu_r$, the absolute material parameters are

$$
\epsilon = \epsilon_0\epsilon_r, \qquad
\mu = \mu_0\mu_r,
$$

where the vacuum permittivity is derived from the speed of light and vacuum
permeability:

$$
\epsilon_0 = \frac{1}{\mu_0 c^2}.
$$

At frequency $f$, the angular frequency, wavenumber, and wave impedance are

$$
\omega = 2\pi f, \qquad
k = \omega\sqrt{\mu\epsilon}, \qquad
\eta = \sqrt{\frac{\mu}{\epsilon}}.
$$

Square roots use the complex branch so negative or complex material parameters
can produce complex wavenumbers and impedances.

## Scalar Helmholtz equation

The field component parallel to the cylinder axis is the scalar unknown:

$$
\psi =
\begin{cases}
E_z, & \text{TM polarization},\\
H_z, & \text{TE polarization}.
\end{cases}
$$

In each homogeneous region $i$, it satisfies

$$
\left(\nabla_t^2 + k_i^2\right)\psi_i = 0,
$$

where $\nabla_t^2$ is the Laplacian in the transverse $(\rho,\phi)$ plane. A
separated cylindrical mode has angular dependence $\cos(n\phi)$ and radial
dependence given by a Bessel or Hankel function.

The incident plane wave travels in the positive $x$ direction and has unit
amplitude in the longitudinal component. Its phasor is

$$
\psi_{\mathrm{inc}} = e^{-j k_{\mathrm{out}}x}
= e^{-j k_{\mathrm{out}}\rho\cos\phi}.
$$

The incident plane wave has the Bessel expansion

$$
\psi_{\mathrm{inc}}=e^{-jk_{\mathrm{out}}x}
=\sum_{n=0}^{\infty}q_nj^{-n}J_n(k_{\mathrm{out}}\rho)\cos(n\phi).
$$

The exterior total field is this incident wave plus the truncated scattered
field:

$$
\psi_{\mathrm{out}}(\rho,\phi) = e^{-jk_{\mathrm{out}}x}
+ \sum_{n=0}^{N-1}q_n a_nH_n^{(2)}(k_{\mathrm{out}}\rho)\cos(n\phi).
$$

Here $a_n$ is the scattered-wave coefficient and

$$
q_n = \begin{cases}1,&n=0,\\2,&n\geq1.\end{cases}
$$

Finite annular layers use both radial Hankel solutions:

$$
\psi_i(\rho,\phi) =
\sum_{n=0}^{N-1} q_n\left[
c_{n,i}H_n^{(1)}(k_i\rho)
\; + \;
d_{n,i}H_n^{(2)}(k_i\rho)
\right]\cos(n\phi),
\qquad i=1,\ldots,L-2.
$$

The regular dielectric core uses

$$
\psi_0(\rho,\phi) =
\sum_{n=0}^{N-1}q_n(c_{n,0}+d_{n,0})J_n(k_0\rho)\cos(n\phi).
$$

Here $L$ is the total number of regions, indexed from the core ($0$) to the
exterior ($L-1$). ScatSol solves the modal coefficients by enforcing the
interface conditions below. The cosine-mode multiplicity is $q_0=1$ and
$q_n=2$ for $n\geq1$.

## Interface conditions

At an interface of radius $R$ between regions $i$ and $i+1$, the tangential
fields are continuous. In terms of the scalar field this becomes

$$
\psi_i(R,\phi) = \psi_{i+1}(R,\phi),
$$

and

$$
\begin{aligned}
\frac{1}{\mu_i}\frac{\partial\psi_i}{\partial\rho}
&= \frac{1}{\mu_{i+1}}\frac{\partial\psi_{i+1}}{\partial\rho},
&&\text{TM},\\[4pt]
\frac{1}{\epsilon_i}\frac{\partial\psi_i}{\partial\rho}
&= \frac{1}{\epsilon_{i+1}}\frac{\partial\psi_{i+1}}{\partial\rho},
&&\text{TE}.
\end{aligned}
$$

These are continuity of $E_z$ and $H_\phi$ for TM, and continuity of $H_z$ and
$E_\phi$ for TE.

## Reconstructing transverse fields

Once $\psi$ is known, Maxwell's equations give the transverse components. With
$\omega=2\pi f$, TM fields are

$$
E_z = \psi, \qquad E_\rho=E_\phi=H_z=0,
$$

$$
H_\rho = -\frac{1}{j\omega\mu\rho}\frac{\partial E_z}{\partial\phi},
\qquad
H_\phi = \frac{1}{j\omega\mu}\frac{\partial E_z}{\partial\rho}.
$$

For TE fields,

$$
H_z = \psi, \qquad H_\rho=H_\phi=E_z=0,
$$

$$
E_\rho = \frac{1}{j\omega\epsilon\rho}\frac{\partial H_z}{\partial\phi},
\qquad
E_\phi = -\frac{1}{j\omega\epsilon}\frac{\partial H_z}{\partial\rho}.
$$

The transverse components are rotated into Cartesian coordinates using

$$
F_x = F_\rho\cos\phi-F_\phi\sin\phi, \qquad
F_y = F_\rho\sin\phi+F_\phi\cos\phi,
$$

where $F$ is the transverse electric or magnetic field. `total_field()` returns
the resulting Cartesian components in columns $x,y,z$.

## PEC core

The PEC is modeled as an exact boundary condition at the surface of the
innermost region, rather than as a large finite permittivity. Vanishing
tangential electric field gives

$$
E_z(R_{\mathrm{PEC}})=0 \quad (\mathrm{TM}),
\qquad
\frac{\partial H_z}{\partial\rho}(R_{\mathrm{PEC}})=0 \quad (\mathrm{TE}).
$$

The interior PEC fields are returned as zero. The current implementation
supports PEC only as the innermost cylinder, surrounded by finite-material
layers.

## Mode count

`solve(n=...)` treats $n$ as a maximum number of modes. At finite interfaces,
ScatSol also applies the size-parameter estimate

$$
N_{\mathrm{est}} =
\left\lceil x_{\max}+4x_{\max}^{1/3}+2\right\rceil,
\qquad
x_{\max}=\max_{R\in\mathcal I}\;\max_{m\in\mathcal F(R)}|k_mR|,
$$

where $\mathcal I$ is the set of finite interface radii and $\mathcal F(R)$
contains the finite-material regions adjacent to interface $R$ (so a PEC
region contributes no wavenumber).

ScatSol uses the smaller of `n` and $N_{\mathrm{est}}$ for the scattered-field
coefficients. The incident plane wave is evaluated directly, so its accuracy
does not depend on this modal truncation. The estimate avoids unnecessary
high-order Hankel evaluations; it is a practical truncation rule, not an error
bound. The mode count actually used is available as
`solution.n`.

## Spherical scattering

For concentric spherical regions, the solver expands the fields in vector
spherical waves. The incident wave is x-polarized and travels in the positive
z direction:

$$
\mathbf E_{\mathrm{inc}}=\hat{\mathbf x}e^{-jk_{\mathrm{out}}z},
\qquad
\mathbf H_{\mathrm{inc}}=\eta_{\mathrm{out}}^{-1}\hat{\mathbf y}
e^{-jk_{\mathrm{out}}z}.
$$

The angular dependence is expressed with associated Legendre functions of
orders $n=1,2,\ldots,N$. Each order has two independent vector-wave channels,
usually called electric (or TM) and magnetic (or TE) multipoles. In each
material, their radial dependence is built from the Riccati spherical Bessel
functions

$$
\psi_n(z)=zj_n(z),\qquad \chi_n(z)=zy_n(z),\qquad
\xi_n^{(2)}(z)=z h_n^{(2)}(z), \qquad z=kr.
$$

The core uses regular waves $j_n$. Finite shells use both $j_n$ and $y_n$,
while the exterior scattered field uses $h_n^{(2)}$; the known incident field
uses $j_n$. For every order and channel, the solver finds the shell and
scattered-wave coefficients by matching tangential electric and magnetic
fields at each radius. This gives a small linear system per order rather than
one system coupling all angular orders. `solution.an` and `solution.bn` hold
the exterior scattered coefficients for the electric and magnetic channels,
respectively. Spherical solutions include both channels in the returned vector
fields; they do not take a `Polarization` argument.

At a dielectric interface, tangential $E$ and $H$ are continuous. In the
Riccati radial basis used by the solver, the two channel matchings use these
weighted value and derivative pairs:

$$
\begin{array}{c|cc}
\text{channel} & \text{radial value weight} &
\text{Riccati derivative weight}\\ \hline
\text{electric} & 1/\mu & 1/k\\
\text{magnetic} & 1/k & 1/\mu
\end{array}
$$

Each weighted quantity is matched between the media on the two sides of the
interface. The derivatives are with respect to $z=kr$, as in the Riccati
functions above.

A PEC sphere is supported as the innermost region. Zero tangential electric
field gives a zero Riccati radial derivative for the electric channel and a
zero Riccati radial value for the magnetic channel at the PEC surface. The PEC
interior fields are zero. PEC is not supported in spherical shells or in the
exterior.

The spherical mode count is truncated using the same size-parameter estimate
described above, based on the largest finite interface size $|kR|$. The
estimate is a practical truncation rule rather than an error bound.
