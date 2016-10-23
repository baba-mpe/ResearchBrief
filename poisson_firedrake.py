# Poisson's equation
# =================
#
# The Poisson equation is a linear elliptic equation describing a 
# potential field arising by a given source. Here we choose to write 
# the Poisson eqaution in two dimensions to demonstrate the use of 
# vector function spaces:
#
# .. math:: 
#
#    \nabla^2 u = f
#
#    u = 0 \textrm{on}\ \Omega_D
#    \frac{\partial u}{\partial n} = 0 \textrm{on}\ \Omega_N
#
# where :math:`\Omega_D` is the Dirichlet part and :math:`\Omega_N` 
# the Neumann part of the domain boundary respectively. The solution 
# :math:`u` is sought in some suitable real-valued function space 
# :math:`V`. We take the inner product with an arbitrary test function 
# :math:`v\in V` and integrate the Laplace term by parts:
#
# .. math::
#
#    \int_\Omega \nabla u\cdot\nabla v \mathrm d x
#    = \int_\Omega f v \mathrm d x.
#
# The boundary conditions have been used to discard the surface
# integral. We can now proceed to set up the problem. We choose a 
# resolution and set up a square mesh::

from firedrake import *
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

x, y = SpatialCoordinate(mesh)
subdomain = SubDomainData(x+y<1)

f = Function(V)
f.interpolate(as_ufl(Expression("sin(4*pi*x[0]) * cos(4*pi*x[1])")), subset=subdomain)

plot(f)
plt.show()

# Define Dirichlet boundary (x = 1 or y = 0)
u0 = Constant(0.0)
bcx_1 = DirichletBC(V, u0,  2)
bcy_0 = DirichletBC(V, u0,  4)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

#f = Function(V)
#f.interpolate(Expression("sin(4*pi*x[0]) * cos(4*pi*x[1])"))
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs = [bcx_1,bcy_0])

# Save solution in VTK format
#file = File("poisson.pvd")
#file << u

# Plot solution
plot(u)
plt.show()
