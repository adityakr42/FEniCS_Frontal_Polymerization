from dolfin import *
import time

T = 5.0            # final time
num_steps = 1000 #T/dt # number of time steps
dt = T/num_steps #0.0005# time step size    

# Create mesh and define function space
nel= 600
mesh = IntervalMesh(nel,0.0,0.005)
Vte = FiniteElement('CG', mesh.ufl_cell(), 1) 
V = FunctionSpace(mesh, MixedElement([Vte, Vte]))

# Define boundary condition
def boundary(x, on_boundary):
	return on_boundary

left =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 0.0, tol=1e-4)
T0=Expression("600.0",degree=1)
bcl = DirichletBC(V.sub(0), T0, left)
bcs = [bcl]

# Define functions
U_ = TestFunction(V)
(v, beta)  = split(U_)
dU = TrialFunction(V)
(du, dalpha) = split(dU)
U = Function(V)
(u, alpha) = split(U)
Uold = Function(V)
(u_n, alpha_n) = split(Uold)

# Define initial value
U_init = Expression(('303.0','0.01'),degree=2)
Uold.interpolate(U_init)
U.interpolate(U_init)



# Define material properties
kappa, rho, Cp, H, A, Er, R, n = 0.21, 1060.0, 1529.0, 510000, 1.79e10, 82200.0, 8.314, 2.23
g=A*exp(-Er/(R*u))*((1-alpha)**n)

# Define variational problem
diff_eq = rho*Cp*u*v*dx - (rho*Cp*u_n)*v*dx + dt*kappa*inner(grad(u), grad(v))*dx  - rho*H*alpha*v*dx  + rho*H*alpha_n*v*dx
reac_eq = -alpha*beta*dx + (dt*g)*beta*dx + alpha_n*beta*dx

eq_form= diff_eq +reac_eq #diff_eq +
Jac = derivative(eq_form, U, dU)


snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "gmres",   #lu or gmres or cg 'preconditioner: ilu, amg, jacobi'
                                          "preconditioner": "amg",						  
                                          "maximum_iterations": 100,
                                          "report": True,
                                          "error_on_nonconvergence": True}}	

snes_solver_parameters2 = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",   #lu or gmres or cg 'preconditioner: ilu, amg, jacobi'						  
                                          "maximum_iterations": 100,
                                          "report": True,
                                          "error_on_nonconvergence": True}}	

# Create VTK file for saving solution
vtkfile = File('TMPTA/1Dexamp.pvd')
vtkfile << (U, 0)

# Time-stepping
t = 0
step=1

while step<num_steps:
	# Update current time
	t += dt
	print('t= %f' %t)
	if t<0.5:
		bct=bcs
	else:
		bct=[]
	
	# Compute solution
	Problem_u = NonlinearVariationalProblem(eq_form, U, bct, J=Jac)
	solver_u  = NonlinearVariationalSolver(Problem_u)
	solver_u.parameters.update(snes_solver_parameters)
	(iter, converged) = solver_u.solve()

	
	# Save to file and plot solution
	if step % 10==0:
		vtkfile << (U.leaf_node(), step)


	# Update previous solution
	Uold.assign(U)
	step+=1