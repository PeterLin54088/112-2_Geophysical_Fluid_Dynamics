import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

# domain size

Lx, Ly = (2.*10000.e3,2.*5000.e3 )
nx, ny = (128*4,64*4)     

beta  = 2.e-11
omega = 7.27e-7
D     = 0.e8

# Create bases and domain

x_basis = de.Fourier('x', nx, interval=(-Lx/2,Lx/2), dealias=3/2)
y_basis = de.Fourier('y',ny, interval=(-Ly/2,Ly/2), dealias=3/2)
domain  = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Equations
# HR83 means: Haidvogel & Rhines (1983); 
# Forcing (F below) has a period of 100-day (i.e. omega = freq = 2*pi/T) 
# a0 (= 1e-11) is the magnitude of wind stress curl (roughly equal to wind stress = 0.5 pa, dy = 100 km, H = 500m)
# i.e. torque by wind in bt vorticity eq ~ tau/rho0/H/Ly ~ 0.5/1e3/500/100e3

problem = de.IVP(domain, variables=['psi'])

problem.parameters['beta']  = beta
problem.parameters['D']     = D              # hyperdiffusion coefficient
problem.parameters['omega'] = omega          # HR83 forcing freq        

# solve the problem from the equations
# ζ = Δψ
# ∂/∂t[∆ψ] + β ∂/∂x[ψ] = -J(ζ, ψ)

problem.substitutions['zeta'] = "  d(psi,x=2) + d(psi,y=2) "
problem.substitutions['u']    = " -dy(psi) "
problem.substitutions['v']    = "  dx(psi) "
problem.substitutions['F']    = "  1.e-11 * exp(-(x**2+y**2)/(10.e3)**2) * cos(omega * t) "

# define Laplcian and Jacobian operators 
problem.substitutions['L(a)']   = "  d(a,x=2) + d(a,y=2) "
problem.substitutions['J(a,b)'] = "  dx(a)*dy(b) - dy(a)*dx(b) "

# Hyperviscosity; biharmonic diffusion for vorticity 

problem.substitutions['HD(a)']         = "  -D*L(L(a)) "

# barotropic vorticity equation
problem.add_equation("dt(zeta) + beta*v -HD(zeta) = F", condition="(nx != 0) or  (ny != 0)")    # linear
#problem.add_equation("dt(zeta) + beta*v -HD(zeta) = -J(psi,zeta) + F", condition="(nx != 0) or  (ny != 0)")    # nonlinear
problem.add_equation("psi = 0",                                         condition="(nx == 0) and (ny == 0)")

# Timestepping

ts = de.timesteppers.RK443
#ts = de.timesteppers.RK222

# Initial Value Problem

solver =  problem.build_solver(ts)

# Now we set integration parameters and the CFL.

#dt = 0.05*Lx/nx
#dt = 0.1*Lx/nx
dt = 0.02*Lx/nx
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=2,
                     max_change=1.2, min_change=0.5, max_dt=3*dt)
CFL.add_velocities(('u','v'))

#cfl = flow_tools.CFL(solver,initial_dt,safety=0.8,threshold=0.05)

# Analysis (output every 12 hours)

snap = solver.evaluator.add_file_handler("snapshots", sim_dt=24.*3600.,max_writes=250)
snap.add_system(solver.state, layout='g')
snap.add_task("zeta", layout='g', name="zeta")
snap.add_task("u", layout='g', name="u")
snap.add_task("v", layout='g', name="v")

#snap.add_task("dx(dx(psi)) + dy(dy(psi))", name="vor")
#snap.add_task("F", layout='g', name="F")
#snap.add_task("psi", layout='c', name="psi_c")

#parameters = solver.evaluator.add_file_handler('parameters', iter=np.inf, max_writes=1)

# Integration parameters (sim time = 100 days)
solver.stop_sim_time = 200. * 86400.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf 

logger.info('Starting loop')
start_run_time = time.time()

while solver.ok:
    dt = CFL.compute_dt()
    solver.step(dt)
    if solver.iteration % 2 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time/86400, dt))


end_run_time = time.time()
logger.info('Iterations: %i' %solver.iteration)
logger.info('Sim end time: %f' %solver.sim_time)
logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))




