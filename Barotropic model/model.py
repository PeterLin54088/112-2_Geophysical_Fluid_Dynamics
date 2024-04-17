import numpy as np

def _Leapfrog_RobertAsselin(x_nm1, x_n, rhs, dt, gamma=0.5):
    x_np1 = x_nm1 + 2 * dt * rhs
    x_n_filtered = x_n + gamma * (x_nm1 - 2*x_n + x_np1)
    return x_np1, x_n_filtered
def _GetUV(psi, kk, ll):
    spe_psi = np.fft.fft2(psi)
    u = -np.fft.ifft2(1j*ll*spe_psi)
    v = np.fft.ifft2(1j*kk*spe_psi)
    return np.real(u), np.real(v)
def _Laplacian(data, kk, ll):
    spe_data = np.fft.fft2(data)
    factor = -kk**2-ll**2
    output = spe_data * factor
    output = np.fft.ifft2(output)
    return np.real(output)
def _InverseLaplacian(data, kk, ll):
    spe_data = np.fft.fft2(data)
    factor = -kk**2-ll**2
    factor[0,0] = 1.0
    output = spe_data / factor
    output[0,0] = 0.0
    output = np.fft.ifft2(output)
    return np.real(output)

class BarotropicModel:
    def __init__(self,
                 vor_init, forcing_struct,
                 omega, beta,
                 Nx, Ny, Lx, Ly,
                 dt, total_t):
        """
        """
        self.dt = dt
        self.total_t = total_t
        self.interval = 86400
        self.n_steps = int(total_t/dt)
        
        self.vort = np.zeros((int(self.total_t/self.interval)+1, Ny, Nx))
        self.vort[0,:,:] = vor_init
        self.psi = np.zeros_like(self.vort)
        self.forcing_structure = forcing_struct
        
        self.k = np.fft.fftfreq(Nx, d=Lx/(2*np.pi)/Nx)
        self.l = np.fft.fftfreq(Ny, d=Ly/(2*np.pi)/Ny)
        self.kk, self.ll = np.meshgrid(self.k, self.l)
        
        self.beta = beta
        self.omega = omega
        self.gamma = 0.5    # the parameter used in leapfrog_asselin_filter
        
    def _compute_rhs(self, psi, time):
        lap = _Laplacian(psi, self.kk, self.ll)
        _, v = _GetUV(psi, self.kk, self.ll)
        planetary_pv_advection = -self.beta*v
        oscillatory_forcing = self.forcing_structure*np.cos(self.omega*time)
        hyperdiffusion = 0.0e-7*_Laplacian(lap, self.kk, self.ll)
        RHS = _InverseLaplacian(oscillatory_forcing + 
                                planetary_pv_advection + 
                                hyperdiffusion, 
                                self.kk, self.ll)
        return RHS, lap
    
    def run(self):
        # initial condition
        self.psi[0,:,:] = _InverseLaplacian(self.vort[0,:,:], self.kk, self.ll)
        ctime = 0
        output_index = 0
        #
        print(f'Step 1 / {self.n_steps} ...     ', end='\r')
        psi_tn = self.psi[0,:,:]
        rhs, vort_tn = self._compute_rhs(psi_tn, ctime)
        psi_tnp1 = psi_tn + self.dt * rhs
        ctime += self.dt
        output_index += 1
        
        # from i_step = n to n+1: leapfrog + RA filter
        for i_step in range(1, self.n_steps):
            print(f'Step {i_step+1} / {self.n_steps} ...     ', end='\r')
            ctime += self.dt
            psi_tnm1 = psi_tn
            psi_tn = psi_tnp1
            rhs, vort_tn = self._compute_rhs(psi_tn, ctime)
            psi_tnp1, psi_tn = _Leapfrog_RobertAsselin(psi_tnm1, psi_tn, rhs, 
                                                       self.dt, self.gamma)
            if ctime==output_index*self.interval:
                self.psi[output_index,:,:] = psi_tn
                self.vort[output_index,:,:] = vort_tn
                output_index += 1
                
        print('\n[Done]')