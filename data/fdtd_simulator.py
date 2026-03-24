import numpy as np
import scipy.signal as signal

class FDTDSimulator:
    """
    A simplified 2D FDTD simulator for generating synthetic MWI S-parameters.
    This simulates the wave propagation in a dielectric medium (phantom).
    """
    def __init__(self, size=(128, 128), freq=2.4e9, dx=0.005):
        self.size = size
        self.freq = freq
        self.dx = dx
        self.dt = dx / (3e8 * np.sqrt(2))  # Courant stability condition
        
        # Initialize fields
        self.Ez = np.zeros(size)
        self.Hx = np.zeros((size[0], size[1]-1))
        self.Hy = np.zeros((size[0]-1, size[1]))
        
        # Dielectric properties (Relative Permittivity)
        self.epsilon_r = np.ones(size)
        
    def add_tumor(self, center, radius, epsilon_tumor=50):
        """Places a tumor (high permittivity) in the phantom."""
        y, x = np.ogrid[:self.size[0], :self.size[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        self.epsilon_r[mask] = epsilon_tumor
        
    def simulate(self, steps=500):
        """Runs the FDTD simulation."""
        # Simple update equations (simplified 2D TMz)
        # In a real scenario, this would involve PML and proper antenna source modeling
        history = []
        for t in range(steps):
            # Update H fields
            self.Hx -= (self.dt / (self.dx * 1.256e-6)) * np.diff(self.Ez, axis=1)
            self.Hy += (self.dt / (self.dx * 1.256e-6)) * np.diff(self.Ez, axis=0)
            
            # Update E field
            eps = self.epsilon_r * 8.854e-12
            self.Ez[1:-1, 1:-1] += (self.dt / (eps[1:-1, 1:-1] * self.dx)) * (
                (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) - 
                (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1])
            )
            
            # Source (Gaussian pulse)
            src_val = np.exp(-((t - 50 )**2) / (2 * 10**2)) * np.sin(2 * np.pi * self.freq * t * self.dt)
            self.Ez[self.size[0]//2, 5] = src_val
            
            # Collect "S-parameters" (approximated as Ez at receiver locations)
            # Sensors at the periphery
            history.append(self.Ez[self.size[0]//2, -5])
            
        return np.array(history)

if __name__ == "__main__":
    sim = FDTDSimulator()
    sim.add_tumor(center=(64, 64), radius=5)
    s_params = sim.simulate()
    print(f"Simulation completed. History length: {len(s_params)}")
