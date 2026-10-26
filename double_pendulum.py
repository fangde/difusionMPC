import numpy as np

class DoublePendulum:
    """
    Planar Underactuated Double Pendulum (Pendubot).
    Joint 1 (shoulder) is actuated: tau_1 = u.
    Joint 2 (elbow) is unactuated: tau_2 = 0.
    
    Coordinate convention:
    theta_1: absolute angle from upright vertical (+y axis).
    theta_2: relative angle of link 2 with respect to link 1.
    Upright equilibrium target: x* = [0, 0, 0, 0]^T.
    """
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, u_max=15.0, dt=0.02):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.r1 = l1 / 2.0
        self.r2 = l2 / 2.0
        self.I1 = (m1 * l1**2) / 12.0
        self.I2 = (m2 * l2**2) / 12.0
        self.g = 9.81
        self.u_max = u_max
        self.dt = dt
        self.state_dim = 4
        self.action_dim = 1

    def get_matrices(self, q, dq):
        th1, th2 = q
        dth1, dth2 = dq
        
        # Mass matrix M(q)
        M11 = self.I1 + self.I2 + self.m1 * self.r1**2 + self.m2 * (self.l1**2 + self.r2**2 + 2 * self.l1 * self.r2 * np.cos(th2))
        M12 = self.I2 + self.m2 * (self.r2**2 + self.l1 * self.r2 * np.cos(th2))
        M22 = self.I2 + self.m2 * self.r2**2
        M = np.array([[M11, M12], [M12, M22]], dtype=np.float64)
        
        # Coriolis & Centrifugal matrix C(q, dq)
        h = -self.m2 * self.l1 * self.r2 * np.sin(th2)
        C = np.array([[h * dth2, h * (dth1 + dth2)],
                      [-h * dth1, 0.0]], dtype=np.float64)
        
        # Gravity vector G(q)
        G1 = -self.m1 * self.g * self.r1 * np.sin(th1) - self.m2 * self.g * (self.l1 * np.sin(th1) + self.r2 * np.sin(th1 + th2))
        G2 = -self.m2 * self.g * self.r2 * np.sin(th1 + th2)
        G = np.array([G1, G2], dtype=np.float64)
        
        return M, C, G

    def continuous_dynamics(self, x, u):
        """Returns x_dot = f(x, u)"""
        q = x[:2]
        dq = x[2:]
        u_clamped = np.clip(float(u), -self.u_max, self.u_max)
        
        M, C, G = self.get_matrices(q, dq)
        tau = np.array([u_clamped, 0.0], dtype=np.float64)
        
        ddq = np.linalg.solve(M, tau - C @ dq - G)
        return np.concatenate([dq, ddq])

    def step(self, x, u):
        """Discrete-time forward step using Runge-Kutta 4th order (RK4)"""
        u_clamped = np.clip(float(u), -self.u_max, self.u_max)
        dt = self.dt
        
        k1 = self.continuous_dynamics(x, u_clamped)
        k2 = self.continuous_dynamics(x + 0.5 * dt * k1, u_clamped)
        k3 = self.continuous_dynamics(x + 0.5 * dt * k2, u_clamped)
        k4 = self.continuous_dynamics(x + dt * k3, u_clamped)
        
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Normalize angles to [-pi, pi]
        x_next[0] = (x_next[0] + np.pi) % (2 * np.pi) - np.pi
        x_next[1] = (x_next[1] + np.pi) % (2 * np.pi) - np.pi
        return x_next

    def get_linearized_dynamics(self, x, u, eps=1e-5):
        """Computes discrete Jacobian matrices A = df/dx, B = df/du"""
        n, m = self.state_dim, self.action_dim
        A = np.zeros((n, n), dtype=np.float64)
        B = np.zeros((n, m), dtype=np.float64)
        
        x_nom = self.step(x, u)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            A[:, i] = (self.step(x + dx, u) - x_nom) / eps
            
        for j in range(m):
            du = eps
            B[:, j] = (self.step(x, u + du) - x_nom) / eps
            
        return A, B
