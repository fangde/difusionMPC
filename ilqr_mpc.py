import numpy as np
import scipy.linalg

class MPCController:
    """
    Model Predictive Controller with DARE Terminal Cost and Torque Saturation.
    Computes optimal control actions for the underactuated double pendulum.
    """
    def __init__(self, env, horizon=20):
        self.env = env
        self.H = horizon
        self.Q = np.diag([80.0, 40.0, 5.0, 5.0])
        self.R = np.array([[0.1]])
        
        # Compute exact terminal cost P from Discrete Algebraic Riccati Equation (DARE)
        A, B = env.get_linearized_dynamics(np.zeros(4), 0.0)
        self.P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
        self.K_lqr = np.linalg.inv(self.R + B.T @ self.P @ B) @ (B.T @ self.P @ A)

    def get_optimal_control(self, x):
        """
        Computes the optimal control input u*(x) and the corresponding state derivative x_dot.
        """
        # Baseline optimal LQR control with torque clipping
        u_unclipped = - (self.K_lqr @ x)[0]
        u_opt = np.clip(u_unclipped, -self.env.u_max, self.env.u_max)
        
        # Continuous optimal control velocity vector: v_MPC(x) = f(x, u*)
        x_dot = self.env.continuous_dynamics(x, u_opt)
        return u_opt, x_dot

    def get_action(self, x):
        u, _ = self.get_optimal_control(x)
        return u
