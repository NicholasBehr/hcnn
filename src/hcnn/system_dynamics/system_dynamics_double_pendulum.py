"""
Double Pendulum System Dynamics with Uniform Mass Distribution

This module implements a double pendulum system using generalized coordinates.
The system consists of two connected pendulums where torque can only be applied
at the first link (base joint).

Physical Model:
- Two rigid links connected in series with uniform mass distribution
- Center of mass at midpoint of each link (lc = l/2)
- Moment of inertia: I = (1/12) * m * l² about center of mass
- Generalized coordinates: q = [θ₁, θ₂] where θ₁ is absolute, θ₂ is relative
- Control input: τ₁ (torque at first joint only)
- State vector: [θ₁, θ₂, θ̇₁, θ̇₂]

The dynamics are derived from the Euler-Lagrange equations for the system
Lagrangian L = T - V, where T is kinetic energy and V is potential energy.
"""

import jax
import jax.numpy as jnp
from flax import struct

from hcnn.system_dynamics.system_dynamics_continuous import (
    SystemDynamicsContinuous,
)


@struct.dataclass
class SystemDynamicsDoublePendulum(SystemDynamicsContinuous):
    """
    Double pendulum dynamics with actuation only at the first joint.

    Assumes uniform mass distribution for both links, which simplifies
    the model:
    - Center of mass is at the midpoint of each link (lc1 = l1/2, lc2 = l2/2)
    - Moment of inertia about center of mass: I = (1/12) * m * l²

    Physical Parameters:
    - m1, m2: masses of the first and second links
    - l1, l2: lengths of the first and second links
    - g: gravitational acceleration
    - b1, b2: damping coefficients at joints

    State representation: [θ₁, θ₂, θ̇₁, θ̇₂]
    - θ₁: angle of first link from vertical (rad)
    - θ₂: angle of second link relative to first link (rad)
    - θ̇₁, θ̇₂: angular velocities (rad/s)

    Action: [τ₁] - torque applied at first joint only
    """

    # Physical parameters - with defaults to avoid field ordering issues
    m1: float = 0.1  # Mass of link 1 (kg)
    m2: float = 0.1  # Mass of link 2 (kg)
    l1: float = 0.1  # Length of link 1 (m)
    l2: float = 0.1  # Length of link 2 (m)
    g: float = 9.81  # Gravitational acceleration (m/s²)
    b1: float = 0.1  # Damping coefficient at joint 1 (N⋅m⋅s/rad)
    b2: float = 0.1  # Damping coefficient at joint 2 (N⋅m⋅s/rad)

    @classmethod
    def factory(
        cls,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        g: float = 9.81,
        b1: float = 0.1,
        b2: float = 0.1,
        dt: float = 0.01,
    ) -> "SystemDynamicsDoublePendulum":
        """
        Factory method to create a double pendulum system with uniform links.

        Args:
            m1, m2: Masses of the links (kg)
            l1, l2: Lengths of the links (m)
            g: Gravitational acceleration (m/s²)
            b1, b2: Damping coefficients (N⋅m⋅s/rad)
            dt: Integration timestep (s)
            integrator: ODE integration method

        Returns:
            SystemDynamicsDoublePendulum instance with state_dim=4, action_dim=1

        Note:
            Assumes uniform mass distribution: center of mass at midpoint,
            moment of inertia I = (1/12) * m * l² about center of mass.
        """
        # Validate physical parameters
        assert m1 > 0 and m2 > 0, "Masses must be positive"
        assert l1 > 0 and l2 > 0, "Lengths must be positive"
        assert g > 0, "Gravitational acceleration must be positive"
        assert b1 >= 0 and b2 >= 0, "Damping coefficients must be non-negative"
        assert dt > 0, "Timestep must be positive"

        return cls(
            state_dim=4,  # [θ₁, θ₂, θ̇₁, θ̇₂]
            action_dim=1,  # [τ₁] - torque at first joint only
            dt=dt,
            m1=m1,
            m2=m2,
            l1=l1,
            l2=l2,
            g=g,
            b1=b1,
            b2=b2,
        )

    def _continuous_dynamics(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """
        Compute the continuous dynamics dx/dt = f(x, u) for the double pendulum.

        The dynamics are derived from the Euler-Lagrange equations:
        d/dt(∂L/∂q̇) - ∂L/∂q = τ

        where L = T - V is the Lagrangian, q = [θ₁, θ₂] are generalized
        coordinates, and τ = [τ₁, 0] since we can only actuate the first joint.

        Args:
            state: [θ₁, θ₂, θ̇₁, θ̇₂] - angles and angular velocities
            action: [τ₁] - torque at first joint

        Returns:
            State derivative [θ̇₁, θ̇₂, θ̈₁, θ̈₂]
        """
        # Extract state variables
        theta1 = state[..., 0]  # Angle of link 1
        theta2 = state[..., 1]  # Angle of link 2
        dtheta1 = state[..., 2]  # Angular velocity of link 1
        dtheta2 = state[..., 3]  # Angular velocity of link 2

        # Extract control input
        tau1 = action[..., 0]  # Torque at first joint

        # Compute the mass matrix M(q), Coriolis matrix C(q,q̇),
        # and gravity vector G(q)
        m11, m12, m22, c1, c2, g1, g2 = self._compute_dynamics_matrices(
            theta1, theta2, dtheta1, dtheta2
        )

        # The equation of motion is: M(q)q̈ + C(q,q̇)q̇ + G(q) + B = τ
        # where B = [b1*θ̇₁ - b2*(θ̇₁-θ̇₂), b2*(θ̇₁-θ̇₂)] is joint damping
        # and τ = [τ₁, 0] since only first joint is actuated

        # Right hand side of the equation
        rhs1 = (
            tau1 - c1 - g1 - self.b1 * dtheta1 + self.b2 * (dtheta2 - dtheta1)
        )
        rhs2 = (
            0.0 - c2 - g2 - self.b2 * (dtheta1 - dtheta2)
        )  # No torque at joint 2

        # Solve for angular accelerations: [θ̈₁, θ̈₂] = M⁻¹ * rhs
        # For 2x2 matrix inversion: [a b; c d]⁻¹ = 1/det * [d -b; -c a]
        # Define numerical stability constant
        eps = 1e-12
        det_m = m11 * m22 - m12 * m12  # M is symmetric, so M21 = M12

        # Ensure numerical stability
        det_m = jnp.where(jnp.abs(det_m) < eps, eps, det_m)

        ddtheta1 = (m22 * rhs1 - m12 * rhs2) / det_m
        ddtheta2 = (-m12 * rhs1 + m11 * rhs2) / det_m

        # Return state derivative
        return jnp.stack([dtheta1, dtheta2, ddtheta1, ddtheta2], axis=-1)

    def _compute_dynamics_matrices(self, theta1, theta2, dtheta1, dtheta2):
        """
        Compute the mass matrix, Coriolis terms, and gravity terms.

        Uses uniform mass distribution assumption:
        - Center of mass at midpoint: lc1 = l1/2, lc2 = l2/2
        - Moment of inertia: I1 = (1/12)*m1*l1², I2 = (1/12)*m2*l2²

        Coordinate system:
        - θ₁: absolute angle of link 1 from vertical
        - θ₂: relative angle of link 2 with respect to link 1

        This implements the standard double pendulum dynamics derivation.
        See Murray, Li, and Sastry "A Mathematical Introduction to
        Robotic Manipulation" or Craig "Introduction to Robotics"
        for detailed derivations.
        """
        # Uniform mass distribution parameters
        lc1 = self.l1 / 2  # Center of mass at midpoint
        lc2 = self.l2 / 2  # Center of mass at midpoint
        i1 = (1.0 / 12.0) * self.m1 * self.l1**2  # Moment of inertia
        i2 = (1.0 / 12.0) * self.m2 * self.l2**2  # Moment of inertia

        # Frequently used terms
        cos_theta2 = jnp.cos(theta2)  # cos of relative angle
        sin_theta2 = jnp.sin(theta2)  # sin of relative angle
        sin_theta1 = jnp.sin(theta1)

        # Mass matrix M(q) - symmetric positive definite
        m11 = (
            self.m1 * lc1**2
            + self.m2 * self.l1**2
            + self.m2 * lc2**2
            + 2 * self.m2 * self.l1 * lc2 * cos_theta2
            + i1
            + i2
        )

        m12 = self.m2 * lc2**2 + self.m2 * self.l1 * lc2 * cos_theta2 + i2

        m22 = self.m2 * lc2**2 + i2

        # Coriolis and centrifugal terms C(q,q̇)q̇
        h = -self.m2 * self.l1 * lc2 * sin_theta2

        c1 = h * dtheta2**2 + 2 * h * dtheta1 * dtheta2
        c2 = -h * dtheta1**2

        # Gravity terms G(q)
        g1 = (
            self.m1 * lc1 + self.m2 * self.l1
        ) * self.g * sin_theta1 + self.m2 * lc2 * self.g * jnp.sin(
            theta1 + theta2
        )

        g2 = self.m2 * lc2 * self.g * jnp.sin(theta1 + theta2)

        return m11, m12, m22, c1, c2, g1, g2

    def forward_kinematics(self, state: jax.Array) -> jax.Array:
        """
        Computes the world-space positions of the link ends.

        The coordinate system has its origin at the first joint (pivot),
        with the y-axis pointing upwards and the x-axis to the right.
        A state of [0, 0, 0, 0] corresponds to the pendulum hanging
        straight down along the negative y-axis.

        Args:
            state: (..., 4) array of [θ₁, θ₂, θ̇₁, θ̇₂].

        Returns:
            (..., 4) array of world coordinates [x1, y1, x2, y2], where
            (x1, y1) is the end of link 1 and (x2, y2) is the end of link 2.
        """
        theta1 = state[..., 0]
        theta2 = state[..., 1]

        # Position of the end of the first link
        x1 = self.l1 * jnp.sin(theta1)
        y1 = -self.l1 * jnp.cos(theta1)

        # Position of the end of the second link, relative to the first
        x2 = x1 + self.l2 * jnp.sin(theta1 + theta2)
        y2 = y1 - self.l2 * jnp.cos(theta1 + theta2)

        return jnp.stack([x1, y1, x2, y2], axis=-1)

    def get_energy(self, state: jax.Array) -> jax.Array:
        """
        Compute the total mechanical energy of the system.

        Args:
            state: [θ₁, θ₂, θ̇₁, θ̇₂]

        Returns:
            Total energy (kinetic + potential)
        """
        theta1 = state[..., 0]
        theta2 = state[..., 1]
        dtheta1 = state[..., 2]
        dtheta2 = state[..., 3]

        # Uniform mass distribution parameters
        lc1 = self.l1 / 2  # Center of mass at midpoint
        lc2 = self.l2 / 2  # Center of mass at midpoint
        i1 = (1.0 / 12.0) * self.m1 * self.l1**2  # Moment of inertia
        i2 = (1.0 / 12.0) * self.m2 * self.l2**2  # Moment of inertia

        # Kinetic energy
        # For link 1: T1 = 1/2 * I1 * θ̇₁² + 1/2 * m1 * v_cm1²
        # For link 2: T2 = 1/2 * I2 * θ̇₂² + 1/2 * m2 * v_cm2²

        # Velocity of center of mass of link 1
        v_cm1_x = -lc1 * dtheta1 * jnp.sin(theta1)
        v_cm1_y = lc1 * dtheta1 * jnp.cos(theta1)
        v_cm1_sq = v_cm1_x**2 + v_cm1_y**2

        # Velocity of center of mass of link 2
        # Link 2 absolute angle is (theta1 + theta2)
        # Link 2 absolute angular velocity is (dtheta1 + dtheta2)
        v_cm2_x = -self.l1 * dtheta1 * jnp.sin(theta1) - lc2 * (
            dtheta1 + dtheta2
        ) * jnp.sin(theta1 + theta2)
        v_cm2_y = self.l1 * dtheta1 * jnp.cos(theta1) + lc2 * (
            dtheta1 + dtheta2
        ) * jnp.cos(theta1 + theta2)
        v_cm2_sq = v_cm2_x**2 + v_cm2_y**2

        t1 = 0.5 * i1 * dtheta1**2 + 0.5 * self.m1 * v_cm1_sq
        t2 = 0.5 * i2 * (dtheta1 + dtheta2) ** 2 + 0.5 * self.m2 * v_cm2_sq
        kinetic_energy = t1 + t2

        # Potential energy (taking lowest point as reference)
        # Height of center of mass above reference
        h1 = -lc1 * jnp.cos(theta1)
        h2 = -self.l1 * jnp.cos(theta1) - lc2 * jnp.cos(theta1 + theta2)

        potential_energy = self.m1 * self.g * h1 + self.m2 * self.g * h2

        return kinetic_energy + potential_energy
