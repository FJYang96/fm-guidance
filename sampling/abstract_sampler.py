from abc import ABC, abstractmethod

import torch
import torchdiffeq


class Sampler(ABC):
    @abstractmethod
    def forward(self, t, x):
        pass

    def sample(
        self,
        ic: torch.Tensor,
        conditions: torch.Tensor | None,
        ts: float = 10.0,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Sample from a vector field using the ODE solver.
        Args:
            vector_field: The vector field to sample from.
            ic: The initial condition.
            conditions: The conditions to sample from.
            ts: The time step.
            num_steps: The number of steps to take.
        Returns:
            The sampled trajectory.
        """
        self.conditions = conditions
        return torchdiffeq.odeint(
            self.forward,
            ic,
            torch.linspace(0, ts, num_steps, device=ic.device).float(),
            atol=1e-4,
            rtol=1e-4,
            method="rk4",
        )
