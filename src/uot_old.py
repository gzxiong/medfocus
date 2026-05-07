import torch
from typing import Optional, Tuple

import torch.nn as nn


class UnbalancedOptimalTransport(nn.Module):
    """
    Unbalanced Optimal Transport (UOT) with GPU support.
    Uses Sinkhorn iterations with KL divergence relaxation.
    """
    
    def __init__(
        self,
        reg_m: float = 1.0,
        reg_e: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            reg_m: Marginal relaxation parameter (KL divergence)
            reg_e: Entropic regularization parameter
            max_iter: Maximum number of Sinkhorn iterations
            tol: Convergence tolerance
            device: Device to run computations on
        """
        super().__init__()
        self.reg_m = reg_m
        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
    
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        M: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute unbalanced optimal transport between distributions a and b.
        
        Args:
            a: Source distribution (n,)
            b: Target distribution (m,)
            M: Cost matrix (n, m)
        
        Returns:
            Transport plan, dual variable u, dual variable v
        """
        a = a.to(self.device)
        b = b.to(self.device)
        M = M.to(self.device)
        
        n, m = M.shape
        
        # Initialize dual variables
        u = torch.ones(n, device=self.device)
        v = torch.ones(m, device=self.device)
        
        # Kernel matrix
        K = torch.exp(-M / self.reg_e)
        
        # Sinkhorn iterations
        for iter_num in range(self.max_iter):
            u_prev = u.clone()
            
            # Update v
            Ktu = K.t() @ u
            v = (b / (Ktu + 1e-16)) ** (self.reg_e / (self.reg_e + self.reg_m))
            
            # Update u
            Kv = K @ v
            u = (a / (Kv + 1e-16)) ** (self.reg_e / (self.reg_e + self.reg_m))
            
            # Check convergence
            err = torch.norm(u - u_prev, p=float('inf'))
            if err < self.tol:
                break
        
        # Compute transport plan
        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        return P, u, v
    
    def transport_cost(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the unbalanced optimal transport cost.
        
        Args:
            a: Source distribution (n,)
            b: Target distribution (m,)
            M: Cost matrix (n, m)
        
        Returns:
            Transport cost (scalar)
        """
        P, _, _ = self.forward(a, b, M)
        cost = torch.sum(P * M)
        return cost


def unbalanced_sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    reg_m: float = 1.0,
    reg_e: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional interface for unbalanced optimal transport.
    
    Args:
        a: Source distribution (n,)
        b: Target distribution (m,)
        M: Cost matrix (n, m)
        reg_m: Marginal relaxation parameter
        reg_e: Entropic regularization parameter
        max_iter: Maximum iterations
        tol: Convergence tolerance
        device: Device for computation
    
    Returns:
        Transport plan P and transport cost
    """
    uot = UnbalancedOptimalTransport(reg_m, reg_e, max_iter, tol, device)
    P, _, _ = uot(a, b, M)
    cost = torch.sum(P * M)
    return P, cost