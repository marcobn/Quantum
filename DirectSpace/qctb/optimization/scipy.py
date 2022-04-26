import scipy.optimize
from qctb.optimization.optimization import Result, Optimizer, register

class BFGS(Optimizer):
    id = 'BFGS'

    def __init__(self,
        maxiter=None,
        gtol=1e-05,
        norm=float('inf'),
        eps=1.4901161193847656e-08,
    ):
        super().__init__(
            self.id,
            {
                'maxiter': maxiter,
                'gtol': gtol,
                'norm': norm,
                'eps': eps,
            },
        )

    def optimize(self, costfn, x0):
        if len(x0) == 0:    return Result(costfn([]), [], 1)

        result = scipy.optimize.minimize(costfn, x0,
            method='BFGS',
            options=self.parameters,
        )
        return Result(
            E           = result['fun'],
            x           = result['x'],
            nfev        = result['nfev'],
            converged   = result['success'],
        )
register(BFGS)

class COBYLA(Optimizer):
    id = 'COBYLA'

    def __init__(self,
        maxiter=1000,
        rhobeg=1.0,
        tol=1e-05,
        catol=0.0002,
        # TODO: angular constraints?
    ):
        super().__init__(
            self.id,
            {
                'maxiter': maxiter,
                'rhobeg': rhobeg,
                'tol': tol,
                'catol': catol,
            },
        )

    def optimize(self, costfn, x0):
        if len(x0) == 0:    return Result(costfn([]), [], 1)

        result = scipy.optimize.minimize(costfn, x0,
            method='COBYLA',
            tol=self.parameters['tol'],
            options=self.parameters,
        )
        return Result(
            E           = result['fun'],
            x           = result['x'],
            nfev        = result['nfev'],
            converged   = result['success'],
        )
register(COBYLA)
