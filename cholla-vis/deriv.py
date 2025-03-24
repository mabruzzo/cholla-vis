import numpy as np

def grid_deriv(y, dx = 1.0, nominal_order = 2, axis = -1):
    """
    Estimates the derivative on uniformly spaced grid along an axis
    """
    deriv = np.empty_like(y, subok = True)
    ndim = np.ndim(y)

    class IdxBuilder:
        def __getitem__(self,ind):
            tmp = [slice(None) for i in range(ndim)]
            tmp[axis] = ind
            return tuple(tmp)
    idx = IdxBuilder()

    assert nominal_order == 2

    # don't divide by dx until later
    deriv[idx[1:-1]] = -0.5*y[idx[:-2]] + 0.5*y[idx[2:]]                # centered diff
    deriv[idx[0]]    = -1.5*y[idx[0]]   + 2*y[idx[1]]  - 0.5*y[idx[2]]  # forward diff
    deriv[idx[-1]]   =  0.5*y[idx[-3]]  - 2*y[idx[-2]] + 1.5*y[idx[-1]] # backwards diff
    deriv /= dx

    return deriv