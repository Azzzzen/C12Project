"""Periodic two/three-body lattice Hamiltonians, with no symmetry projection."""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


def geometry(L, particles):
    """Integer relative coordinates; minimum-image distance; a=t=1."""
    dims = 2 * (particles - 1)
    q = np.indices((L,) * dims, dtype=np.int32).reshape(dims, -1)
    def distance2(x, y):
        x, y = x % L, y % L
        return np.minimum(x, L-x)**2 + np.minimum(y, L-y)**2
    rr = [distance2(q[0], q[1])]
    if particles == 3:
        rr += [distance2(q[2], q[3]), distance2(q[0]-q[2], q[1]-q[3])]
    F = sum(np.exp(-r.astype(float)) for r in rr)
    G = sum(.3 / np.sqrt(r+1.) for r in rr)
    n = L**dims
    index = np.arange(n, dtype=np.int32)
    rows, cols, vals = [index], [index], [np.full(n, 4.*particles)]
    for p in range(particles):
        for axis in range(2):
            for sign in [-1, 1]:
                shifted = q.copy()
                if p == particles-1:
                    for rel in range(particles-1):
                        shifted[2*rel+axis] = (shifted[2*rel+axis]-sign) % L
                else:
                    shifted[2*p+axis] = (shifted[2*p+axis]+sign) % L
                rows.append(index)
                cols.append(np.ravel_multi_index(tuple(shifted), (L,)*dims).astype(np.int32))
                vals.append(np.full(n, -1.))
    T = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n,n)).tocsr()
    T.sum_duplicates()
    return T, F, G


def symmetry_maps(L):
    x,y,u,v = np.indices((L,)*4, dtype=np.int32).reshape(4,-1)
    coords = {
        'P12': (u,v,x,y), 'P13': (-x,-y,u-x,v-y),
        'R90': (-y,x,-v,u), 'Mx': (x,-y,u,-v),
    }
    return {name: np.ravel_multi_index(tuple(z % L for z in c), (L,)*4) for name,c in coords.items()}


def solve(T,F,G,c,k=4,tol=1e-10,seed=20261116):
    H = T + diags(G-c*F)
    rng = np.random.default_rng(seed)
    E, U = eigsh(H, k=k, which='SA', tol=tol, v0=rng.normal(size=H.shape[0]), ncv=max(24,2*k+1))
    order = np.argsort(E)
    E,U = E[order], U[:,order]
    residual = np.linalg.norm(H@U-U*E, axis=0)
    slope = -(U**2).T@F
    return E,U,residual,slope
