"""Spherical Galerkin + conservative pressure flux inversion.

Formal global Gan operator; not a replacement for a globally valid balance theory.
No uniform forcing mean is removed. A constant solution gauge is fixed.
"""
import numpy as np
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import splu

RADIUS, OMEGA, G, RD = 6371200., 7.292e-5, 9.80665, 287.


def need(ok, message):
    if not ok:
        raise ValueError(message)


def basis(mu, m, truncation):
    """Associated Legendre functions normalized to integral B_n^2 dmu=1."""
    mu = np.asarray(mu, float)
    bmm = np.ones_like(mu)/np.sqrt(2.)
    for j in range(1,m+1):
        bmm *= -np.sqrt((2*j+1)/(2*j))*np.sqrt(np.maximum(0.,1-mu*mu))
    result = np.empty((len(mu),truncation-m+1))
    result[:,0] = bmm
    if truncation > m:
        result[:,1] = np.sqrt(2*m+3)*mu*bmm
    for n in range(m+2,truncation+1):
        a = np.sqrt((4*n*n-1)/(n*n-m*m))
        c = np.sqrt((2*n+1)*((n-1)**2-m*m)/((2*n-3)*(n*n-m*m)))
        result[:,n-m] = a*mu*result[:,n-m-1]-c*result[:,n-m-2]
    return result


class Solver:
    def __init__(self,p,mu,weights,sigma,truncation):
        self.p,self.mu,self.w,self.sigma = [np.asarray(x,float) for x in (p,mu,weights,sigma)]
        self.truncation = truncation
        need(len(p)>=3 and np.all(np.diff(p)>0),'Pressure must increase in Pa')
        need(sigma.shape==(len(p),len(mu)) and np.isfinite(sigma).all() and (sigma>0).all(),
             'Invalid global stability; do not clip')
        dp = np.diff(p)
        self.width = np.r_[dp[0]/2,(dp[:-1]+dp[1:])/2,dp[-1]/2]
        self.f2 = (2*OMEGA*self.mu)**2
        self.conduct = 2/(self.sigma[:-1]+self.sigma[1:])

    def heat_source(self,h):
        face = np.concatenate([h[:1],.5*(h[:-1]+h[1:]),h[-1:]],axis=0)
        return self.f2[None,:,None]/G*np.diff(face,axis=0)/self.width[:,None,None]

    def solve_many(self,sources,boundaries,output_lat,output_nlon,tol=1e-8):
        names = list(sources)
        nlon = next(iter(sources.values())).shape[-1]
        nk = len(self.p)
        rhs = {}
        for name in names:
            src = np.asarray(sources[name],float)
            need(src.shape==(nk,len(self.mu),nlon) and np.isfinite(src).all(),'Invalid source')
            gt,gb = boundaries[name]
            b = -RADIUS**2*self.width[:,None,None]*src
            b[0] -= RADIUS**2*self.f2[:,None]*gt/self.sigma[0,:,None]
            b[-1] += RADIUS**2*self.f2[:,None]*gb/self.sigma[-1,:,None]
            rhs[name] = np.fft.rfft(b,axis=-1)/nlon
        outmu = np.sin(np.deg2rad(output_lat))
        out = {name:np.zeros((nk,len(outmu),output_nlon//2+1),complex) for name in names}
        diagnostics = {name:dict(compatibility_relative=None,original_Galerkin_residual_relative=0.,
                                gauge_lagrange_multiplier=0.,forcing_mean_removed=False) for name in names}
        for m in range(self.truncation+1):
            B = basis(self.mu,m,self.truncation)
            gram = B.T@(self.w[:,None]*B)
            need(np.max(np.abs(gram-np.eye(len(gram))))<1e-10,'Spherical basis orthogonality failed')
            bo = basis(outmu,m,self.truncation)
            nb = B.shape[1]
            ell = np.arange(m,self.truncation+1)
            stiffness = []
            for k in range(nk-1):
                coeff = RADIUS**2*self.f2*self.conduct[k]/(self.p[k+1]-self.p[k])
                stiffness.append(csc_matrix(B.T@((self.w*coeff)[:,None]*B)))
            blocks = [[None]*nk for _ in range(nk)]
            for k in range(nk):
                diag = diags(self.width[k]*ell*(ell+1),format='csc')
                if k:diag = diag+stiffness[k-1];blocks[k][k-1] = -stiffness[k-1]
                if k<nk-1:diag = diag+stiffness[k];blocks[k][k+1] = -stiffness[k]
                blocks[k][k] = diag
            A = bmat(blocks,format='csc')
            bs = np.stack([(rhs[name][...,m]*self.w[None,:])@B for name in names],axis=-1).reshape(nk*nb,len(names))
            if m==0:
                null = np.zeros(nk*nb);null[::nb] = 1.
                gauge = np.zeros_like(null);gauge[::nb] = self.width/self.width.sum()
                need(np.max(np.abs(A@null))<1e-12*max(np.abs(A.data).max(),1.),'Constant nullspace failed')
                for i,name in enumerate(names):
                    compatibility = float(abs(null@bs[:,i])/max(
                        np.sum(abs(bs[:,i])),float(np.abs(rhs[name]).max())*nk,1e-300))
                    diagnostics[name]['compatibility_relative'] = compatibility
                    print(f'{name} COMPATIBILITY={compatibility:.3e}',flush=True)
                    need(compatibility<=tol,
                         f'{name}: GLOBAL_COMPATIBILITY_FAILED={compatibility:.3e}; no mean removed')
                aug = bmat([[A,csc_matrix(null[:,None])],[csc_matrix(gauge[None,:]),csc_matrix((1,1))]],format='csc')
                lu = splu(aug)
                r = np.vstack([bs,np.zeros((1,len(names)))])
                sol = lu.solve(r.real)+1j*lu.solve(r.imag)
                x = sol[:-1]
                for i,name in enumerate(names):
                    diagnostics[name]['gauge_lagrange_multiplier'] = float(abs(sol[-1,i]))
            else:
                lu = splu(A);x = lu.solve(bs.real)+1j*lu.solve(bs.imag)
            for i,name in enumerate(names):
                residual = float(np.max(np.abs(A@x[:,i]-bs[:,i]))/max(np.max(np.abs(bs[:,i])),1e-300))
                # Nearly absent high-wave-number sources use a global spectral RHS scale.
                scale = max(float(np.abs(rhs[name]).max()),1e-300)
                residual_global = float(np.max(np.abs(A@x[:,i]-bs[:,i]))/scale)
                diagnostics[name]['original_Galerkin_residual_relative'] = max(
                    diagnostics[name]['original_Galerkin_residual_relative'],residual_global)
                need(residual_global<=tol,f'{name}: original equation residual failed: {residual_global:g}')
                out[name][...,m] = x[:,i].reshape(nk,nb)@bo.T
        fields = {name:np.fft.irfft(modes*output_nlon,n=output_nlon,axis=-1) for name,modes in out.items()}
        return fields,diagnostics


def self_test():
    from numpy.polynomial.legendre import leggauss
    mu,w = leggauss(49)
    for m in range(17):
        B = basis(mu,m,16)
        np.testing.assert_allclose(B.T@(w[:,None]*B),np.eye(B.shape[1]),atol=5e-13)
    errors = []
    for nk in (11,21,41):
        p = np.linspace(10000.,100000.,nk)
        phi = np.arcsin(mu)
        lon = np.arange(64)*2*np.pi/64
        spatial = np.cos(phi)[:,None]*np.cos(lon)[None,:]+.3*mu[:,None]
        sigma = 1e-6*(1+.3*p[:,None]/1e5)*(1+.2*mu[None,:]**2)
        exact = (p[:,None,None]/1e5)**2*spatial[None]
        xp = 2*p[:,None,None]/1e10*spatial[None]
        xpp = 2/1e10*spatial[None]
        sp = 1e-6*.3/1e5*(1+.2*mu**2)
        source = -2/RADIUS**2*exact+(2*OMEGA*mu[None,:,None])**2*(
            xpp/sigma[:,:,None]-xp*sp[None,:,None]/sigma[:,:,None]**2)
        op = Solver(p,mu,w,sigma,16)
        result,diag = op.solve_many({'analytic':source},{'analytic':(xp[0],xp[-1])},np.rad2deg(phi),64)
        delta = result['analytic']-exact
        err = np.sqrt(np.sum(op.width[:,None]*w[None,:]*np.mean(delta**2,axis=-1))/
                      np.sum(op.width[:,None]*w[None,:]*np.mean(exact**2,axis=-1)))
        errors.append(float(err))
        print(f'ANALYTIC nk={nk} relative_RMS={err:.6e}',flush=True)
    need(errors[0]/errors[1]>3.5 and errors[1]/errors[2]>3.5,'Second-order pressure convergence failed')
    zero = np.zeros((len(mu),64))
    try:
        op.solve_many({'bad':np.ones_like(source)},{'bad':(zero,zero)},np.rad2deg(phi),64)
    except ValueError as exc:
        need('GLOBAL_COMPATIBILITY_FAILED' in str(exc),'Wrong rejection')
    else:raise AssertionError('Incompatible forcing was accepted')
    # Conservative thermal source must match Neumann heat flux exactly.
    h = (1+.2*mu[None,:,None]+.3*np.cos(lon)[None,None,:])*np.ones_like(source)*(p[:,None,None]/1e5)
    sh = op.heat_source(h)
    gt = sigma[0,:,None]*h[0]/G;gb = sigma[-1,:,None]*h[-1]/G
    sv = -2/RADIUS**2*spatial[None]*(p[:,None,None]/1e5)
    field,diag = op.solve_many({'heat':sh,'vort':sv,'total':sh+sv},
                              {'heat':(gt,gb),'vort':(zero,zero),'total':(gt,gb)},np.rad2deg(phi),64)
    delta = np.abs(field['total']-field['heat']-field['vort']).max()
    need(delta/(np.abs(field['heat']).max()+np.abs(field['vort']).max())<1e-10,'Additivity failed')
    print('PASS: basis, variable-stability manufactured solution, second-order convergence, incompatibility rejection, heat-flux boundary consistency, independent additivity')
