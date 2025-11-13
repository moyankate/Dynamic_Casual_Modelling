from utilities.globs import *
from pydcm.utils import dense, isvector
from utilities import spm
from utilities import wrappers


# TODO: docstring
def pdcm_fmri_priors(A, B, C, D, options):

    pE = {}
    pC = {}

    # number of regions
    # --------------------------------------------------------------------------
    n = A.shape[0]

    # connectivity priors and intitial states
    # ==========================================================================
    # Havlicek 2015, supplementary info 5

    # initial states (6)
    #----------------------------------------------------------------------
    x  = zeros((n, 6))

    # priors for A borrowed from 1S-DCM
    # precision of connections
    # ---------------------------------------------------------------------
    if 'precision' in options:
        pA = exp(options['precision'])
    else:
        pA = 64
    if 'decay' in options:
        dA = options['decay']
    else:
        dA = 1

    # prior expectations
    # ----------------------------------------------------------------------
    if isvector(A):
        A = A.astype(bool)
        pE['A']  = (A.ravel() - 1) * dA
    else:
        A = (A - np.diag(np.diag(A))).astype(bool)
        pE['A']  = A / 128
    pE['B']  = B * 0
    pE['C']  = C * 0
    pE['D']  = D * 0

    # prior covariances
    # ----------------------------------------------------------------------
    if isvector(A):
        pC['A']  = A.ravel()
    else:
        A = atleast_3d(A)
        pC['A'] = zeros(A.shape)
        for i in range(A.shape[2]):
            pC['A'][:, :, i] = A[:, :, i] / pA + eye(n,n) / pA
    pC['B']  = B.astype(bool) * exp(-2)  # = B
    pC['C']  = C.astype(bool) * exp(-1)  # = C
    pC['D']  = D.astype(bool) * exp(-2)  # = D

    # other neuronal priors
    # ----------------------------------------------------------------------
    pE['sigmas']   = zeros((n, 1))
    pC['sigmas']   = zeros((n, 1)) + exp(-4)
    pE['mus']      = zeros((n, 1))
    pC['mus']      = zeros((n, 1)) + exp(-4)
    pE['lambdas']  = zeros((n, 1))
    pC['lambdas']  = zeros((n, 1)) + exp(-4)

    # hemodynamic priors
    # =======================================================================
    pE['transit'] = zeros((n, 1))
    pC['transit'] = zeros((n, 1)) + exp(-4)
    pE['signaldecay'] = zeros((1, 1))
    pC['signaldecay'] = exp(-4)  # not fit?
    # pE['decay'] = zeros((1, 1))
    # pC['decay'] = exp(-4)

    pE['epsilon'] = zeros((1, 1))
    pC['epsilon'] = exp(-6)

    # p-dcm specific
    # ----------------------------------------------------------------------
    pE['gain']      = zeros((1, 1))
    pC['gain']      = exp(-4)
    pE['flowdecay'] = zeros((1, 1))
    pC['flowdecay'] = 1  # not fit
    pE['visco']     = zeros((1, 1))
    pC['visco']     = exp(-2)

    # prior covariance matrix
    # --------------------------------------------------------------------------
    pC = np.diag(spm.vec(pC).ravel())

    return pE, pC, x, spm.vec(pE)



def fx_fmri_pdcm(x, u, P, M=None, nargout=1):
    """
    P-DCM State equation for a dynamic [bilinear/nonlinear/Balloon] model of fMRI
    responses
    FORMAT [f,dfdx,D,dfdu] = spm_fx_fmri(x,u,P,M)
    x      - state vector
    x(:,1) - excitatory neuronal activity            ue
    x(:,2) - vascular signal                          s
    x(:,3) - rCBF                                  ln(f)
    x(:,4) - venous volume                         ln(v)
    x(:,5) - deoyxHb                               ln(q)
    [x(:,6) - inhibitory neuronal activity             ui
    """
    #### Added by Kate to ensure proper matrix multiplication ####
    u = np.asarray(u).reshape(-1)
    expected_len = P['C'].shape[1]
    
    if u.shape[0] > expected_len:
        u = u[:expected_len]  # truncate extra elements if needed
    elif u.shape[0] < expected_len:
        raise ValueError(f"Input vector u has length {u.shape[0]}, expected {expected_len}")

    # neuronal parameters
    #--------------------------------------------------------------------------
    MU = 0.4      # μ, excitatory self connection (Hz)
    SIGMA = 0.5   # σ, inhibitory-excitatory connection (Hz)
    LAMBDA = 0.2  # λ, inhibitory gain factor (Hz)

    # neurovascular coupling parameters
    #--------------------------------------------------------------------------
    PHId  = 0.6   # φ, decay of vasoactive signal(Hz), maybe fixed?
    PHIg  = 1.5   # Φ, gain of vasoactive signal (Hz), maybe fixed?
    CHI   = 0.6   # χ, decay of blood inflow signal (Hz), fixed

    # hemodynamic model parameters
    #--------------------------------------------------------------------------
    MTT   = 2.00  # mean transit time (sec)
    TAU   = 4     # τ, viscoelastic time (sec)
    ALPHA = 0.32  # α, aka Grubb's exp
    E0    = 0.4   # oxygen extraction fraction at rest

    P = copy.deepcopy(P)
    # Neuronal motion
    #==========================================================================
    # matlab: takes full() of A, B and D
    P['A'] = atleast_1d(dense(P['A']))                       # linear parameters
    P['B'] = atleast_1d(dense(P['B']))                       # bi-linear parameters
    P['C'] = atleast_1d(P['C']) / 16                    # exogenous parameters

    n  = P['A'].shape[0]            # number of regions
    uB = zeros((n, n))
    
    V     = [0.6, 1.5, 0.6]
    
    de1   = V[0]*np.ones((n,1))*exp(P['signaldecay'])

    ga = V[1]*np.ones((n,1))*exp(P['gain'])
    
    de2 = V[2]*np.ones((n,1))*exp(P['flowdecay'])
    
    H = [2, 0.32, 3, 3, 6]

    tt = H[0]*exp(P['transit'])

    al = H[1]*np.ones((n,1))

    nr = H[2]*np.ones((n,1))

    ve_in = H[3]*np.ones((n,1))*exp(P['visco'])

    ve_de = H[4]*np.ones((n,1))*exp(P['visco'])

    ve = ve_in

    # implement differential state equation y = dx/dt (neuronal)
    #--------------------------------------------------------------------------
    f = copy.copy(x)
    x = copy.copy(x)

    # two neuronal states per region
    #======================================================================

    # input dependent modulation
    #----------------------------------------------------------------------
    for i in range(P['B'].shape[0]):
        uB = uB + u[i] * P['B'][i,:,:]

    # extrinsic (two neuronal states)
    #----------------------------------------------------------------------

    # P-DCM equations:
    #  d/dt Xe[t] = J[+] * Xe[t] + J[-] * Xi[t] +  C * U[t]
    #  d/dt Xi[t] = 𝔊[Xe[t] - Xi[t]]
    #
    #  J[+]_ij = A + uB
    #  J[-]_ij = 0
    #  G_ij    = 0
    #
    #  J[+]_ii = -σ * exp(σ~ + uB)
    #  J[-]_ii = -μ * exp(μ~_i + Σb_μi * u_μk)
    #  G_ii    =  λ * exp(λ~_i + Σb_λi * u_λl )

    I = eye(n).astype(bool)
    JP = P['A'] + uB
    JN = zeros((n, n))
    G  = zeros((n, n))


    JP[I] = - SIGMA * exp(P['sigmas'].ravel() + np.diag(uB))
    JN[I] = - MU * exp(P['mus'].ravel() + np.diag(uB))
    G[I]  = LAMBDA * exp(P['lambdas'].ravel() + np.diag(uB))

    # motion - excitatory and inhibitory: f = dx/dt
    #----------------------------------------------------------------------
    # d/dt Xe[t] = J[+] *  Xe[t] + J[-] *  Xi[t] +   C * U[t]
    f[:, 0] =  JP  @ x[:, 0] + JN   @ x[:, 5] + P['C'] @ u.ravel()
    # d/dt Xi[t] = G * ( Xe[t] - Xi[t] )
    f[:, 5] =  G @ (x[:,0] - x[:,5])
    
    EE = JP
    IE = -JN
    II = -G
    EI = -G

    # Hemodynamic motion
    #==========================================================================

    # neurovascular coupling and hemodynamic variables
    #--------------------------------------------------------------------------
    #  a[t]: vasoactive signal
    #  f[t]: blood flow
    #  v[t]: blood volume
    #  q[t]: dHb content
    #
    # neurovascular equations
    #--------------------------------------------------------------------------
    #  d/dt a[t] = -φ * a[t] + x[t]
    #  d/dt f[t] =  Φ * a[t] - χ * [f[t] - 1]
    #
    # hemodynamic equations (same as 1s and 2d dcm)
    #--------------------------------------------------------------------------
    #  d/dt v[t] = 1/MTT * [f[t] - fout(v,t)]
    #  d/dt q[t] = 1/MTT * [f[t] * E[f] / E0 - fout(v,t) * q[t]/v[t]]

    # exponentiation of hemodynamic state variables
    #--------------------------------------------------------------------------
    x[:, 2:5] = np.exp(x[:, 2:5])  # f, v, q

    # scale variables
    #--------------------------------------------------------------------------
    sd  = PHId * exp(P['signaldecay']).ravel()  # φ signal decay
    sg  = PHIg * exp(P['gain']).ravel()         # Φ signal gain
    #fd  = CHI * exp(P['flowdecay']).ravel()    # χ flow decay
    fd  = CHI                        # not fit (suppl. info 5)
    tt  = MTT * exp(P['transit']).ravel()       # transit time, fit (suppl. info 5)
    vt  = TAU * exp(P['visco']).ravel()         # τ viscoelastic time, fit

    # Fout = f[v] - outflow             fout(v,t)
    #--------------------------------------------------------------------------
    # P-DCM includes a viscoelastic effect
    fv = (tt *  x[:, 3] ** (1 / ALPHA) + vt * x[:, 2]) / (vt + tt)

    # e = f[f] - oxygen extraction      E[f]/E0
    #--------------------------------------------------------------------------
    ff = (1 - (1 - E0) ** (1 / x[:, 2])) / E0

    # a[t]: vasoactive signal
    #--------------------------------------------------------------------------
    f[:, 1] = - sd * x[:, 1] + x[:, 0]

    # f[t]: flow  (log units)
    #--------------------------------------------------------------------------
    f[:, 2] = (sg * x[:, 1] - fd * (x[:, 2] - 1)) / x[:, 2]

    # v[t]: blood volume  (log units)
    #--------------------------------------------------------------------------
    f[:, 3] =  (x[:, 2] - fv) / (tt * x[:, 3])

    # q[t]: dHB content  (log units)
    #--------------------------------------------------------------------------
    f[:, 4] = (ff  * x[:, 2] - fv  * x[:, 4] / x[:, 3]) / (tt * x[:, 4])

    #import pdb; pdb.set_trace()

    f = f.ravel(order='F')
    
    dfdx = [[np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))]]
    
    dfdx[0][0] = EE
#        for i = 1:size(D,3)
#            Di  = D(:,:,i) + diag((diag(EE) - 1).*diag(D(:,:,i)));
#            dfdx{1,1}(:,i) = dfdx{1,1}(:,i) + Di*x(:,1);
#        end
    dfdx[0][5] = -IE
    dfdx[1][0] = spm.speye(n,n)
    dfdx[1][1] = spm.diag(-de1[:,0])
    dfdx[2][1] = spm.diag(ga[:,0]/x[:,2])
    dfdx[2][2] = spm.diag(-(de2[:,0]+ga[:,0]*x[:,1])/(x[:,2]**2))
    dfdx[3][2] = spm.diag(1/(x[:,3]*(tt+ve[:,0])))
    dfdx[3][3] = spm.diag(-(al[:,0]*x[:,2]+(x[:,3]**(1/al[:,0]))*(1-al[:,0]))/(al[:,0]*(x[:,3]**2)*(tt + ve[:,0])))
    dfdx[4][2] = spm.diag((1/nr[:,0] - (ve[:,0]*x[:,4])/(x[:,3]*(tt + ve[:,0])))/(tt*x[:,4]))
    dfdx[4][3] = spm.diag(((al[:,0]*ve[:,0]*x[:,2] - tt*x[:,3]**(1/al[:,0]) + al[:,0]*tt*x[:,3]**(1/al[:,0])))/(al[:,0]*tt*(x[:,3]**2)*(tt + ve[:,0])))
    dfdx[4][4] = spm.diag(-(nr[:,0] + x[:,2] -1)/(nr[:,0]*tt*x[:,4]**2))
    dfdx[5][0] = EI
    dfdx[5][5] = -II
    
#    dfdx_merged = dfdx[0][0]
#    for jte in range(1,6):
#        dfdx_merged = np.concatenate((dfdx_merged,dfdx[0][jte]),axis=1)
#    for ite in range(1,6):
#        dfdx_dummy = dfdx[ite][0]
#        for jte in range(1,6):
#            print(ite, jte, dfdx[ite][jte].shape)
#            dfdx_dummy = np.concatenate((dfdx_dummy,dfdx[ite][jte]),axis=1)
#        dfdx_merged = np.concatenate((dfdx_merged, dfdx_dummy), axis=0)
#    
#    print(dfdx_merged.shape)
    
    dfdx = spm.cat(dfdx)
    

    if nargout == 1:
        return f
    else:
        return f, dfdx

# TODO: docstring
def pdcm_fmri_priors_new(A, B, C, D, options):

    pE = {}
    pC = {}

    # number of regions
    # --------------------------------------------------------------------------
    n = A.shape[0]

    # connectivity priors and intitial states
    # ==========================================================================
    # Havlicek 2015, supplementary info 5

    # initial states (6)
    #----------------------------------------------------------------------
    x  = zeros((n, 6))

    # priors for A borrowed from 1S-DCM
    # precision of connections
    # ---------------------------------------------------------------------
    if 'precision' in options:
        pA = exp(options['precision'])
    else:
        pA = 64
    if 'decay' in options:
        dA = options['decay']
    else:
        dA = 1

    # prior expectations
    # ----------------------------------------------------------------------
    if isvector(A):
        A = A.astype(bool)
        pE['A']  = (A.ravel() - 1) * dA
    else:
        A = (A - np.diag(np.diag(A))).astype(bool)
        pE['A']  = A / 128
    pE['A']  = A * 0
    pE['B']  = B * 0
    pE['C']  = C * 0
    pE['D']  = D * 0

    # prior covariances
    # ----------------------------------------------------------------------
#    if isvector(A):
#        pC['A']  = A.ravel()
#    else:
#        A = atleast_3d(A)
#        pC['A'] = zeros(A.shape)
#        for i in range(A.shape[2]):
#            pC['A'][:, :, i] = A[:, :, i] / pA + eye(n,n) / pA
    pC['A']  = A.astype(bool) / pA
    pC['B']  = B.astype(bool) * exp(0)  # = B
    pC['C']  = C.astype(bool) * exp(0)  # = C
    pC['D']  = D.astype(bool) * exp(-2) * 0  # = D

    # other neuronal priors
    # ----------------------------------------------------------------------
    pE['sigmas']   = zeros((1, 1))
    pC['sigmas']   = ones((1, 1))[0,0] * exp(-1)
    pE['mus']      = zeros((n, 1))
    pC['mus']      = ones((n, 1)) * exp(-2)
    pE['lambdas']  = zeros((n, 1))
    pC['lambdas']  = ones((n, 1)) * exp(-2)

    # hemodynamic priors
    # =======================================================================
    pE['transit'] = zeros((n, 1))
    pC['transit'] = ones((n, 1)) * exp(-4)

    pE['epsilon'] = zeros((1, 1))
    pC['epsilon'] = exp(-6)

    
    # pE['signaldecay'] = zeros((1, 1))
    # pC['signaldecay'] = exp(-4)  # not fit?
    # pE['decay'] = zeros((1, 1))
    # pC['decay'] = exp(-4)

    # p-dcm specific
    # ----------------------------------------------------------------------
    pE['gain']      = zeros((1, 1))
    pC['gain']      = 0 * exp(-4)
    pE['flowdecay'] = zeros((1, 1))
    pC['flowdecay'] = 0 * 1  # not fit
    pE['visco_in']     = zeros((n, 1))
    pC['visco_in']     = ones((n,1)) * exp(-1)
    pE['visco_de']     = zeros((n, 1))
    pC['visco_de']     = ones((n,1)) * exp(-1)

    # prior covariance matrix
    # --------------------------------------------------------------------------
    pC = np.diag(spm.vec(pC).ravel())

    return pE, pC, x, spm.vec(pE)
    
def fx_fmri_pdcm_new(x, u, P, M=None, nargout=1):
    """
    P-DCM State equation for a dynamic [bilinear/nonlinear/Balloon] model of fMRI
    responses
    FORMAT [f,dfdx,D,dfdu] = spm_fx_fmri(x,u,P,M)
    x      - state vector
    x(:,1) - excitatory neuronal activity            ue
    x(:,2) - vascular signal                          s
    x(:,3) - rCBF                                  ln(f)
    x(:,4) - venous volume                         ln(v)
    x(:,5) - deoyxHb                               ln(q)
    [x(:,6) - inhibitory neuronal activity             ui
    """

    P = copy.deepcopy(P)
    # Neuronal motion
    #==========================================================================
    # matlab: takes full() of A, B and D
    A = atleast_1d(dense(P['A']))                       # linear parameters
    B = atleast_1d(dense(P['B']))                       # bi-linear parameters
    C = atleast_1d(P['C']) / 16                    # exogenous parameters

    n  = A.shape[0]            # number of regions
    
    Tn = ones((n,1))
    Tn = ones((n,1))
    Tn = ones((n,1))
    Tn = ones((n,1))
    
    # Local neuronal parameters:
    #==========================================================================
    #   N(1) - inhibitory-excitatory connection (IE)            mu     (Hz)
    #   N(2) - inhibitory gain  (EI and II)                     lambda (Hz)
    #--------------------------------------------------------------------------
    N     = [0.8, 0.2]
    try:
        sigma = 0.5*exp(P['sigmas'])[0,0]
    except:
        sigma = 0.5*exp(P['sigmas'])
    A     = A - np.diagflat(np.diag(A)) - np.diagflat(sigma*exp(np.diag(A)))
    
    nb = B.shape[0]
    for i in range(nb):
        A = A + u[i]*B[i,:,:]   
    
    EE    = A
    mu    = P['mus']
    lam   = P['lambdas']
    
    IE    = np.diagflat(sigma*N[0]*exp(mu)) # global scaling by sigma
    EI    = np.diagflat(N[1]*exp(lam))
    II    = EI
    
    # uB = zeros((n, n))
    
    V     = [0.6, 1.5, 0.6]
    
    de1   = V[0]*np.ones((n,1))

    ga = V[1]*np.ones((n,1))*exp(P['gain'])
    
    de2 = V[2]*np.ones((n,1))*exp(P['flowdecay'])
    
    H = [2, 0.35, 3, 3, 6]

    tt = H[0]*exp(P['transit'])

    al = H[1]*np.ones((n,1))

    nr = H[2]*np.ones((n,1))

    ve_in = H[3]*exp(P['visco_in'])

    ve_de = H[4]*exp(P['visco_de'])

    ve = ve_in

    # implement differential state equation y = dx/dt (neuronal)
    #--------------------------------------------------------------------------
    f = copy.copy(x)
    x = copy.copy(x)
    
    x[:, 2:5] = np.exp(x[:, 2:5])  # f, v, q
    
    # f[:, 0] =  EE @ x[:,0]  - IE @ x[:,5] + C @ u.ravel()
    u_mod = u[:3].reshape(-1)  # ensures it's (3,) # !!!
    f[:, 0] = EE @ x[:,0] - IE @ x[:,5] + C @ u_mod # !!!

    f[:, 5] = -II @ x[:,5] + EI @ x[:,0]

    # m = (f-1 + nr)/nr - oxygen metabolism
    # --------------------------------------------------------------------------
    m  = (x[:,2]-1+nr[:,0])/nr[:,0]
    # implement differential state equation y = dx/dt (hemodynamic)
    # --------------------------------------------------------------------------
    f[:,1]   =  x[:,0] - de1[:,0]*(x[:,1])
    
    dfin     = (ga[:,0]*x[:,1] - de2[:,0]*(x[:,2]-1))
    
    f[:,2]   =  dfin/x[:,2]
    
    # simple test for inflation and deflation
    fv_de    = (tt[:,0]*x[:,3]**(1/al[:,0]) + ve_de[:,0]*x[:,2])/(tt[:,0]+ve_de[:,0])
    
    dv_de = (x[:,2] - fv_de)/tt[:,0]
    
    ve[dv_de<0]  = ve_de[dv_de<0]
    
    fv    = (tt[:,0]*x[:,3]**(1/al[:,0]) + ve[:,0]*x[:,2])/(tt[:,0]+ve[:,0])
    
    f[:,3]   = (x[:,2] - fv)/(tt[:,0]*x[:,3])
    
    f[:,4]   = (m - fv*x[:,4]/x[:,3])/(tt[:,0]*x[:,4])
    
    f = f.ravel(order='F')

    #import pdb; pdb.set_trace()
    
#    dfdx = [[np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
#             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
#             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
#             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
#             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
#             [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))]]
    
    dfdx = []
    for ix in range(6):
        dfdy = []
        for jx in range(6):
            dfdy.append(np.zeros((n,n)))
        dfdx.append(dfdy)
    
    dfdx[0][0] = EE
#        for i = 1:size(D,3)
#            Di  = D(:,:,i) + diag((diag(EE) - 1).*diag(D(:,:,i)));
#            dfdx{1,1}(:,i) = dfdx{1,1}(:,i) + Di*x(:,1);
#        end
    dfdx[0][5] = -IE
    dfdx[1][0] = spm.speye(n,n)
    dfdx[1][1] = spm.diag(-de1[:,0])
    dfdx[2][1] = spm.diag(ga[:,0]/x[:,2])
    dfdx[2][2] = spm.diag(-(de2[:,0]+ga[:,0]*x[:,1])/(x[:,2]**2))
    dfdx[3][2] = spm.diag(1/(x[:,3]*(tt[:,0]+ve[:,0])))
    dfdx[3][3] = spm.diag(-(al[:,0]*x[:,2]+(x[:,3]**(1/al[:,0]))*(1-al[:,0]))/(al[:,0]*(x[:,3]**2)*(tt[:,0] + ve[:,0])))
    dfdx[4][2] = spm.diag((1/nr[:,0] - (ve[:,0]*x[:,4])/(x[:,3]*(tt[:,0] + ve[:,0])))/(tt[:,0]*x[:,4]))
    dfdx[4][3] = spm.diag(((al[:,0]*ve[:,0]*x[:,2] - tt[:,0]*x[:,3]**(1/al[:,0]) + al[:,0]*tt[:,0]*x[:,3]**(1/al[:,0])))/(al[:,0]*tt[:,0]*(x[:,3]**2)*(tt[:,0] + ve[:,0])))
    dfdx[4][4] = spm.diag(-(nr[:,0] + x[:,2] -1)/(nr[:,0]*tt[:,0]*x[:,4]**2))
    dfdx[5][0] = EI
    dfdx[5][5] = -II
    
#    dfdx_merged = dfdx[0][0]
#    for jte in range(1,6):
#        dfdx_merged = np.concatenate((dfdx_merged,dfdx[0][jte]),axis=1)
#    for ite in range(1,6):
#        dfdx_dummy = dfdx[ite][0]
#        for jte in range(1,6):
#            print(ite, jte, dfdx[ite][jte].shape)
#            dfdx_dummy = np.concatenate((dfdx_dummy,dfdx[ite][jte]),axis=1)
#        dfdx_merged = np.concatenate((dfdx_merged, dfdx_dummy), axis=0)
#    
#    print(dfdx_merged.shape)
    
    dfdx = spm.cat(dfdx)
    

    if nargout == 1:
        return f
    else:
        return f, dfdx
        
