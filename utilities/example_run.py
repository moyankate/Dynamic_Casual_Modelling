#### import external libraries ####

import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle

#### import internal libraries and functions ####
from utilities.models.pdcm import pdcm_fmri_priors_new
from utilities.io import loadmat
from utilities.spm.integrate import spm_int_IT
from utilities.plot_utils import PlotGraph
from models.hemodynamic import *
from spm.nlsi import spm_nlsi_GN
warnings.filterwarnings('ignore')

### load data ####
SPM = loadmat('SPM.mat')

Y = {'y':np.zeros((360,3))}
V1 = loadmat('VOI_V1_1.mat')
Y['y'][:,0] = V1['xY']['u'][:,0] # region 0 = V1
V5 = loadmat('VOI_V5_1.mat')
Y['y'][:,1] = V5['xY']['u'][:,0] # region 1 = V5
SPC = loadmat('VOI_SPC_1.mat')
Y['y'][:,2] = SPC['xY']['u'][:,0] # region 2 = SPC
Y['dt'] = SPM['SPM']['xY']['RT']
Y['X0'] = np.concatenate((np.ones((V1['xY']['X0'].shape[0],1)),V1['xY']['X0'][:,1:6]), axis=1)

scale   = np.max(Y['y']) - np.min(Y['y'])
scale   = 4/max(scale,4)
Y['y']     = Y['y']*scale
Y['scale'] = scale

nr = int(Y['y'].shape[1])         # ensure integer
Y['Q'] = np.eye(nr)               # default noise precision

u_idx = [1, 2, 0]
Sess   = SPM['SPM']['Sess']
U = {}
U['name'] = []
U['u'] = np.zeros((5760, 1))
for i in range(0,len(u_idx)):
    u = u_idx[i]
    for j in range(0,1):
        U['u']             = np.concatenate((U['u'], np.expand_dims(Sess['U'][u]['u'][32:,j], 1)), axis=1)
        U['name'].append(Sess['U'][u]['name'][0])

U['u'] = U['u'][:,1:]        
U['dt']   = Sess['U'][0]['dt']
U['u'][U['u']>1] = 1

print (U)

A = np.array([[0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]])
B = np.array([[[0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]]])
    
C = np.array([[0, 0, 1],
          [0, 0, 0],
          [0, 0, 0]])

D = np.array([[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]])
    
pE, pC, x, _ = pdcm_fmri_priors_new(A, B, C, D, {'decay':1})

M = {}
B0      = 3
TE      = 0.04
nr      = Y['y'].shape[1]
M['delays'] = np.ones((1,nr))*Y['dt']/2 
M['TE']    = TE
M['B0']    = B0
M['m']     = nr
M['n']     = 6         
M['l']     = nr
M['N']     = 64
M['dt']    = U['dt']
M['ns']    = Y['y'].shape[0]
M['x']     = x
M['IS']    = spm_int_IT # 'spm.int_IT'
M['f']   = 'fx_fmri_pdcm_new'
M['g']   = 'gx_all_fmri'
M['Tn']  = []                     
M['Tc']  = []
M['Tv']  = []
M['Tm']  = []

n           = nr

M['pE'] = pE
M['pC'] = pC

Ep,Cp,Eh,F,L,dFdp,dFdpp = spm_nlsi_GN(M, U, Y)

y = spm_int_IT(Ep, M, U)

########## store results ############# added by Kate
# Store model results and relevant data
results = {
    'Ep': Ep,
    'Cp': Cp,
    'Eh': Eh,
    'F': F,
    'L': L,
    'dFdp': dFdp,
    'dFdpp': dFdpp,
    'predicted_BOLD': spm_int_IT(Ep, M, U),
    'region_names': ['V1', 'V5', 'SPC'],  # Adjust as needed
}

# Save to file
with open('dcm_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(" Model results saved to dcm_results.pkl")

