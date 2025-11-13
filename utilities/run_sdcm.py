#### new! modified by Kate ####
#### this code runs the sdcm model ####

#### import external libraries ####
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import os
import pandas as pd
from datetime import datetime

#### import internal libraries and functions ####
from utilities.spm.models.sdcm import spm_dcm_fmri_priors
from utilities.spm.models.sdcm import spm_fx_fmri
from utilities.io import loadmat
from utilities.spm.integrate import spm_int_IT
from utilities.plot_utils import PlotGraph
from models.hemodynamic import *
from spm.nlsi import spm_nlsi_GN
warnings.filterwarnings('ignore')

#### U is a dictionary for stimulus timeseries, different for each experiment ####
#### Load U from DCM_blue.mat ####
DCM = loadmat('U.mat')['DCM']
U = DCM['U']
U['dt'] = float(U['dt'])  # Ensure dt is a float
U['u'] = U['u'].toarray() if hasattr(U['u'], 'toarray') else U['u']  # Convert sparse to dense if needed
U['name'] = [str(n[0]) if isinstance(n, (list, np.ndarray)) else str(n) for n in U['name']]  # Clean names

#### load data ####
data = loadmat('PatientData.mat')
Yall = data['Yall']

#### Change this index to select a different subject ####
subject_idx = 5

#### Extract time series for this subject's ROIs #### 
n_rois = 9 # Yall.shape[1]
roi_series = [Yall[subject_idx, i].squeeze() for i in range(n_rois)]
min_len = min(len(ts) for ts in roi_series)
roi_series = [ts[:min_len] for ts in roi_series]
Ymat = np.stack(roi_series, axis=1)

print(f"\nAnalyzing Subject {subject_idx}")
print(f"Shape of extracted time series: {Ymat.shape}")
print(f"Preview of first 5 timepoints across {n_rois} ROIs:")
print(Ymat[:5, :])
print("============================================\n")

Y = {}
Y['y'] = Ymat
Y['dt'] = U['dt']
Y['X0'] = np.ones((Ymat.shape[0], 1))  

scale = np.max(Y['y']) - np.min(Y['y'])
scale = 4 / max(scale, 4)
Y['y'] = Y['y'] * scale
Y['scale'] = scale

nr = int(Y['y'].shape[1])
Y['Q'] = np.eye(nr)

#### A matrix is intrinsic connectivity ####
A = np.array([
    [0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 0]
    ])

#### B matrix describe how experimental conditions change the connections ####
#### Your number of B matrix need to match the number of stimulus in you experimental setup ####
#### For example, 2 stimulus (1 visual 1 auditory), you will need 2 B matrices ####
B = np.zeros((9, 9, 2))  

B[:, :, 0] = np.array([ 
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

B[:, :, 1] = np.array([  
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])
   
#### C matrix tells which region were directly activated by the stimulus ####
#### Should be n_regions x n_stimulus #### 
C = np.zeros((9, 2))
C[5, 0] = 1  # IOG-L
C[6, 0] = 1  # IOG-R

#### D matrix are not used in this model ####
#### But it normally descirbes nonlinear modulation ####
D = np.zeros((9, 9))

# Sanity checks for A/B/C/D and U
n_inputs = U['u'].shape[1]
assert A.shape == (n_rois, n_rois)
# assert B.shape == (n_rois, n_rois, n_inputs)
assert C.shape == (n_rois, n_inputs)
assert D.shape == (n_rois, n_rois)

pE, pC, x = spm_dcm_fmri_priors(A, B, C, D, {'decay': 1})

M = {}
B0 = 3
TE = 0.04

nr = Y['y'].shape[1]
M['delays'] = np.ones((1, nr)) * Y['dt'] / 2
M['TE'] = TE
M['B0'] = B0
M['m'] = nr
M['n'] = 6
M['l'] = nr
M['N'] = 64
M['dt'] = U['dt']
M['ns'] = Y['y'].shape[0]
M['x'] = x
M['IS'] = spm_int_IT
M['f'] = spm_fx_fmri
M['g'] = gx_all_fmri
M['Tn'] = []
M['Tc'] = []
M['Tv'] = []
M['Tm'] = []

M['pE'] = pE
M['pC'] = pC

Ep, Cp, Eh, F, L, dFdp, dFdpp = spm_nlsi_GN(M, U, Y)
y = spm_int_IT(Ep, M, U)

# clean U (stimulus time course) before saving data, since it is matlab
U_clean = {
    'u': U['u'],
    'dt': U['dt'],
    'name': [str(n) for n in U['name']]  # Convert names to plain Python strings
}

results = {
    'Ep': Ep,
    'Cp': Cp,
    'Eh': Eh,
    'F': F,
    'L': L,
    'dFdp': dFdp,
    'dFdpp': dFdpp,
    'predicted_BOLD': y,
    'actual_BOLD': Y['y'],  # Added
    'region_names': ['ACC-R', 'AI-L', 'AI-R', 'dlPFC-R', 'mPFC-R', 'IOG-L', 'IOG-R', 'thal-R', 'VTA-R'],
    'U': U  # Added
}

with open(f'sdcm_results_subject{subject_idx}.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"Model results saved to sdcm_results_subject{subject_idx}.pkl")

#### Store F values for later model comparison ####
f_filename = 'sdcm_F_values.xlsx'
model_name = 's_dcm'  

num_timepoints = Y['y'].shape[0]
F_normalized = F / num_timepoints

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

new_row = {
    'subject_idx': subject_idx,
    'model': model_name,
    'F_value_raw': F,
    'num_timepoints': num_timepoints,
    'F_value_normalized' : F_normalized,
    'timestamp': timestamp
}

if os.path.exists(f_filename):
    df_existing = pd.read_excel(f_filename)
    df_updated = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
else:
    df_updated = pd.DataFrame([new_row])

df_updated.to_excel(f_filename, index=False)
print(f"Appended F value for subject {subject_idx} to {f_filename}")
