import h5py    
import numpy as np   
import pandas as pd

core = h5py.File('results/test/core_test.hdf','r')   


FEATURE_NAMES = ['pdbid','x','y','z','B','C','N','O','P','S','Se','halogen','metal','hyb', 'heavyvalence', 'heterovalence',
            'partialcharge','molcode','hydrophobic', 'aromatic', 'acceptor', 'donor','ring']

df = pd.DataFrame(columns = FEATURE_NAMES)
for pdbid in list(core.keys()):
    data = core[pdbid]['pybel']['processed']['pdbbind']['data']
    n2 = pd.DataFrame(np.array(data),columns=FEATURE_NAMES[1:])
    n2['pdbid'] = pdbid
    df = df.append(n2)

# for pdbid in list(refined_val.keys()):
#     data = refined_val[pdbid]['pybel']['processed']['pdbbind']['data']
#     n2 = pd.DataFrame(np.array(data),columns=FEATURE_NAMES[1:])
#     n2['pdbid'] = pdbid
#     df = df.append(n2)

df.to_csv('../../Core.csv')