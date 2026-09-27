import numpy as np
from spatial_memory.appearance_prior import diffuse_structure_mask


def test_diffuse_prior_selects_only_supported_wall_and_ceiling_regions():
    wall={'id':'w','metadata':{'plane':[0,0,1,-2]},'center':dict(x=0,y=0,z=2),
          'axes':np.eye(3).tolist(),'size':dict(x=2,y=2)}
    floor={'id':'f','metadata':{'plane':[0,0,1,0]},'center':dict(x=0,y=0,z=0),
           'axes':np.eye(3).tolist(),'size':dict(x=2,y=2)}
    manifest={'metrics':{'scene_diagonal':5},'components':[wall,floor],
              'structural_layout':{'enabled':True,'components':[
                  {'kind':'wall','metadata':{'raw_component_ids':['w']}},
                  {'kind':'floor','metadata':{'raw_component_ids':['f']}}]}}
    xyz=np.array([[0,0,2],[0,0,1.8],[0,0,0],[3,0,2],[0,0,2]])
    normals=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[1,0,0]])
    assert diffuse_structure_mask(xyz,normals,manifest).tolist()==[True,False,False,False,False]
