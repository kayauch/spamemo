"""Geometric wall evidence; no invented pixels or relaxed occlusion."""
import numpy as np


def depth_normals(depths, extrinsics, intrinsics):
    h,w=depths.shape[1:]
    y,x=np.mgrid[:h,:w]
    pixels=np.stack([x,y,np.ones_like(x)],-1)
    output=np.zeros((*depths.shape,3),np.float16)
    for i,(depth,e,k) in enumerate(zip(depths,extrinsics,intrinsics)):
        camera=(pixels@np.linalg.inv(k).T)*depth[...,None]
        world=(camera-e[:,3])@e[:,:3]
        normal=np.cross(world[1:-1,2:]-world[1:-1,:-2],world[2:,1:-1]-world[:-2,1:-1])
        length=np.linalg.norm(normal,axis=2,keepdims=True)
        normal=np.divide(normal,length,out=np.zeros_like(normal),where=length>1e-9)
        output[i,1:-1,1:-1]=np.nan_to_num(normal)
    return output


def matching_normals(normals,uv,normal):
    h,w=normals.shape[:2]
    pixels=np.rint(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)).astype(np.int64)
    valid=np.isfinite(uv).all(1)&(pixels[:,0]>=1)&(pixels[:,0]<w-1)&(pixels[:,1]>=1)&(pixels[:,1]<h-1)
    samples=normals[np.clip(pixels[:,1],0,h-1),np.clip(pixels[:,0],0,w-1)].astype(np.float32)
    return valid&(np.abs(samples@normal)>=.97)


def independent_support(groups,count,project,minimum=3):
    votes=np.zeros(count,np.uint16)
    for group in dict.fromkeys(groups):
        observed=np.zeros(count,bool)
        for i,g in enumerate(groups):
            if g==group:
                mask,score=project(i)
                observed|=np.asarray(mask,bool)&np.isfinite(score)&(score>0)
        votes+=observed
    return votes>=minimum,votes
