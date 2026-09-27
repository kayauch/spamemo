"""Extend the same source photographs without inventing or repeating pixels."""
import numpy as np


def horizontal_photo_regions(depths, extrinsics, intrinsics, room_matrix):
    masks=[];h,w=depths.shape[1:];y,x=np.mgrid[:h,:w]
    pixels=np.stack([x,y,np.ones_like(x)],-1)
    for depth,e,k in zip(depths,extrinsics,intrinsics):
        camera=(pixels@np.linalg.inv(k).T)*depth[:,:,None]
        world=(camera-e[:,3])@e[:,:3]
        room=world@room_matrix[:3,:3].T+room_matrix[:3,3]
        normal=np.cross(room[1:-1,2:]-room[1:-1,:-2],room[2:,1:-1]-room[:-2,1:-1])
        length=np.linalg.norm(normal,axis=2)
        horizontal=np.zeros((h,w),bool)
        horizontal[1:-1,1:-1]=np.isfinite(normal).all(2)&(length>1e-9)&(np.abs(normal[:,:,2])>.92*length)
        masks.append(horizontal)
    return np.asarray(masks)


def extend_same_photos(texture, supported, selected, project, sample):
    output=texture.copy();filled=np.zeros(len(supported),bool);sources=np.full(len(supported),-1,np.int32)
    for frame in selected:
        valid,score=project(frame)
        take=np.asarray(valid,bool)&~supported&~filled&np.isfinite(score)&(np.asarray(score)>0)
        if take.any():
            output[take]=np.clip(sample(frame,take),0,255).astype(np.uint8)
            filled[take]=True;sources[take]=frame
    return output,filled,sources
