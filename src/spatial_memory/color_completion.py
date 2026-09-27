"""Low-frequency colour completion. Never changes photographic support."""
import cv2
import numpy as np


def dominant_color(pixels):
    pixels=np.asarray(pixels,dtype=np.uint8).reshape(-1,3)
    if len(pixels)<32:return None
    # The median avoids choosing a narrow dark shadow as the whole floor colour.
    return np.median(pixels,axis=0).astype(np.float32)


def fill_missing_color(texture,support,region,fallback=None):
    image=np.asarray(texture,np.uint8);support=np.asarray(support,bool);region=np.asarray(region,bool)
    result=image.copy();missing=region&~support
    base=dominant_color(image[support&region])
    if base is None:base=fallback
    if base is None or not missing.any():return result,np.zeros_like(support)
    base=np.asarray(base,np.float32)
    # Extend colour and gentle illumination only, not furniture edges/patterns.
    trusted=support&region&(np.linalg.norm(image.astype(float)-base,axis=2)<110)
    weight=trusted.astype(np.float32)
    sigma=max(3.,max(image.shape[:2])/24)
    denominator=cv2.GaussianBlur(weight,(0,0),sigma)
    sums=cv2.GaussianBlur(np.clip(image.astype(np.float32),base-30,base+30)*weight[:,:,None],(0,0),sigma)
    local=sums/np.maximum(denominator[:,:,None],1e-6)
    influence=np.clip(denominator*12,0,1)[:,:,None]
    color=base*(1-influence)+local*influence
    result[missing]=np.rint(np.clip(color[missing],0,255)).astype(np.uint8)
    return result,missing
