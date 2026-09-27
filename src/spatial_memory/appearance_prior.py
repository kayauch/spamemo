"""Use VGGT structural planes to identify conservative diffuse appearance regions."""
import numpy as np


def diffuse_structure_mask(xyz, normals, manifest):
    selected = np.zeros(len(xyz), dtype=bool)
    if not manifest.get('structural_layout', {}).get('enabled'):
        return selected
    raw = {c['id']: c for c in manifest['components']}
    tolerance = manifest['metrics']['scene_diagonal'] * .006
    for component in manifest['structural_layout'].get('components', []):
        # Keep view-dependent floor reflections and furniture appearance intact.
        if component['kind'] not in {'wall', 'ceiling'}:
            continue
        for source in component.get('metadata', {}).get('raw_component_ids', []):
            candidate = raw.get(source)
            if not candidate or 'plane' not in candidate.get('metadata', {}):
                continue
            plane = np.asarray(candidate['metadata']['plane'], dtype=float)
            plane /= max(np.linalg.norm(plane[:3]), 1e-12)
            center = np.array([candidate['center'][key] for key in 'xyz'])
            extent = np.array([candidate['size'][key] for key in 'xy']) / 2
            local = (xyz - center) @ np.asarray(candidate['axes'])
            selected |= ((np.abs(xyz @ plane[:3] + plane[3]) <= tolerance)
                         & (np.abs(local[:, :2]) <= extent).all(axis=1)
                         & (np.abs(normals @ plane[:3]) >= .9))
    return selected
