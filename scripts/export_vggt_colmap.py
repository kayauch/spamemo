import argparse
import json
from spatial_memory.colmap_export import export_scene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--max-points', type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps(export_scene(args.scene_dir, args.output_dir, args.max_points), indent=2))
