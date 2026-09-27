import argparse
import json
from spatial_memory.texturing import build_textured_scene

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project saved video frames onto structural mesh faces')
    parser.add_argument('manifest')
    args = parser.parse_args()
    print(json.dumps(build_textured_scene(args.manifest), indent=2))
