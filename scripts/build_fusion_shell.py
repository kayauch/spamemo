import argparse
import json
from spatial_memory.fusion import build_fusion_shell

parser = argparse.ArgumentParser()
parser.add_argument('manifest')
args = parser.parse_args()
print(json.dumps(build_fusion_shell(args.manifest), indent=2))
