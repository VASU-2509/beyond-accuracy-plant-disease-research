import json
import shutil
from research.src.utils import environment
from research.storage import storage_root, paths

if __name__ == "__main__":
    print(json.dumps({**environment(), "storage_root": str(storage_root()), "paths": {k: str(v) for k, v in paths().items()}, "disk_free_gib": shutil.disk_usage(storage_root().anchor).free / 1024**3}, indent=2))
