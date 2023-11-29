import json

import lmdb

from serve.utils_general import save_to_cache


def jsonl_to_lmdb(jsonl_file: str, lmdb_file: str):
    env = lmdb.open(lmdb_file, map_size=1e9)  # 1GB size, adjust if needed
    with open(jsonl_file, "r") as f:
        for line in f:
            item = json.loads(line)
            print(item["key"])
            save_to_cache(item["key"], item["value"], env)
    env.close()


if __name__ == "__main__":
    jsonl_file = "cache_vlm2.jsonl"
    lmdb_file = "cache/cache_vlm"
    jsonl_to_lmdb(jsonl_file, lmdb_file)
