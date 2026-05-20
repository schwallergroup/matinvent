import os
import pickle
from pathlib import Path
from typing import Any

import lmdb


def lmdb_open(db_path: str | os.PathLike, readonly: bool = False) -> lmdb.Environment:
    if readonly:
        return lmdb.open(
            str(db_path), subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False, max_readers=1,
        )
    return lmdb.open(
        str(db_path), map_size=1099511627776 * 2,
        subdir=False, meminit=False, map_async=True,
    )


def lmdb_read_metadata(db_path: str | os.PathLike, key: str, default=None) -> Any:
    with lmdb_open(db_path, readonly=True) as db:
        with db.begin() as txn:
            result = lmdb_get(txn, key, default=default)
    return result


def lmdb_put(txn: lmdb.Transaction, key: str, value: Any) -> bool:
    return txn.put(
        key.encode("ascii"),
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
    )


def lmdb_get(txn: lmdb.Transaction, key: str, default: Any = None, raise_if_missing: bool = True) -> Any:
    value = txn.get(key.encode("ascii"))
    if value is None:
        if default is None and raise_if_missing:
            raise KeyError(f"Key {key} not found in database.")
        return default
    return pickle.loads(value)
