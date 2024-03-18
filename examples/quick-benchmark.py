#!/usr/bin/env python3

from random import randbytes
from time import process_time_ns

import reed_solomon_leopard

def generate_shards(shard_bytes, count):
    return [randbytes(shard_bytes) for _ in range(count)]


def encode(original, recovery_count):
    start = process_time_ns()
    result = reed_solomon_leopard.encode(original, recovery_count)
    end = process_time_ns()

    elapsed_seconds = (end-start) / 1e9

    return result, elapsed_seconds


def decode(original, recovery):
    # Throw away maximum number of shards
    for i in range(min(len(original), len(recovery))):
        original[i] = None

    if len(recovery) > len(original):
        for i in range(len(original), len(recovery)):
            recovery[i] = None

    original_dict = {idx: val for idx, val in enumerate(original) if val != None}
    recovery_dict = {idx: val for idx, val in enumerate(recovery) if val != None}

    start = process_time_ns()
    recovery_shards = reed_solomon_leopard.decode(len(original), len(recovery), original_dict, recovery_dict)
    end = process_time_ns()

    elapsed_seconds = (end-start) / 1e9

    return recovery_shards, elapsed_seconds


def benchmark(original_count, recovery_count, shard_bytes):
    assert reed_solomon_leopard.supports(original_count, recovery_count)

    original_shards = generate_shards(shard_bytes, original_count)

    # MiB/s are w.r.t the total amount of data
    total_size_MiB = (((original_count + recovery_count) * shard_bytes) / 1024**2)

    recovery_shards, elapsed_seconds = encode(original_shards, recovery_count)
    MiB_per_second = total_size_MiB / elapsed_seconds
    print(f"Encoding {original_count} shards of size {shard_bytes} to {recovery_count} recovery shards to took {elapsed_seconds:.2f} seconds ({MiB_per_second:.2f} MiB/s)")

    restored_shards, elapsed_seconds = decode(original_shards.copy(), recovery_shards)
    MiB_per_second = total_size_MiB / elapsed_seconds
    print(f"Decoding with 100% loss took {elapsed_seconds:.2f} seconds ({MiB_per_second:.2f} MiB/s)")

    for idx, shard in restored_shards.items():
        assert shard == original_shards[idx]


def main():
    benchmark(128,   64,    1024 * 512)
    benchmark(2048,  4096,  1024 * 256)
    benchmark(16384, 16384, 1024 * 64)
    benchmark(32768, 32768, 1024 * 32)
    benchmark(49152, 16384, 1024 * 16)
    benchmark(16384, 49152, 1024 * 16)

if __name__ == "__main__":
    main()
