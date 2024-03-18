reed-solomon-leopard
====

Fast Reed-Solomon encoding and decoding for Python using the Rust crate [reed-solomon-simd](https://crates.io/crates/reed-solomon-simd).

- `O(n log n)` complexity.
- Any combination of 1 - 32768 original shards with 1 - 32768 recovery shards.
- Up to 65535 original or recovery shards is also possible with following limitations:

| `original_count` | `recovery_count` |
| ---------------- | ---------------- |
| `<= 2^16 - 2^n`  | `<= 2^n`         |
| `<= 61440`       | `<= 4096`        |
| `<= 57344`       | `<= 8192`        |
| `<= 49152`       | `<= 16384`       |
| **`<= 32768`**   | **`<= 32768`**   |
| `<= 16384`       | `<= 49152`       |
| `<= 8192`        | `<= 57344`       |
| `<= 4096`        | `<= 61440`       |
| `<= 2^n`         | `<= 2^16 - 2^n`  |

Note: this library does not detect or correct errors within a shard. So if data corruption is a likely scenario, you should include an error detection hash with each shard, and skip feeding the corrupted shards to the decoder.

Installation
----

You can install reed-solomon-leopard via pip:

```bash
pip install reed-solomon-leopard
```

Usage
----

```python
import reed_solomon_leopard

# Our data shards. Each shard must be of the same length as all the other
# shards, and the length must be a multiple of 64.
original = [
    b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ",
    b"eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut e",
    b"nim ad minim veniam, quis nostrud exercitation ullamco laboris n",
];

# Generate 5 recovery shards.
recovery = reed_solomon_leopard.encode(original, 5)

# Simulate losing some shards. Any 3 of the 8 shards (3 original +
# 5 recovery) can be used to restore the data. Let's assume we lost
# all, except one original and two recovery shards.
original_partial = {
        1: original[1],
}
recovery_partial = {
        1: recovery[1],
        4: recovery[4],
}

# Let the reed solomon library do its magic.
restored = reed_solomon_leopard.decode(len(original), len(recovery), original_partial, recovery_partial)

# We got the two missing original shards back
assert restored[0] == original[0]
assert restored[2] == original[2]
```

Benchmarks
----
```
$ examples/quick-benchmark.py
Encoding 128 shards of size 524288 to 64 recovery shards to took 0.06 seconds (1563.03 MiB/s)
Decoding with 100% loss took 0.15 seconds (660.80 MiB/s)
Encoding 2048 shards of size 262144 to 4096 recovery shards to took 1.21 seconds (1268.50 MiB/s)
Decoding with 100% loss took 3.33 seconds (461.51 MiB/s)
Encoding 16384 shards of size 65536 to 16384 recovery shards to took 1.69 seconds (1215.11 MiB/s)
Decoding with 100% loss took 4.36 seconds (470.06 MiB/s)
Encoding 32768 shards of size 32768 to 32768 recovery shards to took 1.68 seconds (1217.89 MiB/s)
Decoding with 100% loss took 4.51 seconds (453.93 MiB/s)
Encoding 49152 shards of size 16384 to 16384 recovery shards to took 0.91 seconds (1125.27 MiB/s)
Decoding with 100% loss took 2.15 seconds (476.35 MiB/s)
Encoding 16384 shards of size 16384 to 49152 recovery shards to took 0.95 seconds (1078.45 MiB/s)
Decoding with 100% loss took 2.04 seconds (500.95 MiB/s)
```
With an AMD Ryzen 5 3600 CPU.

Credits
----
[Leopard-RS](https://github.com/catid/leopard)
