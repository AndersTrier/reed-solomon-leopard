#!/usr/bin/env python3
import reed_solomon_leopard

def reed_solomon_leopard_example():
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

if __name__ == "__main__":
    reed_solomon_leopard_example()
