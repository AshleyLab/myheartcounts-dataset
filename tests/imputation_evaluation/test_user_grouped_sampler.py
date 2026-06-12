"""Unit tests for UserGroupedBatchSampler.

The sampler is used by personalized imputers under the lazy per-user
state contract to keep each impute() batch confined to one user's
samples (or a small packing of complete users), so the worker's lazy
cache doesn't thrash.
"""

from __future__ import annotations

from imputation_evaluation.data.data_loader import UserGroupedBatchSampler


def _flatten(batches):
    out = []
    for b in batches:
        out.extend(b)
    return out


class TestUserGroupedBatchSampler:
    def test_keeps_a_user_contiguous_within_a_batch(self):
        # User u0 has 3 samples, u1 has 2, u2 has 5. batch_size=4.
        # Expected: u0 + u1 don't fit together (3+2=5 > 4) → flush u0 in
        # its own batch; u1 in next; u2 has 5 > 4 → chunked into [5,6,7,8]
        # and [9].
        ptu = ["u0", "u0", "u0", "u1", "u1", "u2", "u2", "u2", "u2", "u2"]
        s = UserGroupedBatchSampler(ptu, batch_size=4)
        batches = list(iter(s))
        assert len(s) == len(batches)
        # Every batch is contiguous within one user (the simple invariant).
        for b in batches:
            uids = {ptu[i] for i in b}
            assert len(uids) == 1, f"batch {b} spans users {uids}"
        # No position dropped.
        assert sorted(_flatten(batches)) == list(range(len(ptu)))

    def test_packs_small_users_together(self):
        # Three small users (2 samples each) at batch_size=8 → all six
        # positions fit in one batch.
        ptu = ["u0", "u0", "u1", "u1", "u2", "u2"]
        s = UserGroupedBatchSampler(ptu, batch_size=8)
        batches = list(iter(s))
        assert len(batches) == 1
        assert sorted(batches[0]) == list(range(6))

    def test_splits_user_larger_than_batch_size_into_consecutive_batches(self):
        # Single user with 10 samples, batch_size=4 → batches of 4, 4, 2.
        ptu = ["u0"] * 10
        s = UserGroupedBatchSampler(ptu, batch_size=4)
        batches = list(iter(s))
        assert [len(b) for b in batches] == [4, 4, 2]
        assert _flatten(batches) == list(range(10))

    def test_iterating_twice_is_stable(self):
        ptu = ["u0", "u0", "u1", "u1", "u1", "u2"]
        s = UserGroupedBatchSampler(ptu, batch_size=3)
        first = list(iter(s))
        second = list(iter(s))
        assert first == second
