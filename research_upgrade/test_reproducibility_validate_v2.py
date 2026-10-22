import unittest
from research_upgrade.reproducibility_validate_v2 import check_pair


class PairIntegrityTests(unittest.TestCase):
    def test_exact_flag_and_numeric_values_must_agree_with_snapshot_hash(self):
        diag = dict(snapshots=[dict(sha256='identical')], bought_queries=[])
        row = dict(step='0', exact_numeric_equality='True', maximum_absolute_difference='0',
                   mean_absolute_difference='0', threshold_05_disagreement_fraction='0', paid_query_prefix_identical='True')
        check_pair(row, diag, diag)
        for field, value in [('exact_numeric_equality', 'False'), ('mean_absolute_difference', '0.1'),
                             ('maximum_absolute_difference', 'nan'), ('paid_query_prefix_identical', 'False')]:
            with self.assertRaises(AssertionError):
                check_pair(dict(row, **{field: value}), diag, diag)
