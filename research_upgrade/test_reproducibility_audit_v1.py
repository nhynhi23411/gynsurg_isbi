import unittest
from research_upgrade.reproducibility_audit_v1 import historical_margins, query_comparison


class RecordAuditTests(unittest.TestCase):
    def test_order_change_is_not_set_change(self):
        history = [(0, 0), (49, 1), (24, 1), (0, 1), (49, 0)]
        altered = history[:3] + history[3:][::-1]
        self.assertEqual(query_comparison(history, altered), (4, True))

    def test_missing_query_is_error_not_pass(self):
        with self.assertRaises(ValueError):
            query_comparison([(0, 0)], [(0, 0)])

    def test_exact_entropy_tie_uses_frame_then_head(self):
        data = [dict(head=k, subset=s, frame=f, entropy=.5) for k in (0, 1) for s in range(8) for f in (0, 24, 49)]
        history = [(0, 0), (0, 1), (24, 0), (24, 1), (49, 0)]
        margins = historical_margins(data, history)
        self.assertEqual([r['entropy_margin'] for r in margins], [0] * 5)
        self.assertEqual(margins[2]['historical_paid_head0'], 1)
        with self.assertRaises(ValueError):
            historical_margins(data, history[::-1])


if __name__ == '__main__':
    unittest.main()
