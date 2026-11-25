from pathlib import Path
import tempfile
import unittest

from capture import dump, previous
from summary import partition, report


class CaptureTest(unittest.TestCase):
    def test_partitions_only_the_lifetime_and_does_not_double_count(self):
        spans = [dict(kind='join', check=1, start=10, end=40), dict(kind='join', check=1, start=30, end=42),
                 dict(kind='other', check=None, start=35, end=55), dict(kind='join', check=2, start=70, end=120)]
        self.assertEqual(partition(20, 100, 1, spans), dict(joinOnly=15, otherWorkOnly=13, joinedElsewhere=30, conflicting=7, unobserved=15))

    def test_dispatch_uses_latest_visible_success_and_only_an_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for num, key, start, end, code in [(1,'a',1,11,0),(2,'a',13,28,0),(3,'a',2,21,1),(4,'b',5,22,0),(5,'a',23,None,0)]:
                d=root/f'check-{num}'; d.mkdir(); dump(d/'start',dict(ns=start,key=key,prior_ns=None))
                if end is not None: dump(d/'end',dict(ns=end,code=code))
            self.assertEqual(previous(root,'a',24),10)
            self.assertEqual(previous(root,'a',28),15)
            self.assertIsNone(previous(root,'b',21))
            scored=report(root)
            self.assertEqual((scored['allSuccessful']['calls'],scored['censoredOrNotLaunched'],scored['failed']),(3,1,1))
            self.assertIsNone(scored['priorAtLeast60s']['previousRuntimeMaeSeconds'])

    def test_retains_fast_outcomes_after_a_previously_slow_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); d=root/'check-2'; d.mkdir()
            dump(d/'start',dict(ns=1000,key='a',prior_ns=70_000_000_000))
            dump(d/'end',dict(ns=2_000_001_000,code=0))
            r=report(root)['priorAtLeast60s']
            self.assertEqual((r['calls'],r['previousRuntimeMaeSeconds'],r['activitySeconds']['unobserved']),(1,68,2))

    def test_estimates_only_remaining_wait_at_a_targeted_blocking_join(self):
        ns=1_000_000_000
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); d=root/'check-1'; d.mkdir()
            dump(d/'start',dict(ns=0,key='same',prior_ns=90*ns))
            dump(d/'end',dict(ns=100*ns,code=0))
            dump(root/'span-one',dict(start=80*ns,end=101*ns,kind='join',check=1))
            r=report(root)['priorAtLeast60s']
            self.assertEqual((r['joinsScoredAtEntry'],r['remainingJoinMaeSeconds'],r['serialRuntimeVsRecordedJoinMaeSeconds']),(1,10,70))
            dump(root/'span-one',dict(start=102*ns,end=103*ns,kind='join',check=1))
            self.assertIsNone(report(root)['priorAtLeast60s']['remainingJoinMaeSeconds'])



if __name__ == '__main__': unittest.main()
