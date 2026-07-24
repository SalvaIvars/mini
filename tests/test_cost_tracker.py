import pytest

from mini.cost_tracker import CostTracker
from mini.exceptions import LimitsExceeded


class TestCostTracker:
    def test_update_tracks_uncached_input(self):
        tracker = CostTracker(input_cost_per_token=0.001, output_cost_per_token=0.002)
        tracker.update({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 0},
        })
        assert tracker.total_uncached_tokens == 100
        assert tracker.total_cached_tokens == 0
        assert tracker.total_completion_tokens == 50
        assert tracker.cost == pytest.approx(0.2)

    def test_update_tracks_cached_input(self):
        tracker = CostTracker(
            input_cost_per_token=0.001,
            cached_input_cost_per_token=0.0005,
            output_cost_per_token=0.002,
        )
        tracker.update({
            "prompt_tokens": 200,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 100},
        })
        assert tracker.total_uncached_tokens == 100
        assert tracker.total_cached_tokens == 100
        assert tracker.total_completion_tokens == 50
        assert tracker.cost == pytest.approx(0.1 + 0.05 + 0.1)

    def test_multiple_updates_accumulate(self):
        tracker = CostTracker(input_cost_per_token=0.001, output_cost_per_token=0.001)
        tracker.update({"prompt_tokens": 10, "completion_tokens": 5})
        tracker.update({"prompt_tokens": 20, "completion_tokens": 10})
        assert tracker.total_uncached_tokens == 30
        assert tracker.total_completion_tokens == 15
        assert tracker.cost == pytest.approx(0.045)

    def test_check_limits_passes_when_under(self):
        tracker = CostTracker(call_limit=10)
        tracker.n_calls = 5
        tracker.check_limits()  # should not raise

    def test_check_limits_raises_on_call_limit(self):
        tracker = CostTracker(call_limit=5)
        tracker.n_calls = 5
        with pytest.raises(LimitsExceeded) as exc:
            tracker.check_limits()
        assert exc.value.messages[0]["content"] == "LimitsExceeded"

    def test_check_limits_raises_on_cost_limit(self):
        tracker = CostTracker(cost_limit=1.0)
        tracker.cost = 1.0
        with pytest.raises(LimitsExceeded) as exc:
            tracker.check_limits()
        assert exc.value.messages[0]["content"] == "CostLimitExceeded"

    def test_record_call_increments(self):
        tracker = CostTracker()
        tracker.record_call()
        tracker.record_call()
        assert tracker.n_calls == 2

    def test_reset_clears_everything(self):
        tracker = CostTracker()
        tracker.update({"prompt_tokens": 10, "completion_tokens": 5})
        tracker.record_call()
        tracker.reset()
        assert tracker.cost == 0
        assert tracker.n_calls == 0
        assert tracker.total_uncached_tokens == 0
        assert tracker.total_completion_tokens == 0

    def test_report(self):
        tracker = CostTracker(
            input_cost_per_token=0.001,
            cached_input_cost_per_token=0.0005,
            output_cost_per_token=0.002,
        )
        tracker.update({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 20},
        })
        tracker.record_call()
        report = tracker.report()
        assert report["cost"] == tracker.cost
        assert report["calls"] == 1
        assert report["uncached_tokens"] == 80
        assert report["cached_tokens"] == 20
        assert report["completion_tokens"] == 50
        assert report["uncached_cost"] == pytest.approx(0.08)
        assert report["cached_cost"] == pytest.approx(0.01)
        assert report["completion_cost"] == pytest.approx(0.1)
