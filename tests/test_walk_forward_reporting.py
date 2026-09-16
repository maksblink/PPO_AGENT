import numpy as np
import pandas as pd
import pytest

from train_and_eval.walk_forward.reporting import combine_tests
from train_and_eval.walk_forward.service import select_candidate


def test_additive_test_path_and_global_drawdown():
    a = pd.DataFrame({"timestamp": ["2020-01-06", "2020-01-07"], "agent_equity": [.2, .1], "always_long_equity": [.05, .1]})
    b = pd.DataFrame({"timestamp": ["2020-01-13", "2020-01-14"], "agent_equity": [-.15, -.1], "always_long_equity": [-.05, .1]})
    curve, summary = combine_tests([a,b], 1000)
    np.testing.assert_allclose(curve.agent_equity, [.2,.1,-.05,0])
    assert summary["agent"]["total_pnl_pln"] == pytest.approx(0)
    assert summary["agent"]["max_drawdown_pln"] == pytest.approx(-250)
    assert summary["always_long"]["total_return_on_stake"] == pytest.approx(.2)  # not .21


def test_test_path_includes_initial_zero_peak():
    part = pd.DataFrame({"timestamp": ["2020-01-06"], "agent_equity": [-.1], "always_long_equity": [-.2]})
    _, summary = combine_tests([part], 1000)
    assert summary["agent"]["max_drawdown_pln"] == -100
    with pytest.raises(ValueError, match="overlap"):
        combine_tests([part,part], 1000)
