import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.fill_model_msg import fill_lane_head, lane_head_marginals
from openpilot.selfdrive.modeld.parse_model_outputs import Parser

N = ModelConstants.LANE_HEAD_MAX
CLASSES = [(c, i) for c in range(1, N + 1) for i in range(c)]  # (count, index), as gta5-train's lane_class


class TestLaneHead:
  def test_classes(self):
    for k, (c, i) in enumerate(CLASSES):
      assert k == c * (c - 1) // 2 + i
      probs = np.full(len(CLASSES), 1e-4, np.float32)
      probs[k] = 1.0
      lh = log.ModelDataV2.new_message().laneHead
      fill_lane_head(lh, probs / probs.sum())
      assert (lh.laneCount, lh.laneIdx) == (c, i)

  def test_marginals(self):
    p = np.random.default_rng(0).random(len(CLASSES)).astype(np.float32)
    p /= p.sum()
    idx, count = lane_head_marginals(p)
    assert np.isclose(idx.sum(), 1) and np.isclose(count.sum(), 1)
    assert np.isclose(idx[0], sum(p[k] for k, (_, i) in enumerate(CLASSES) if i == 0))
    assert np.isclose(count[2], sum(p[k] for k, (c, _) in enumerate(CLASSES) if c == 3))

  def test_parse(self):
    logits = np.random.default_rng(1).normal(size=(1, len(CLASSES))).astype(np.float32)
    outs = Parser(ignore_missing=True).parse_policy_outputs({'lane_head': logits.copy()})
    assert np.isclose(outs['lane_head'].sum(), 1) and outs['lane_head'].argmax() == logits.argmax()
    assert 'lane_head' not in Parser(ignore_missing=True).parse_policy_outputs({})

  def test_default_without_head(self):
    lh = log.ModelDataV2.new_message().laneHead
    assert (lh.laneIdx, lh.laneCount) == (-1, -1)
