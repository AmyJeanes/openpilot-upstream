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


# n16 (comma's nav model) as gta5-train exports it with the head: n16's own slices unchanged, its pad included, then
# lane_head after its 18580 floats (gta5_train/model.py nav_output_slices)
N16_SLICES = {'lane_lines': slice(0, 528), 'lane_lines_prob': slice(528, 536), 'road_edges': slice(536, 800),
              'meta': slice(800, 855), 'desire_pred': slice(855, 887), 'pose': slice(887, 899),
              'wide_from_device_euler': slice(899, 905), 'road_transform': slice(905, 917), 'plan': slice(917, 1907),
              'lead': slice(1907, 2051), 'lead_prob': slice(2051, 2054), 'desire_state': slice(2054, 2062),
              'action': slice(2062, 2066), 'hidden_state': slice(2066, 18450), 'map_features': slice(18450, 18514),
              'navigation_features': slice(18514, 18578), 'pad': slice(18578, 18580)}
N16_LANE_SLICES = N16_SLICES | {'lane_head': slice(18580, 18580 + len(CLASSES))}


def slice_outputs(model_output: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
  return {k: model_output[np.newaxis, v] for k, v in output_slices.items()}  # modeld's ModelState.slice_outputs


class TestLaneHeadN16:
  def test_n16_layout(self):
    from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg
    out = np.random.default_rng(2).normal(size=N16_LANE_SLICES['lane_head'].stop).astype(np.float32)
    k = 30  # (count 8, index 2)
    out[N16_LANE_SLICES['lane_head']] = 0
    out[N16_LANE_SLICES['lane_head'].start + k] = 9
    msgs = []
    for slices in (N16_SLICES, N16_LANE_SLICES):
      outs = Parser().parse_outputs(slice_outputs(out[:max(s.stop for s in slices.values())].copy(), slices))
      msg = log.Event.new_message()
      msg.init('modelV2')
      fill_model_msg(msg, outs, log.ModelDataV2.Action.new_message(), PublishState(), 0, 0, 0, 0., 0, 0., True)
      msgs.append(msg.modelV2)
    without, with_head = msgs
    assert (without.laneHead.laneIdx, without.laneHead.laneCount) == (-1, -1)
    assert (with_head.laneHead.laneCount, with_head.laneHead.laneIdx) == CLASSES[k] == (8, 2)
    assert with_head.laneHead.prob > 0.99
    # every shipped output reads the same with the head after them
    assert list(with_head.position.y) == list(without.position.y)
    assert list(with_head.meta.desirePrediction) == list(without.meta.desirePrediction)
    assert [ll.y[-1] for ll in with_head.laneLines] == [ll.y[-1] for ll in without.laneLines]
