"""audit_list.py <disagreements_top.jsonl> <out.jsonl> [per issue]: a short list to check in game: the most persistent
disagreements of each issue type, at most one per link and none within 150 m of another pick, with both images."""
import json, sys, math

T = [json.loads(l) for l in open(sys.argv[1])]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 10
picks = []
for issue in ('camera_only', 'files_only', 'type', 'offset'):
  n = 0
  for t in sorted((t for t in T if t['issue'] == issue), key=lambda t: -t['sections']):
    if t.get('x') is None or any(math.hypot(t['x'] - p['x'], t['y'] - p['y']) < 150 for p in picks):
      continue
    picks.append(t); n += 1
    if n >= per:
      break
with open(sys.argv[2], 'w') as f:
  for p in picks:
    f.write(json.dumps({k: p[k] for k in ('issue', 'a', 'b', 's', 's_range', 'sections', 'dir', 'x', 'y', 'camera', 'files', 'img', 'files_img') if k in p}) + '\n')
print(len(picks), 'picks')
