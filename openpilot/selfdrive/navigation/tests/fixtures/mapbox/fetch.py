#!/usr/bin/env python3
"""Refetch the fixtures here: MAPBOX_TOKEN from the environment only (never printed or saved), one request each.

  MAPBOX_TOKEN=... python fetch.py [name ...]      # all, or the ones named; prints status and route count only
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
PARAMS = {'steps': 'true', 'banner_instructions': 'true', 'annotations': 'maxspeed,congestion,distance,duration',
          'overview': 'full', 'geometries': 'polyline6', 'alternatives': 'false', 'language': 'en'}


def fetch(path: str) -> requests.Response:
  return requests.get('https://api.mapbox.com' + path, params={**PARAMS, 'access_token': os.environ['MAPBOX_TOKEN']}, timeout=30)


def main():
  token = os.environ['MAPBOX_TOKEN']
  names = sys.argv[1:] or sorted(p.stem for p in HERE.glob('*.json'))
  for name in names:
    f = HERE / f'{name}.json'
    rec = json.loads(f.read_text())
    r = fetch(rec['request']['path'])
    print(name, r.status_code, len(r.json().get('routes', [])) if r.ok else '')
    if not r.ok:
      continue
    rec.update(fetched=time.strftime('%Y-%m-%d'), response=r.json())
    text = json.dumps(rec, separators=(',', ':'))
    assert token not in text
    f.write_text(text + '\n')


if __name__ == '__main__':
  main()
