# Paint score: /home/user/build/live21

## Map vs paint (matched paint length with the map's nearest line over 0.5 / 1.0 m)

| class | km | > 0.5 m | > 1.0 m | base > 0.5 m | base > 1.0 m |
|---|---|---|---|---|---|
| all | 937.93 | 25.1% | 8.6% | 25.2% | 8.7% |
| motorway | 235.07 | 18.8% | 7.7% | 18.8% | 7.7% |
| none | 0.34 | 75.9% | 59.8% | 75.3% | 60.1% |
| primary | 196.45 | 34.1% | 9.6% | 34.7% | 9.9% |
| residential | 399.8 | 23.9% | 7.0% | 24.0% | 7.0% |
| service | 18.52 | 55.2% | 34.8% | 55.2% | 34.8% |
| track | 0.75 | 48.6% | 34.4% | 48.5% | 34.1% |
| trunk | 55.14 | 11.2% | 5.3% | 11.1% | 5.5% |
| unclassified | 31.85 | 36.6% | 20.5% | 36.5% | 20.5% |

Same samples before and after (matched in both):

| class | km | base > 1 m | now > 1 m | m nearer by 0.5 | m further by 0.5 |
|---|---|---|---|---|---|
| all | 934.1 | 8.6% | 8.5% | 4396 | 1725 |
| motorway | 234.2 | 7.7% | 7.6% | 854 | 483 |
| primary | 194.1 | 9.7% | 9.2% | 2622 | 717 |
| residential | 399.4 | 6.9% | 6.9% | 677 | 448 |
| service | 18.5 | 34.8% | 34.8% | 29 | 12 |
| track | 0.7 | 34.2% | 34.4% | 0 | 0 |
| trunk | 55.1 | 5.5% | 5.3% | 153 | 2 |
| unclassified | 31.8 | 20.4% | 20.4% | 57 | 62 |

## Drift re-score: target centre vs paint lane centre, share of frames

| target | > 0.5 m | > 1.0 m | > 1.0 m j20 | base > 0.5 m | base > 1.0 m | base > 1.0 m j20 |
|---|---|---|---|---|---|---|
| static | 10.17% | 3.60% | 2.50% | 10.39% | 3.63% | 2.50% |
| section_at | 9.80% | 3.36% | 2.30% | 10.19% | 3.61% | 2.51% |
| planned line | 10.86% | 4.18% | 2.46% | 11.45% | 4.46% | 2.57% |

| steps | per km | away | away per km | away > 2 m | paint-checked | spurious |
|---|---|---|---|---|---|---|
| static | 5.61 | 164 | 1.021 | 67 | 77 | 23 |
| base static | 5.55 | 162 | 1.008 | 66 | 77 | 23 |
| section_at | 3.24 | 52 | 0.324 | 36 | 25 | 1 |
| base section_at | 3.27 | 58 | 0.361 | 42 | 30 | 2 |

Rebuilt vs recorded static target (meaningful on the recorded map only): {'frames': 156170, 'within_0.1': 0.935, 'within_0.5': 0.9941, 'p99': 0.403}

## Events and places: largest |target - paint centre| (m), static / section_at / planned line

| kind | name | cause | static | section_at | line | base line | status |
|---|---|---|---|---|---|---|---|
| event | map1011g-006@478.5,-482.4 | map_offset | 2.73 | 2.74 | 2.74 | 2.74 | same |
| event | map1011g-007@599.2,-1233.5 | unknown | 2.58 | 2.58 | 2.77 | 2.74 | same |
| event | map1011g-014@120.0,-802.0 | unknown | 2.53 | 2.52 | 2.2 | 2.2 | same |
| event | map1011g-014@742.9,-655.0 | unknown | 2.81 | 3.01 | 3.25 | 3.0 | same |
| event | map1011g-018@706.9,-1249.2 | map_step | 0.21 | 0.15 | 0.42 | 0.19 | same |
| event | map1011g-018@729.0,-1237.0 | mixed | 1.4 | 3.03 | 2.9 | 2.99 | same |
| event | map1011g-021@1079.5,-1546.2 | map_step | 2.91 | 1.37 | 1.1 | 1.1 | same |
| event | map1011g-021@256.7,-1176.0 | map_offset | 2.07 | None | None | 1.65 |  |
| event | map1011g-046@-127.7,-1182.5 | mixed | 2.52 | 2.52 | 2.49 | 3.51 | better |
| event | map1011g-046@-388.5,-975.8 | unknown | 2.39 | 2.39 | 2.4 | 2.84 | better |
| event | map1011g-047@32.6,-1372.3 | plan | 0.39 | 0.22 | 0.28 | 0.19 | same |
| event | map1011g-049@256.7,-1176.0 | map_offset | 2.23 | None | None | 1.68 |  |
| event | map1011g-053@-374.4,-537.0 | map_offset | 0.14 | 0.14 | 0.14 | 0.14 | same |
| event | map1011g-056@1190.9,441.6 | plan | 3.4 | 2.89 | 2.79 | 3.36 | better |
| event | map1011g-067@1135.2,-3114.7 | plan | 2.61 | 2.61 | 2.12 | 2.12 | same |
| event | map1011g-068@-378.3,-2061.8 | map_step | 2.75 | 2.57 | 2.5 | 2.5 | same |
| event | map1011g-075@883.9,-786.5 | map_step | 2.87 | 2.87 | 2.85 | 2.85 | same |
| event | map1011g-077@901.4,-695.3 | map_offset | 2.53 | 2.53 | 2.47 | 2.53 | same |
| event | map1011g-077@530.7,-498.4 | map_offset | 3.04 | 1.81 | 1.81 | 3.03 | better |
| event | map1011g-080@134.0,-492.0 | map_step | 2.52 | 2.51 | 2.51 | 2.51 | same |
| event | map1011g-082@-378.2,-2061.8 | map_step | 2.7 | 2.56 | 2.48 | 2.48 | same |
| event | map1011g-082@11.1,-2044.4 | map_offset | 2.29 | 2.18 | 2.25 | 2.25 | same |
| event | map1011h-003@262.8,-1177.4 | map_offset | 2.14 | None | None | 1.63 |  |
| event | map1011h-006@478.5,-481.9 | map_offset | 2.73 | 2.74 | None | None |  |
| recurring | strawberry_exit |  | 2.23 | None | None | 1.68 |  |
| recurring | diverge_-378_-2062 |  | 2.75 | 2.57 | 2.5 | 2.5 | same |
| recurring | step_68_-1146 |  | 3.46 | 2.71 | 2.66 | 2.65 | same |
| recurring | plan_867_-1746 |  | 1.31 | 1.31 | 1.31 | 1.31 | same |
| recurring | plan_-1521_-243 |  | 1.45 | 0.9 | 2.91 | 2.91 | same |
| named | chain_1079_-1546 |  | 2.91 | 1.37 | 1.1 | 1.1 | same |
| named | drop_1191_442 |  | 3.4 | 2.89 | 2.79 | 3.36 | better |
| named | offset_901_-695 |  | 2.53 | 2.53 | 2.47 | 2.53 | same |
| named | offset_531_-498 |  | 3.04 | 1.81 | 1.81 | 3.03 | better |
| named | offset_126_-1367 |  | 2.68 | 2.68 | 2.68 | 2.68 | same |
| named | offset_-14_275 |  | 1.94 | 1.88 | 0.73 | 0.73 | same |
| named | plan_1135_-3115 |  | 2.72 | 2.72 | 2.23 | 2.23 | same |
| named | stacked_478_-482 |  | 2.73 | 2.74 | 2.74 | 2.74 | same |
