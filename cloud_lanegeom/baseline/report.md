# Paint score: /home/user/live/live19b

## Map vs paint (matched paint length with the map's nearest line over 0.5 / 1.0 m)

| class | km | > 0.5 m | > 1.0 m | base > 0.5 m | base > 1.0 m |
|---|---|---|---|---|---|
| all | 935.62 | 25.2% | 8.7% | 25.2% | 8.7% |
| motorway | 234.36 | 18.8% | 7.7% | 18.8% | 7.7% |
| none | 0.34 | 75.3% | 60.1% | 75.3% | 60.1% |
| primary | 194.88 | 34.7% | 9.9% | 34.7% | 9.9% |
| residential | 399.92 | 24.0% | 7.0% | 24.0% | 7.0% |
| service | 18.48 | 55.2% | 34.8% | 55.2% | 34.8% |
| track | 0.75 | 48.5% | 34.1% | 48.5% | 34.1% |
| trunk | 55.07 | 11.1% | 5.5% | 11.1% | 5.5% |
| unclassified | 31.82 | 36.5% | 20.5% | 36.5% | 20.5% |

Same samples before and after (matched in both):

| class | km | base > 1 m | now > 1 m | m nearer by 0.5 | m further by 0.5 |
|---|---|---|---|---|---|
| all | 935.6 | 8.7% | 8.7% | 0 | 0 |
| motorway | 234.4 | 7.7% | 7.7% | 0 | 0 |
| primary | 194.9 | 9.9% | 9.9% | 0 | 0 |
| residential | 399.9 | 7.0% | 7.0% | 0 | 0 |
| service | 18.5 | 34.8% | 34.8% | 0 | 0 |
| track | 0.7 | 34.1% | 34.1% | 0 | 0 |
| trunk | 55.1 | 5.5% | 5.5% | 0 | 0 |
| unclassified | 31.8 | 20.5% | 20.5% | 0 | 0 |

## Drift re-score: target centre vs paint lane centre, share of frames

| target | > 0.5 m | > 1.0 m | > 1.0 m j20 | base > 0.5 m | base > 1.0 m | base > 1.0 m j20 |
|---|---|---|---|---|---|---|
| static | 10.39% | 3.63% | 2.50% | 10.39% | 3.63% | 2.50% |
| section_at | 10.19% | 3.61% | 2.51% | 10.19% | 3.61% | 2.51% |
| planned line | 11.45% | 4.46% | 2.57% | 11.45% | 4.46% | 2.57% |

| steps | per km | away | away per km | away > 2 m | paint-checked | spurious |
|---|---|---|---|---|---|---|
| static | 5.55 | 162 | 1.008 | 66 | 77 | 23 |
| base static | 5.55 | 162 | 1.008 | 66 | 77 | 23 |
| section_at | 3.27 | 58 | 0.361 | 42 | 30 | 2 |
| base section_at | 3.27 | 58 | 0.361 | 42 | 30 | 2 |

Rebuilt vs recorded static target (meaningful on the recorded map only): {'frames': 156170, 'within_0.1': 0.9728, 'within_0.5': 0.9971, 'p99': 0.211}

## Events and places: largest |target - paint centre| (m), static / section_at / planned line

| kind | name | cause | static | section_at | line | base line | status |
|---|---|---|---|---|---|---|---|
| event | map1011g-006@478.5,-482.4 | map_offset | 2.73 | 2.74 | 2.74 | 2.74 | same |
| event | map1011g-007@599.2,-1233.5 | unknown | 2.58 | 2.58 | 2.74 | 2.74 | same |
| event | map1011g-014@120.0,-802.0 | unknown | 2.53 | 2.52 | 2.2 | 2.2 | same |
| event | map1011g-014@742.9,-655.0 | unknown | 2.81 | 2.96 | 3.0 | 3.0 | same |
| event | map1011g-018@706.9,-1249.2 | map_step | 0.21 | 0.15 | 0.19 | 0.19 | same |
| event | map1011g-018@729.0,-1237.0 | mixed | 1.4 | 2.84 | 2.99 | 2.99 | same |
| event | map1011g-021@1079.5,-1546.2 | map_step | 2.91 | 1.37 | 1.1 | 1.1 | same |
| event | map1011g-021@256.7,-1176.0 | map_offset | 2.07 | 1.76 | 1.65 | 1.65 | same |
| event | map1011g-046@-127.7,-1182.5 | mixed | 2.52 | 3.21 | 3.51 | 3.51 | same |
| event | map1011g-046@-388.5,-975.8 | unknown | 2.39 | 2.39 | 2.84 | 2.84 | same |
| event | map1011g-047@32.6,-1372.3 | plan | 0.39 | 0.21 | 0.19 | 0.19 | same |
| event | map1011g-049@256.7,-1176.0 | map_offset | 2.23 | 1.55 | 1.68 | 1.68 | same |
| event | map1011g-053@-374.4,-537.0 | map_offset | 0.14 | 0.14 | 0.14 | 0.14 | same |
| event | map1011g-056@1190.9,441.6 | plan | 3.4 | 3.4 | 3.36 | 3.36 | same |
| event | map1011g-067@1135.2,-3114.7 | plan | 2.61 | 2.61 | 2.12 | 2.12 | same |
| event | map1011g-068@-378.3,-2061.8 | map_step | 2.75 | 2.57 | 2.5 | 2.5 | same |
| event | map1011g-075@883.9,-786.5 | map_step | 2.87 | 2.87 | 2.85 | 2.85 | same |
| event | map1011g-077@901.4,-695.3 | map_offset | 2.53 | 2.53 | 2.53 | 2.53 | same |
| event | map1011g-077@530.7,-498.4 | map_offset | 3.04 | 3.04 | 3.03 | 3.03 | same |
| event | map1011g-080@134.0,-492.0 | map_step | 2.52 | 2.51 | 2.51 | 2.51 | same |
| event | map1011g-082@-378.2,-2061.8 | map_step | 2.7 | 2.56 | 2.48 | 2.48 | same |
| event | map1011g-082@11.1,-2044.4 | map_offset | 2.29 | 2.18 | 2.25 | 2.25 | same |
| event | map1011h-003@262.8,-1177.4 | map_offset | 2.14 | 1.71 | 1.63 | 1.63 | same |
| event | map1011h-006@478.5,-481.9 | map_offset | 2.73 | 2.74 | None | None |  |
| recurring | strawberry_exit |  | 2.23 | 1.76 | 1.68 | 1.68 | same |
| recurring | diverge_-378_-2062 |  | 2.75 | 2.57 | 2.5 | 2.5 | same |
| recurring | step_68_-1146 |  | 3.46 | 2.71 | 2.65 | 2.65 | same |
| recurring | plan_867_-1746 |  | 1.31 | 1.31 | 1.31 | 1.31 | same |
| recurring | plan_-1521_-243 |  | 1.45 | 2.57 | 2.91 | 2.91 | same |
| named | chain_1079_-1546 |  | 2.91 | 1.37 | 1.1 | 1.1 | same |
| named | drop_1191_442 |  | 3.4 | 3.4 | 3.36 | 3.36 | same |
| named | offset_901_-695 |  | 2.53 | 2.53 | 2.53 | 2.53 | same |
| named | offset_531_-498 |  | 3.04 | 3.04 | 3.03 | 3.03 | same |
| named | offset_126_-1367 |  | 2.68 | 2.68 | 2.68 | 2.68 | same |
| named | offset_-14_275 |  | 1.94 | 1.88 | 0.73 | 0.73 | same |
| named | plan_1135_-3115 |  | 2.72 | 2.72 | 2.23 | 2.23 | same |
| named | stacked_478_-482 |  | 2.73 | 2.74 | 2.74 | 2.74 | same |
