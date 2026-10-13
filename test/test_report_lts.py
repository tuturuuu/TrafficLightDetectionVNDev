import copy
import json

import pytest

from adaptile.report_lts import METRICS, PROTOCOL_KEYS, aggregate


def write_runs(tmp_path):
    paths = []
    for seed, score in zip((0, 42, 100), (.8, .9, 1.0)):
        meta = {k: 'matched' for k in PROTOCOL_KEYS}
        meta.update(seed=seed, experiment_id='synthetic', dataset='synthetic', selector={'name': 'none'},
                    detector_sha256=f'detector_seed{seed}')
        row = {k: score for k in METRICS}
        row.update(schema_version='lat_yolo_v1', provenance=meta, tile_size=640,
                   method='uniform', coverage_visibility=.5)
        path = tmp_path / f'{seed}.json'
        path.write_text(json.dumps([row])); paths.append(path)
    return paths


def test_mean_sample_sd_and_reject_missing_or_duplicate_seeds(tmp_path):
    paths = write_runs(tmp_path)
    metrics = aggregate(paths)[0]['metrics']['mAP50']
    assert metrics['mean'] == pytest.approx(.9)
    assert metrics['std'] == pytest.approx(.1)
    with pytest.raises(ValueError, match='expected seeds'):
        aggregate(paths[:2])
    with pytest.raises(ValueError, match='Duplicate seed'):
        aggregate(paths + paths[:1])


def test_reject_protocol_change_and_unmatched_selector_weights(tmp_path):
    paths = write_runs(tmp_path)
    rows = json.loads(paths[0].read_text())
    rows[0]['provenance']['profile_stages'] = False
    paths[0].write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='incompatible settings'):
        aggregate(paths)
    extra = copy.deepcopy(rows[0])
    extra['method'] = 'lts'
    extra['provenance']['selector'] = {'name': 'lts', 'threshold': .5}
    extra['provenance']['detector_sha256'] = 'different'
    paths[0].write_text(json.dumps(rows + [extra]))
    with pytest.raises(ValueError, match='Unmatched detector'):
        aggregate(paths)
