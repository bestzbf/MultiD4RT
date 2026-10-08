"""Admission must fail closed when the downloaded exclusion set changes."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.audit_syn4d import compare_camera_pose
from scripts.audit_syn4d_pool import reviewed_rejections


def test_camera_comparison_accepts_one_ulp_but_rejects_real_translation_error():
    official = np.eye(4, dtype=np.float32)
    official[:3, 3] = [888., -556., -231.]
    rounded = official.copy()
    rounded[1, 3] = np.nextafter(rounded[1, 3], np.float32(-np.inf))
    assert compare_camera_pose(official, rounded) == pytest.approx(6.103515625e-5)
    rounded[1, 3] += .001
    with pytest.raises(AssertionError, match='translation'):
        compare_camera_pose(official, rounded)
    rounded = official.copy(); rounded[0, 0] += .001
    with pytest.raises(AssertionError):
        compare_camera_pose(official, rounded)


def test_pool_audit_requires_exact_reviewed_exclusions(tmp_path):
    item = dict(source='v1', path='scene/camera.csv', reason='missing tracking',
                review='Confirmed absent shard; retain other complete cameras.')
    manifest = dict(version=1, exclusions=[item])
    rejected = [dict(path=str(tmp_path/item['path']), reason=item['reason'])]
    assert reviewed_rejections('v1', tmp_path, rejected, manifest) == [item]
    assert reviewed_rejections('sim', tmp_path, [], manifest) == []
    with pytest.raises(ValueError, match='unreviewed'):
        reviewed_rejections('v1', tmp_path, rejected, dict(version=1, exclusions=[]))
    with pytest.raises(ValueError, match='stale'):
        reviewed_rejections('v1', tmp_path, [], manifest)
    with pytest.raises(ValueError, match='new='):
        reviewed_rejections('v1', tmp_path, [dict(rejected[0], reason='different error')], manifest)


@pytest.mark.parametrize('change', [dict(path='../camera.csv'), dict(path='/camera.csv'), dict(review='')])
def test_pool_audit_rejects_unscoped_or_unexplained_exclusions(tmp_path, change):
    manifest = dict(version=1, exclusions=[dict(source='v1', path='camera.csv',
                                              reason='missing tracking', review='Checked absence')])
    changed = deepcopy(manifest)
    changed['exclusions'][0].update(change)
    with pytest.raises(ValueError, match='root-relative'):
        reviewed_rejections('v1', tmp_path, [], changed)


def test_missing_mesh_review_does_not_permit_a_missing_camera(tmp_path):
    item = dict(source='v1', path='metadata/object.npz', reason='mesh absent',
                review='Confirmed absent in the pinned upstream release; mask motion.')
    manifest = dict(version=1, exclusions=[], missing_objects=[item])
    missing = [dict(path=str(tmp_path/item['path']), reason=item['reason'])]
    assert reviewed_rejections('v1', tmp_path, missing, manifest, collection='missing_objects') == [item]
    with pytest.raises(ValueError, match='unreviewed'):
        reviewed_rejections('v1', tmp_path, missing, manifest)
    with pytest.raises(ValueError, match='unreviewed'):
        reviewed_rejections('v1', tmp_path, missing, dict(version=1), collection='missing_objects')
