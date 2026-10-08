import json

import pytest

from scripts import continue_background_recovery as recovery


def controller(tmp_path):
    job=dict(repo=str(tmp_path),control_dir=str(tmp_path),output_root=str(tmp_path/'runs'),
             python='python',data_config=str(tmp_path/'filtered.yaml'),target_step=50000)
    path=tmp_path/'job.json';path.write_text(json.dumps(job))
    return recovery.BackgroundRecoveryController(path)


@pytest.mark.parametrize('first',[True,False])
def test_filtered_resume_preserves_cursor_and_moments_with_only_first_phase_lr_override(tmp_path,monkeypatch,first):
    c=controller(tmp_path);c.progress['restored_training_state']={'loss_config':{'same':True}}
    state=dict(dense_target_version=2,kubric_background_policy='objects_ground',loss_config={'same':True})
    monkeypatch.setattr(recovery,'validate_checkpoint',lambda path,step:state)
    calls=[];c.run_process=lambda name,argv:calls.append((name,argv))
    checkpoint=c.train_filtered(tmp_path/'old.pt',45000,45200,first)
    command=calls[0][1]
    assert '--resume' in command and '--pretrained' not in command
    assert '--reset-data-cursor' not in command
    assert ('--override-lr' in command)==first
    assert '--allow-shared' in command and command[command.index('--gpus')+1]=='4,5'
    assert '--no-query-chunking' in command
    if first:
        assert command[command.index('--lr-end-step')+1]=='50000'
        assert command[command.index('--warmup-steps')+1]=='100'
    assert checkpoint.name=='latest.pt'
    assert c.progress['resumed_training_state']==state


def test_migration_rejects_checkpoint_without_corrected_protocol(tmp_path,monkeypatch):
    c=controller(tmp_path);c.progress['restored_training_state']={'loss_config':{}}
    monkeypatch.setattr(recovery,'validate_checkpoint',lambda path,step:{'loss_config':{}})
    c.run_process=lambda *args:None
    with pytest.raises(RuntimeError,match='corrected target protocol'):
        c.train_filtered(tmp_path/'old.pt',45000,45200,True)
