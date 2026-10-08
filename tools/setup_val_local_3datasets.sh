#!/usr/bin/env bash
set -euo pipefail

VAL_ROOT="${VAL_ROOT:-/data1/zbf/d4rt_val_datasets}"
POINTODYSSEY_SRC="${POINTODYSSEY_SRC:-/data2/d4rt/datasets/PointOdyssey}"
DYNAMIC_REPLICA_SRC="${DYNAMIC_REPLICA_SRC:-/data1/d4rt/datasets/Dynamic_Replica}"
BLENDEDMVS_SRC="${BLENDEDMVS_SRC:-/data/d4rt/data/BlendedMVS}"
LOG_ROOT="${LOG_ROOT:-/data1/zbf/d4rt_val_copy_logs}"

mkdir -p "$VAL_ROOT" "$LOG_ROOT"

echo "[setup_val_local] VAL_ROOT=$VAL_ROOT"
echo "[setup_val_local] POINTODYSSEY_SRC=$POINTODYSSEY_SRC"
echo "[setup_val_local] DYNAMIC_REPLICA_SRC=$DYNAMIC_REPLICA_SRC"
echo "[setup_val_local] BLENDEDMVS_SRC=$BLENDEDMVS_SRC"
echo "[setup_val_local] LOG_ROOT=$LOG_ROOT"
echo "[setup_val_local] Co3Dv2 is intentionally excluded from local validation."
echo "[setup_val_local] BlendedMVS validation uses validation_list.txt only."

if [[ ! -d "$POINTODYSSEY_SRC/val" ]]; then
  echo "[setup_val_local] missing PointOdyssey val: $POINTODYSSEY_SRC/val" >&2
  exit 1
fi
if [[ ! -d "$DYNAMIC_REPLICA_SRC/valid" ]]; then
  echo "[setup_val_local] missing Dynamic_Replica valid: $DYNAMIC_REPLICA_SRC/valid" >&2
  exit 1
fi
if [[ ! -f "$BLENDEDMVS_SRC/validation_list.txt" ]]; then
  echo "[setup_val_local] missing BlendedMVS validation_list.txt: $BLENDEDMVS_SRC/validation_list.txt" >&2
  exit 1
fi

systemctl stop d4rt-val-copy-pointodyssey 2>/dev/null || true
systemd-run \
  --unit=d4rt-val-copy-pointodyssey \
  --description="D4RT local copy PointOdyssey val" \
  --setenv=HOME=/root \
  --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --collect \
  /bin/bash -lc "\
    mkdir -p '$VAL_ROOT/PointOdyssey' '$LOG_ROOT' && \
    exec rsync -a --info=progress2 \
      '$POINTODYSSEY_SRC/val/' \
      '$VAL_ROOT/PointOdyssey/val/' \
      >> '$LOG_ROOT/pointodyssey_val.log' 2>&1"

systemctl stop d4rt-val-copy-dynamic-replica 2>/dev/null || true
systemd-run \
  --unit=d4rt-val-copy-dynamic-replica \
  --description="D4RT local copy Dynamic_Replica valid" \
  --setenv=HOME=/root \
  --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --collect \
  /bin/bash -lc "\
    mkdir -p '$VAL_ROOT/Dynamic_Replica' '$LOG_ROOT' && \
    exec rsync -a --info=progress2 \
      '$DYNAMIC_REPLICA_SRC/valid/' \
      '$VAL_ROOT/Dynamic_Replica/valid/' \
      >> '$LOG_ROOT/dynamic_replica_valid.log' 2>&1"

systemctl stop d4rt-val-copy-blendedmvs 2>/dev/null || true
systemd-run \
  --unit=d4rt-val-copy-blendedmvs \
  --description="D4RT local copy BlendedMVS validation scenes" \
  --setenv=HOME=/root \
  --setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --collect \
  /bin/bash -lc "\
    set -euo pipefail
    mkdir -p '$VAL_ROOT/BlendedMVS' '$LOG_ROOT'
    cp '$BLENDEDMVS_SRC/validation_list.txt' '$VAL_ROOT/BlendedMVS/validation_list.txt'
    while IFS= read -r scene; do
      [[ -z \"\$scene\" ]] && continue
      rsync -a --info=progress2 \
        '$BLENDEDMVS_SRC/'\"\$scene\"'/' \
        '$VAL_ROOT/BlendedMVS/'\"\$scene\"'/' \
        >> '$LOG_ROOT/blendedmvs_val.log' 2>&1
    done < '$BLENDEDMVS_SRC/validation_list.txt'"

echo "[setup_val_local] started background copy units:"
echo "  systemctl status d4rt-val-copy-pointodyssey"
echo "  systemctl status d4rt-val-copy-dynamic-replica"
echo "  systemctl status d4rt-val-copy-blendedmvs"
echo "[setup_val_local] logs: $LOG_ROOT"
