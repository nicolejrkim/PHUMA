"""Convert a matrix-form SMPL-X npz (SOMA-X pose_converter --target smplx output)
into PHUMA's raw human-pose format.

Unlike Motion-X/AMASS SMPL-X (already axis-angle — PHUMA reads it near-straight
via preprocess_motionx_format.py), SOMA-X emits SMPL-X as rotation *matrices*,
so this converter mirrors that role for the matrix encoding: matrix->axis-angle
plus a 55->22 body-joint slice. SEED (Bones Studio) mocap has no SMPL data;
SOMA-X's tools/pose_converter.py fits its SOMA-skeleton clips to SMPL-X first,
producing an npz with
    target_rotations       (N, 55, 3, 3)  SMPL-X joint rotation matrices
    target_root_translation (N, 3)        y-up meters (SOMA world frame)

Output: (N, 69) float32 .npy = [transl(3), global_orient(3), body_pose(63)]
(axis-angle) — the layout of data/human_pose/*.npy. Feed the result to
src/curation/preprocess_smplx.py, then src/retarget/motion_adaptation.py.

    python src/utils/smplx_npz_to_npy.py \
        --input <clip>_smplx.npz --output data/human_pose/seed/<clip>.npy \
        [--src_fps 120 --target_fps 30]

SEED clips are 120 fps; PHUMA assumes 30 fps (its --fps default) — the
default 120->30 subsamples 4:1. To keep all frames instead, pass equal fps
here and --fps 120 to the downstream scripts.

Also reports the SOMA->SMPL-X per-vertex fit error stored in the npz.
"""
import argparse
import os
from os.path import dirname

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SOMA-X smplx npz to PHUMA 69-dim npy format")
    parser.add_argument("--input", type=str, required=True, help="SOMA-X pose_converter smplx npz")
    parser.add_argument("--output", type=str, required=True, help="Output npy path (e.g. data/human_pose/seed/<clip>.npy)")
    parser.add_argument("--src_fps", type=float, default=120.0, help="Source frame rate (SEED: 120)")
    parser.add_argument("--target_fps", type=float, default=30.0, help="Target frame rate (default: 30)")
    return parser.parse_args()


def convert_npz_to_npy(npz_path, src_fps=120.0, target_fps=30.0):
    data = np.load(npz_path, allow_pickle=True)
    rots = np.asarray(data["target_rotations"])            # (N, 55, 3, 3)
    transl = np.asarray(data["target_root_translation"])   # (N, 3)
    N = len(rots)

    step = max(1, int(round(src_fps / target_fps)))
    idx = np.arange(0, N, step)
    rots, transl = rots[idx], transl[idx]

    # SMPL-X: joint 0 = pelvis (global_orient), joints 1..21 = body_pose
    aa = R.from_matrix(rots[:, :22].reshape(-1, 3, 3)).as_rotvec().reshape(len(rots), 22, 3)
    pose = np.concatenate([transl, aa[:, 0], aa[:, 1:22].reshape(len(rots), 63)],
                          axis=1).astype(np.float32)
    assert pose.shape[1] == 69, pose.shape

    fit_error = np.asarray(data["per_vertex_error"]) if "per_vertex_error" in data.files else None
    return pose, src_fps / step, fit_error


def main(args):
    pose, out_fps, fit_error = convert_npz_to_npy(args.input, args.src_fps, args.target_fps)

    os.makedirs(dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, pose)

    msg = f"[smplx2npy] wrote {args.output}  {pose.shape} @ {out_fps:g} fps"
    if fit_error is not None:
        msg += (f" | SOMA->SMPL-X fit error: mean {fit_error.mean()*1000:.1f} mm, "
                f"max {fit_error.max()*1000:.1f} mm")
    print(msg)


if __name__ == '__main__':
    args = parse_args()
    main(args)
