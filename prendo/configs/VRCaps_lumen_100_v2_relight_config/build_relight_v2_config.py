"""B1 only: deterministic relight metadata and exact local triangle queries, no rendering."""
import argparse
import copy
import importlib.util
import json
import math
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT/'wk5/VRCaps_lumen_100_v2'
RUNTIME = Path('D:/Unity/Week5Point/20261112T004200Z')
NAME = 'VRCaps_lumen_100_v2_relight_config'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def evenly(items, n):
    return [items[int(math.floor(x+.5))] for x in np.linspace(0, len(items)-1, n)]


class Surface:
    def __init__(self, folder):
        vertices = np.fromfile(folder/'mesh_vertices_world.f32', '<f4').reshape(-1, 3).astype(float)
        vertices[:, 0] *= -1  # Same RH world as the saved OpenGL camera matrices.
        indices = np.fromfile(folder/'mesh_triangles.i32', '<i4').reshape(-1, 3)
        self.tri = vertices[indices]
        self.lo, self.hi = self.tri.min(1), self.tri.max(1)
        self.tree = cKDTree(vertices)
        self.clearance_cache = {}

    def clearance(self, point):
        key = tuple(point)
        if key in self.clearance_cache:
            return self.clearance_cache[key]
        # A nearest vertex provides a true upper bound. AABB pruning below that
        # bound cannot discard the closest triangle; it is only acceleration.
        upper = self.tree.query(point)[0]
        delta = np.maximum(np.maximum(self.lo-point, point-self.hi), 0)
        ids = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= upper**2+1e-12)
        tri = self.tri[ids]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac, ap = b-a, c-a, point-a
        normal = np.cross(ab, ac)
        normal2 = np.einsum('ij,ij->i', normal, normal)
        aa = np.einsum('ij,ij->i', ab, ab)
        bb = np.einsum('ij,ij->i', ab, ac)
        cc = np.einsum('ij,ij->i', ac, ac)
        d = np.einsum('ij,ij->i', ap, ab)
        e = np.einsum('ij,ij->i', ap, ac)
        denom = aa*cc-bb*bb
        safe = np.where(denom > 1e-24, denom, 1)
        u, v = (cc*d-bb*e)/safe, (aa*e-bb*d)/safe
        signed = np.einsum('ij,ij->i', ap, normal)
        inside = (denom > 1e-24) & (u >= 0) & (v >= 0) & (u+v <= 1)
        dist2 = np.where(inside, signed**2/np.maximum(normal2, 1e-30), np.inf)
        for x, y in [(a, b), (b, c), (c, a)]:
            edge = y-x
            t = np.clip(np.einsum('ij,ij->i', point-x, edge)/np.maximum(np.einsum('ij,ij->i', edge, edge), 1e-30), 0, 1)
            offset = point-(x+edge*t[:, None])
            dist2 = np.minimum(dist2, np.einsum('ij,ij->i', offset, offset))
        j = int(np.argmin(dist2))
        result = (float(np.sqrt(dist2[j])), int(ids[j]))
        self.clearance_cache[key] = result
        return result

    def obstruction(self, start, end):
        direction = end-start
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return {'obstructed': False, 'segment_length': length, 'first_hit_fraction': None}
        # Two-sided segment/triangle intersections, independent of mesh winding.
        candidates = np.flatnonzero(((self.hi >= np.minimum(start, end)-1e-10) &
                                      (self.lo <= np.maximum(start, end)+1e-10)).all(1))
        t = self.tri[candidates]
        e1, e2 = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
        p = np.cross(direction, e2)
        det = np.einsum('ij,ij->i', e1, p)
        ok = abs(det) > 1e-14
        inv = np.where(ok, 1/np.where(ok, det, 1), 0)
        off = start-t[:, 0]
        u = np.einsum('ij,ij->i', off, p)*inv
        q = np.cross(off, e1)
        v = q@direction*inv
        frac = np.einsum('ij,ij->i', e2, q)*inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (frac >= -1e-9) & (frac <= 1+1e-9)
        return {'obstructed': bool(hit.any()), 'segment_length': length,
                'first_hit_fraction': float(frac[hit].min()) if hit.any() else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    a = p.parse_args()
    root = a.run_root.resolve()
    assert str(root).startswith('D:\\Unity\\') and 'OneDrive' not in str(root)
    out = root/NAME
    assert not out.exists()
    (out/'relight').mkdir(parents=True)
    meta = read(BASE/'transforms.json')
    held = read(BASE/'transforms_test.json')
    actual = read(BASE/'light_settings.json')
    setup = read(BASE/'sidecars/v2_capture/setup.json')
    geometry_source = RUNTIME/'point_full'
    mesh_setup = read(geometry_source/'setup.json')
    for field in ['meshAsset', 'meshWorld', 'vertices', 'triangles', 'materialHash']:
        assert setup[field] == mesh_setup[field], field
    frames = {int(f['file_path']): f for f in meta['frames']}
    train = sorted(frames)
    test = {int(f['file_path']) for f in held['frames']}
    assert len(train) == 88 and len(test) == 12 and not set(train) & test
    main_ids = evenly(train, 30)
    # Original source-frame windows preserve every-other-frame temporal spacing.
    # Earliest and latest eligible 20-frame windows have no held-out samples.
    starts = [s for s in range(81) if all(i in frames for i in range(s, s+20, 2))]
    segments = {f'seg{j+1}': list(range(s, s+20, 2)) for j, s in enumerate([starts[0], starts[-1]])}
    assert segments['seg1'][-1] < segments['seg2'][0]
    sequence = sum(segments.values(), [])
    calib = evenly([i for i in train if i not in set(main_ids+sequence)], 5)
    selected = sorted(set(main_ids+sequence+calib))
    groups = {i: ([g for g, ids in [('main', main_ids), ('calib', calib)] if i in ids]+
                  [s for s, ids in segments.items() if i in ids]) for i in selected}
    selection = {'method': 'Deterministic; no random sampling', 'eligible_camera_split': '88 training views only',
                 'main': main_ids, 'sequence_segments': segments, 'calib': calib,
                 'main_rule': '30 evenly spaced ranks in ordered training frames; round half up',
                 'sequence_rule': 'Earliest/latest non-overlapping eligible original 20-frame windows, offsets 0,2,...,18; all sampled views must be training views',
                 'calib_rule': '5 evenly spaced ranks in remaining training views after excluding main AND sequence; round half up',
                 'main_sequence_overlap': sorted(set(main_ids)&set(sequence)),
                 'unique_selected_frames': selected,
                 'relight_split': 'calib only for calibration frames; all main/sequence frames test. These are lighting roles at training camera views, not the baseline camera split.',
                 'deduplication': 'One camera record per frame and one light record per (config,frame); overlapping L0 main/sequence requests are merged'}
    write(out/'selection.json', selection)
    copied = copy.deepcopy(meta)
    copied['frames'] = [dict(copy.deepcopy(frames[i]), frame_id=f'{i:04d}', split='calib' if i in calib else 'test', groups=groups[i]) for i in selected]
    write(out/'relight/transforms_relight.json', copied)
    shutil.copy2(BASE/'transforms.json', out/'transforms.json')
    shutil.copy2(BASE/'light_settings.json', out/'baseline_light_settings.json')
    configs = []

    def add(id, level, scale=1., color=None, offset=None, yaw=0, pitch=0, segment=None):
        spot = level == 'L2'
        configs.append({'id': id, 'level': level, 'type': 'spot' if spot else 'point', 'intensity_scale': scale,
                        'color_scale': color or [1, 1, 1], 'offset_dref': None if segment else (offset or [0, 0, 0]),
                        'yaw_deg': yaw, 'pitch_deg': pitch, 'spot_inner_deg': 0 if spot else None,
                        'spot_outer_deg': 30 if spot else None, 'unity_full_inner_angle_deg': 0 if spot else None,
                        'unity_full_spot_angle_deg': 60 if spot else None, 'HDRP_innerSpotPercent': 0 if spot else None,
                        'spot_angle_status': 'Provisional handoff default; Yutong FOV/cone review pending' if spot else None,
                        'spot_power_n': None, 'world_fixed': segment is not None,
                        'segment': segment, 'variants': ['noshadow', 'shadow'] if level in ['L3', 'L5'] else ['noshadow'],
                        'variants_status': 'Planned B2 variants; no reference images rendered'})
    add('L0', 'L0')
    for s in [.5, 2.]: add(f'L1_s{s:.1f}', 'L1', scale=s)
    add('L2_fwd', 'L2')
    for y in [10, -10, 20, -20, 40, -40]: add(f'L2_yaw{y:+d}', 'L2', yaw=y)
    for t in [20, -20]: add(f'L2_pitch{t:+d}', 'L2', pitch=t)
    for value in [.10, .25, .50, 1.00]: add(f'L3_r{value:.2f}', 'L3', offset=[value, 0, 0])
    for value in [.25, .50]: add(f'L3_u{value:.2f}', 'L3', offset=[0, value, 0])
    add('L3_f+0.25', 'L3', offset=[0, 0, .25])
    add('L3_f-0.25', 'L3', offset=[0, 0, -.25])
    for name, rgb in [('red', [1, .5, .5]), ('green', [.5, 1, .5]), ('blue', [.5, .5, 1])]: add('L4_'+name, 'L4', color=rgb)
    for seg in segments: add('L5_'+seg, 'L5', segment=seg)
    requests = {}

    def request(cfg, i, group): requests.setdefault((cfg, i), []).append(group)
    for i in main_ids:
        for cfg in configs:
            if cfg['level'] != 'L5': request(cfg['id'], i, 'main')
    for seg, ids in segments.items():
        for i in ids:
            request('L0', i, seg)
            request('L5_'+seg, i, seg)
    for i in calib:
        for cfg in ['L0', 'L2_fwd', 'L2_yaw+20', 'L3_r0.25']: request(cfg, i, 'calib')
    cfgs = {c['id']: c for c in configs}
    dref = actual['depth_reference']['d_ref_world_units']
    surface = Surface(geometry_source)
    threshold = .05*dref
    entries, queries = [], []
    print('Computing genuine L3/L5 triangle clearance and segment checks', flush=True)
    for (id, i), source_groups in sorted(requests.items()):
        cfg = cfgs[id]
        matrix = np.array(frames[i]['transform_matrix'])
        r, u, f, centre = matrix[:3, 0], matrix[:3, 1], -matrix[:3, 2], matrix[:3, 3]
        offset = cfg['offset_dref'] or [0, 0, 0]
        pos = centre+dref*(offset[0]*r+offset[1]*u+offset[2]*f)
        if cfg['world_fixed']:
            anchor = segments[cfg['segment']][0]
            pos = np.array(frames[anchor]['transform_matrix'])[:3, 3]
        yaw, pitch = np.radians([cfg['yaw_deg'], cfg['pitch_deg']])
        direction = np.cos(pitch)*(np.cos(yaw)*f+np.sin(yaw)*r)+np.sin(pitch)*u
        direction /= np.linalg.norm(direction)
        valid, reasons = True, []
        if cfg['level'] in ['L3', 'L5']:
            clearance, nearest = surface.clearance(pos)
            obstruction = surface.obstruction(centre, pos)
            if clearance < threshold: reasons.append('surface_clearance_below_0.05_dref')
            if obstruction['obstructed']: reasons.append('camera_to_light_segment_intersects_surface')
            valid = not reasons
            queries.append({'config': id, 'frame': f'{i:04d}', 'light_pos_world': pos.tolist(),
                            'minimum_surface_distance': clearance, 'nearest_triangle_index': nearest,
                            'required_clearance': threshold, **obstruction, 'valid': valid, 'invalid_reasons': reasons})
        entries.append({'config': id, 'frame': f'{i:04d}', 'light_pos_world': pos.tolist(), 'light_dir_world': direction.tolist(),
                        'valid': valid, 'invalid_reasons': reasons, 'groups': source_groups,
                        'validity_basis': 'Exact transformed mesh triangle clearance and two-sided segment intersections' if cfg['level'] in ['L3', 'L5'] else 'Unchanged validated training camera optical centre',
                        'B2_reference_generated': False})
    lights = {'mm_per_unit': None, 'd_ref': dref, 'coordinate_system': meta['pose_axes'],
              'base_light': {'type': 'point', 'intensity': actual['light']['intensity'], 'intensity_unit': actual['light']['intensity_unit'],
                             'color': [1, 1, 1], 'range': actual['light']['range_world_units'], 'offset': [0, 0, 0]},
              'spot_angle_convention': 'spot_inner_deg/spot_outer_deg are HALF angles; Unity full angles are separately recorded',
              'spot_power_n': None, 'spot_power_status': 'Pending Yutong reference calibration; no exponent fitted or guessed',
              'B1_only': True, 'configs': configs, 'frames': entries}
    write(out/'relight/lights.json', lights)
    write(out/'geometry_checks.json', {'method': 'Nearest point on transformed colon triangles plus two-sided camera-to-light segment/triangle intersection; AABBs only accelerate exact triangle queries',
                                     'mesh_triangles': len(surface.tri), 'threshold_world_units': threshold,
                                     'lumen_basis': 'Camera centres are the preserved validated lumen trajectory. L3 candidates must retain surface clearance and an unobstructed segment from that lumen witness; no bounding-box containment or global watertight solid claim is used.',
                                     'entries': queries})
    first = read(BASE/'sidecars/v2_capture/0000.json')
    fx, fy = meta['fl_x'], meta['fl_y']
    export = {'stage': 'B1 metadata only', 'reference_images_generated': False, 'depth_channel': None,
              'actual_v2_baseline': actual,
              'actual_camera': {'Unity': setup['unityVersion'], 'pipeline': 'HDRP 7.3.1', 'project_color_space': 'Linear (m_ActiveColorSpace=1)',
                                'width': meta['w'], 'height': meta['h'], 'antialiasing': 'None',
                                'intrinsics': {k: meta[k] for k in ['fl_x', 'fl_y', 'cx', 'cy']},
                                'projection_horizontal_FOV_deg': math.degrees(2*math.atan(meta['w']/(2*fx))),
                                'projection_vertical_FOV_deg': math.degrees(2*math.atan(meta['h']/(2*fy))),
                                'serialized_Unity_camera_fieldOfView_deg': first['fov']},
              'actual_training_depth': {'format': 'Unchanged float32 TIFF', 'channel': 'ViewLayer.Depth.Z', 'convention': 'axial camera-Z in consistent unverified world units', 'depth_scale': 1.0},
              'proposed_B2_only': {'status': 'NOT RENDERED; future reference work requires separate authorisation',
                                   'RGB': 'Exposed linear RGB float EXR; same fixed HDRP exposure once, no PNG gamma in EXR',
                                   'depth': 'Single-channel axial Z EXR; invalid/background zero; channel name Z proposed, not produced',
                                   'noshadow': 'Shadows off', 'shadow': 'L3/L5 only: hard shadows, highest supported resolution; effective resolution not yet set or measured',
                                   'inherit': 'Baseline intensity/exposure/range/material/pipeline unless a config explicitly changes light type, intensity, color, position or planned shadow variant',
                                   'L2': {'outer_half_deg': 30, 'inner_half_deg': 0, 'Unity_full_outer_deg': 60, 'HDRP_innerSpotPercent': 0,
                                          'status': 'Provisional table default; substantially narrower than camera FOV; Yutong confirmation/calibration pending', 'n': None}},
              'physical_scale_verified': False, 'mm_per_unit': None}
    write(out/'relight/export_settings.json', export)
    # Metadata-only load through the team's actual schema helpers; no image IO.
    spec = importlib.util.spec_from_file_location('team_relight_io', PROJECT/'Group-Project/relight_eval/relight_io.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = module.Dataset(out)
    assert len(configs) == 25 and len(cfgs) == 25
    assert Counter(c['level'] for c in configs) == {'L0': 1, 'L1': 2, 'L2': 9, 'L3': 8, 'L4': 3, 'L5': 2}
    assert not set(selected) & test and not set(calib) & set(main_ids+sequence)
    for frame in copied['frames']:
        assert frame['transform_matrix'] == frames[int(frame['file_path'])]['transform_matrix']
    for e in entries:
        assert e['frame'] in loaded.relight_frames
        assert abs(np.linalg.norm(e['light_dir_world'])-1) < 1e-12
        if not cfgs[e['config']]['world_fixed']:
            pos, direction = module.expected_light(cfgs[e['config']], frames[int(e['frame'])]['transform_matrix'], dref)
            assert np.allclose(pos, e['light_pos_world'], atol=1e-12, rtol=0)
            assert np.allclose(direction/np.linalg.norm(direction), e['light_dir_world'], atol=1e-12, rtol=0)
    planned_variants = sum(len(cfgs[e['config']]['variants']) for e in entries)
    valid_variants = sum(len(cfgs[e['config']]['variants']) for e in entries if e['valid'])
    check = {'passed': True, 'configs': len(configs), 'unique_cameras': len(selected), 'main_frames': len(main_ids),
             'sequence_frames': len(sequence), 'calibration_frames': len(calib), 'light_frame_entries': len(entries),
             'geometry_queries': len(queries), 'invalid_entries': sum(not e['valid'] for e in entries),
             'invalid_by_config': dict(Counter(e['config'] for e in entries if not e['valid'])),
             'planned_B2_variant_paths_before_validity_filter': planned_variants,
             'eligible_B2_variant_paths_after_validity_filter': valid_variants,
             'B2_RGB_or_depth_images_generated': 0, 'team_Dataset_metadata_load': 'passed',
             'copied_camera_matrices_and_intrinsics': 'exact; no held-out camera views selected',
             'prior_geometry_tests_repeated': False, 'baseline_modified': False, 'physical_scale_verified': False}
    write(out/'CHECKS.json', check)
    write(out/'generation_provenance.json', {'generated_UTC': datetime.now(timezone.utc).isoformat(),
          'source_baseline': str(BASE), 'source_geometry': str(geometry_source),
          'source_geometry_matches_v2': ['mesh asset', 'actual meshWorld transform', 'vertex/triangle counts', 'material hash'],
          'mesh_asset': setup['meshAsset'], 'meshWorld_Unity_row_major': setup['meshWorld'],
          'world_conversion': 'Reflect Unity X for mesh vertices; cameras copied from actual RH/OpenGL baseline',
          'selection': 'selection.json', 'generator': 'build_relight_v2_config.py',
          'source_handoff': 'Group-Project/handoff VR-Caps section 6 and data-flow document, read as task inputs',
          'HDRP_spot_evidence': {'version': '7.3.1', 'HDAdditionalLightData.cs': 'k_MinSpotInnerPercent=0; setter clamps to [0,100]; SetSpotAngle(60,0) uses innerSpotPercent, not legacy Light.innerSpotAngle',
                                 'LightLoop.cs': 'cos(inner half angle)=cos(full spot angle * 0.5 * innerSpotPercent01); yields half angle 0 degrees'},
          'Unity_invocations': 0, 'renders': 0, 'source_hash_sweep': False})
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    shutil.copy2(PROJECT/'wk5/code/RELIGHT_V2_CONFIG_README.md', out/'README.md')
    archive = root/(NAME+'.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file(): z.write(f, f.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as z:
        files = [f for f in out.rglob('*') if f.is_file()]
        assert len(z.infolist()) == len(files)
        assert sum(i.file_size for i in z.infolist()) == sum(f.stat().st_size for f in files)
    destination = PROJECT/'wk5'/NAME
    assert not destination.exists() and not (PROJECT/'wk5'/archive.name).exists()
    shutil.copytree(out, destination)
    shutil.copy2(archive, PROJECT/'wk5'/archive.name)
    print(json.dumps({**check, 'files': len(files), 'zip_bytes': archive.stat().st_size, 'local_root': str(root)}), flush=True)


if __name__ == '__main__':
    main()
