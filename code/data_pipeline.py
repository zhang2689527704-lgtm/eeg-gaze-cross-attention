"""Build the frozen BALLADEER window cache.

Gaze context is resampled to 50 uniformly spaced points over each 2 s window
by linear interpolation (t_k = t0 + k * window/50). EEG windows are rejected
when peak-to-peak amplitude exceeds 250 uV. All downstream experiments read
the same cache, so windows, folds, and normalization cannot drift apart.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

warnings.filterwarnings('ignore')

CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
# Ocular-artifact-sensitive frontal channels (excluded in the frontal-excluded ablation)
FRONTAL_SENSITIVE = ['AF3', 'AF4', 'F7', 'F8', 'F3', 'F4']

EEG_SR = 128.0
WINDOW_SEC = 2.0
STEP_SEC = 0.5
HORIZON_SEC = 0.5
N_GAZE = 50
PTP_THRESHOLD_UV = 250.0

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (REPO_ROOT / 'DATA' / 'BALLADEER ADHD DATASET'
                / 'BALLADEER ADHD DATASET')
CACHE_DIR = REPO_ROOT / 'cache'


def _butter_bandpass(lo, hi, fs, order=4):
    nyq = fs / 2
    return signal.butter(order, [lo / nyq, hi / nyq], btype='band')


class BalladeerData:
    def __init__(self, root=None, cache_dir=None,
                 eeg_sr=EEG_SR, window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                 n_gaze=N_GAZE, ptp_threshold_uv=PTP_THRESHOLD_UV):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.eeg_sr = eeg_sr
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.n_gaze = n_gaze
        self.ptp_threshold_uv = ptp_threshold_uv
        self.window_samples = int(window_sec * eeg_sr)
        self.step_samples = int(step_sec * eeg_sr)
        self.b_bp = _butter_bandpass(1.0, 45.0, eeg_sr)

    def demographics(self):
        with open(self.root / 'users_demographics.json') as f:
            data = json.load(f)
        return {u['user']: u for u in data}

    def find_files(self, user):
        user_dir = self.root / user / 'AttentionRobotsDesktop'
        if not user_dir.exists():
            return {}
        sessions = [d for d in user_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not sessions:
            return {}
        session_dir = sessions[0]
        files = {}
        for f in session_dir.iterdir():
            name = f.name
            if 'EPOC' in name and name.endswith('.csv') and 'EYE' not in name and 'GAME' not in name:
                files['eeg'] = str(f)
            elif 'EYE_TRACKING' in name:
                files['eyetracking'] = str(f)
        return files

    def preprocess_eeg(self, eeg_raw):
        """Detrend -> 1-45 Hz bandpass -> regression-based EOG correction.

        AF3/AF4 serve as EOG proxies; their linear contribution is regressed
        out of the remaining channels.
        """
        eeg = signal.detrend(eeg_raw, axis=0)
        eeg = signal.filtfilt(*self.b_bp, eeg, axis=0)

        eog_idx = [0, 13]  # AF3, AF4
        eog = eeg[:, eog_idx]
        for ch in range(eeg.shape[1]):
            if ch in eog_idx:
                continue
            beta = np.linalg.lstsq(eog, eeg[:, ch], rcond=None)[0]
            eeg[:, ch] = eeg[:, ch] - eog @ beta
        return eeg.astype(np.float32)

    def participant_arrays(self, user):
        files = self.find_files(user)
        if 'eeg' not in files or 'eyetracking' not in files:
            return None
        with open(files['eeg']) as f:
            meta_line = f.readline()
        session_start = None
        if 'start timestamp' in meta_line:
            try:
                session_start = float(meta_line.split('start timestamp:')[1].split(',')[0].strip())
            except (IndexError, ValueError):
                pass
        eeg_df = pd.read_csv(files['eeg'], skiprows=1)
        eeg_t = eeg_df['Timestamp'].values.astype(np.float64)
        eeg_raw = eeg_df[[f'EEG.{ch}' for ch in CHANNELS]].values.astype(np.float64)
        interp_flag = eeg_df['EEG.Interpolated'].values.astype(np.float64) if 'EEG.Interpolated' in eeg_df.columns else None

        gaze_df = pd.read_csv(files['eyetracking'])
        t0 = session_start if session_start is not None else eeg_t[0]
        gaze_t = t0 + gaze_df['timeChecked'].values.astype(np.float64)
        gaze_col = gaze_df['looked_col'].values.astype(np.float64)
        gaze_row = gaze_df['looked_row'].values.astype(np.float64)

        eeg = self.preprocess_eeg(eeg_raw)
        return {
            'eeg': eeg, 'eeg_t': eeg_t,
            'gaze_t': gaze_t, 'gaze_col': gaze_col, 'gaze_row': gaze_row,
            'duration_sec': float(eeg_t[-1] - eeg_t[0]),
            'interp_frac': float(np.mean(interp_flag)) if interp_flag is not None else 0.0,
        }

    def create_windows(self, arr):
        """Sliding windows with resampled 50-point gaze contexts.

        Returns list of window dicts. Target = change from the last gaze
        sample inside the window to the first gaze sample after the window
        ends (a forward-prediction target; no causal interpretation implied).
        """
        eeg, eeg_t = arr['eeg'], arr['eeg_t']
        gaze_t, gcol, grow = arr['gaze_t'], arr['gaze_col'], arr['gaze_row']
        n = len(eeg)
        win, step = self.window_samples, self.step_samples
        horizons = int(HORIZON_SEC * self.eeg_sr)

        offsets = np.arange(self.n_gaze) * (self.window_sec / self.n_gaze)

        order = np.argsort(gaze_t)
        gaze_t, gcol, grow = gaze_t[order], gcol[order], grow[order]

        windows = []
        for start in range(0, n - win - horizons, step):
            end = start + win
            t0, t1 = eeg_t[start], eeg_t[end - 1]

            m = (gaze_t >= t0) & (gaze_t <= t1)
            n_raw = int(m.sum())
            if n_raw < 2:
                continue

            gt, gc, gr = gaze_t[m], gcol[m], grow[m]
            tk = t0 + offsets
            col50 = np.interp(tk, gt, gc).astype(np.float32)
            row50 = np.interp(tk, gt, gr).astype(np.float32)

            f = gaze_t > t1
            if not np.any(f):
                continue
            idx = np.argmax(f)
            delta = np.array([gcol[idx] - gc[-1], grow[idx] - gr[-1]], dtype=np.float32)

            eeg_win = eeg[start:end]
            ptp = float(np.max(np.ptp(eeg_win, axis=0)))
            windows.append({
                'pid': -1,          # filled by caller
                'start': start,
                'target': delta,
                'gaze50': np.stack([col50, row50], axis=-1),
                'n_gaze_raw': n_raw,
                'ptp': ptp,
                'qc_pass': ptp <= self.ptp_threshold_uv,
            })
        return windows

    def build(self, usernames=None):
        demo = self.demographics()
        if usernames is None:
            usernames = sorted(demo.keys())

        part_idx, eeg_arrays, offsets = {}, [], [0]
        windows = []
        meta = {'participants': [], 'channels': CHANNELS,
                'config': {'eeg_sr': self.eeg_sr, 'window_sec': self.window_sec,
                           'step_sec': self.step_sec, 'n_gaze': self.n_gaze,
                           'ptp_threshold_uv': self.ptp_threshold_uv,
                           'eog_correction': 'AF3/AF4 regression'}}
        for user in usernames:
            arr = self.participant_arrays(user)
            if arr is None or len(arr['eeg']) < self.window_samples:
                print(f'  {user}: skipped (no data)')
                continue
            pid = len(part_idx)
            part_idx[user] = pid
            w = self.create_windows(arr)
            if not w:
                continue
            for ww in w:
                ww['pid'] = pid
            windows.extend(w)
            eeg_arrays.append(arr['eeg'])
            offsets.append(offsets[-1] + len(arr['eeg']))
            meta['participants'].append({
                'user': user, 'pid': pid,
                'diagnosed': demo[user].get('diagnosed', 'unknown'),
                'age': demo[user].get('age', None),
                'duration_sec': arr['duration_sec'],
                'interp_frac': arr['interp_frac'],
                'n_windows': len(w),
            })
            print(f'  {user}: {len(w)} windows, dur={arr["duration_sec"]:.1f}s')

        eeg_all = np.concatenate(eeg_arrays, axis=0)
        meta['participants_users'] = list(part_idx.keys())
        meta['eeg_offsets'] = offsets
        meta['n_windows_total'] = len(windows)

        np.savez_compressed(
            self.cache_dir / 'windows.npz',
            eeg=eeg_all.astype(np.float32),
            win_pid=np.array([w['pid'] for w in windows], dtype=np.int32),
            win_start=np.array([w['start'] for w in windows], dtype=np.int32),
            targets=np.stack([w['target'] for w in windows]),
            gaze50=np.stack([w['gaze50'] for w in windows]),
            n_gaze_raw=np.array([w['n_gaze_raw'] for w in windows], dtype=np.int32),
            ptp=np.array([w['ptp'] for w in windows], dtype=np.float32),
            qc_pass=np.array([w['qc_pass'] for w in windows], dtype=bool),
        )
        with open(self.cache_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'\nCached {len(windows)} windows from {len(meta["participants"])} participants')
        qc = np.array([w['qc_pass'] for w in windows])
        print(f'QC pass: {qc.sum()} ({100*qc.mean():.1f}%), rejected: {(~qc).sum()}')
        return meta


if __name__ == '__main__':
    d = BalladeerData()
    d.build()
