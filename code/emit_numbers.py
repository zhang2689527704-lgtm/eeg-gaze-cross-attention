"""Emit LaTeX macros for every number used in the manuscript, sourced from the
experiment registry files. Also audits that every \\num... token used in the
manuscript is defined and that every defined macro is used.

LaTeX macro names must contain letters only (no digits).
"""

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
MANU = os.path.join(HERE, '..', 'manuscript')

MISSING = r'\textcolor{red}{MISSING}'


def load(name, default=None):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return default
    try:
        return json.load(open(p))
    except json.JSONDecodeError:
        return default


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and (v != v)):
        return MISSING
    return f'{v:.{nd}f}'


def signed(v, nd=3):
    if v is None:
        return MISSING
    return f'{"+" if v >= 0 else "-"}{abs(v):.{nd}f}'


def pval(v):
    if v is None:
        return MISSING
    if v < 1e-3:
        mant, exp = f'{v:.1e}'.split('e')
        return f'\\ensuremath{{{mant}\\!\\times\\!10^{{{int(exp)}}}}}'
    return f'{v:.3f}'


def main():
    s = load('summary.json', {'models': {}, 'paired': {}})
    ver = load('mask_verification.json', {}) or {}
    lag = load('lagged_analysis.json', [])
    chz = load('channel_zeroout.json', {}) or {}
    grp = load('group_analysis.json', {}) or {}

    L = {}

    # Dataset-level statistics stored with the release.  These allow the
    # demographic macros to be regenerated without the 246 MB window cache.
    stats = load('dataset_stats.json', {}) or {}
    if stats:
        L['ParticipantsListed'] = str(stats['participants_listed'])
        L['Participants'] = str(stats['participants'])
        L['ParticipantsQC'] = str(stats['participants_qc'])
        L['Adhd'] = str(stats['adhd'])
        L['Control'] = str(stats['control'])
        L['Undetermined'] = str(stats['undetermined'])
        L['WindowsTotal'] = f"{stats['windows_total']:,}".replace(',', r'\,' )
        L['WindowsQC'] = f"{stats['windows_qc']:,}".replace(',', r'\,' )
        L['PctWindowsQC'] = fmt(float(stats['pct_windows_qc']), 1)
        L['DurMedian'] = fmt(float(stats['duration_median_sec']), 0)
        L['GazeMedian'] = str(int(stats['gaze_median']))
        L['GazePfive'] = str(int(stats['gaze_p5']))
        L['GazePninetyfive'] = str(int(stats['gaze_p95']))
        L['TgtColMean'] = fmt(float(stats['target_col_mean']), 2)
        L['TgtColStd'] = fmt(float(stats['target_col_std']), 2)
        L['TgtRowMean'] = fmt(float(stats['target_row_mean']), 2)
        L['TgtRowStd'] = fmt(float(stats['target_row_std']), 2)
        L['VarRatio'] = fmt(float(stats['var_ratio']), 1)
        L['PtpThreshold'] = str(stats['ptp_threshold_uv'])
        L['AgeAdhdMean'] = fmt(float(stats['age_adhd_mean']), 1)
        L['AgeAdhdSd'] = fmt(float(stats['age_adhd_sd']), 1)
        L['AgeControlMean'] = fmt(float(stats['age_control_mean']), 1)
        L['AgeControlSd'] = fmt(float(stats['age_control_sd']), 1)

    try:
        demo_path = os.path.join(HERE, '..', 'DATA',
                                 'BALLADEER ADHD DATASET',
                                 'BALLADEER ADHD DATASET',
                                 'users_demographics.json')
        demo = json.load(open(demo_path))
        L['ParticipantsListed'] = str(len(demo))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        meta = json.load(open(os.path.join(HERE, '..', 'cache', 'meta.json')))
        import numpy as np
        z = np.load(os.path.join(HERE, '..', 'cache', 'windows.npz'))
        from collections import Counter
        cnt = Counter(p['diagnosed'] for p in meta['participants'])
        durs = [p['duration_sec'] for p in meta['participants']]
        ng = z['n_gaze_raw']
        tg = z['targets']
        L['Participants'] = str(len(meta['participants']))
        L['ParticipantsQC'] = str(len(np.unique(z['win_pid'][z['qc_pass']])))
        L.setdefault('ParticipantsListed', str(len(meta['participants'])))
        L['Adhd'] = str(cnt.get('yes', 0))
        L['Control'] = str(cnt.get('no', 0))
        L['Undetermined'] = str(cnt.get('undetermined', 0))
        L['WindowsTotal'] = f"{meta['n_windows_total']:,}".replace(',', r'\,' )
        L['WindowsQC'] = f"{int(z['qc_pass'].sum()):,}".replace(',', r'\,' )
        L['PctWindowsQC'] = fmt(100 * z['qc_pass'].mean(), 1)
        L['DurMedian'] = fmt(float(np.median(durs)), 0)
        L['GazeMedian'] = str(int(np.median(ng)))
        L['GazePfive'] = str(int(np.percentile(ng, 5)))
        L['GazePninetyfive'] = str(int(np.percentile(ng, 95)))
        L['TgtColMean'] = fmt(float(tg[:, 0].mean()), 2)
        L['TgtColStd'] = fmt(float(tg[:, 0].std()), 2)
        L['TgtRowMean'] = fmt(float(tg[:, 1].mean()), 2)
        L['TgtRowStd'] = fmt(float(tg[:, 1].std()), 2)
        L['VarRatio'] = fmt(float(tg[:, 0].var() / tg[:, 1].var()), 1)
        L['PtpThreshold'] = '250'
    except FileNotFoundError:
        pass

    def model_macros(key, prefix):
        m = s['models'].get(key)
        for suffix in ['Rcol', 'RcolCi', 'Rsqcol', 'RsqcolCi', 'Rmse', 'RmseCi',
                       'Rrow', 'Rsqrow', 'Params']:
            L.setdefault(prefix + suffix, MISSING)
        if not m:
            return
        L[prefix + 'Rcol'] = fmt(m.get('r_col_mean'))
        L[prefix + 'RcolCi'] = fmt(m.get('r_col_ci95'))
        L[prefix + 'Rsqcol'] = fmt(m.get('R2_col_mean'))
        L[prefix + 'RsqcolCi'] = fmt(m.get('R2_col_ci95'))
        L[prefix + 'Rmse'] = fmt(m.get('rmse_mean'))
        L[prefix + 'RmseCi'] = fmt(m.get('rmse_ci95'))
        L[prefix + 'Rrow'] = fmt(m.get('r_row_mean'))
        L[prefix + 'Rsqrow'] = fmt(m.get('R2_row_mean'))
        if m.get('pooled'):
            L[prefix + 'RcolPooled'] = fmt(m['pooled'].get('r_col'))
            L[prefix + 'RsqcolPooled'] = fmt(m['pooled'].get('R2_col'))
        if m.get('params'):
            L[prefix + 'Params'] = f"{m['params']:,}".replace(',', r'\,' )
        else:
            L[prefix + 'Params'] = '0'

    for key, prefix in [('TempAlignEEGNet', 'Full'), ('GazeOnly', 'GazeOnly'),
                        ('EEGOnly', 'EEGOnly'), ('LSTM', 'LSTM'),
                        ('Transformer', 'Transformer'), ('LTC', 'LTC'),
                        ('WaveletTransformer', 'Wavelet'), ('NoMask', 'NoMask'),
                        ('FrontalExcluded', 'FrontalEx'), ('TempStructNet', 'TempStruct'),
                        ('PermutedEEG', 'PermEEG'), ('PermutedTarget', 'PermTgt'),
                        ('MeanPredictor', 'Mean'), ('PersistenceZero', 'PersZero'),
                        ('PersistenceTrend', 'PersTrend'), ('RidgeGaze', 'Ridge'),
                        ('BandTheta', 'BandTheta'), ('BandAlpha', 'BandAlpha'),
                        ('BandBeta', 'BandBeta'), ('LTC_param_matched', 'LTCPm'),
                        ('MaskedLSTM', 'MaskedLSTM'),
                        ('MaskedTransformer', 'MaskedTransformer')]:
        model_macros(key, prefix)

    PAIRS = [('GazeOnly', 'PairGazeOnly'), ('EEGOnly', 'PairEEGOnly'),
             ('LTC', 'PairLTC'), ('LSTM', 'PairLSTM'),
             ('Transformer', 'PairTransformer'), ('WaveletTransformer', 'PairWavelet'),
             ('NoMask', 'PairNoMask'), ('FrontalExcluded', 'PairFrontalEx'),
             ('PermutedEEG', 'PairPermEEG'), ('PermutedTarget', 'PairPermTgt'),
             ('TempStructNet', 'PairTempStruct'),
             ('RidgeGaze', 'PairRidge'), ('PersistenceTrend', 'PairPersTrend'),
             ('BandTheta', 'PairBandTheta'), ('BandAlpha', 'PairBandAlpha'),
             ('BandBeta', 'PairBandBeta')]
    for key, prefix in PAIRS:
        for suffix in ['Dr', 'P', 'CiLo', 'CiHi', 'Drsq', 'PRsq', 'Drmse', 'Prmse']:
            L.setdefault(prefix + suffix, MISSING)
        p = s.get('paired', {}).get(key, {})
        pc = p.get('r_col')
        if pc:
            L[prefix + 'Dr'] = signed(pc.get('mean_diff'))
            L[prefix + 'P'] = pval(pc.get('wilcoxon_p'))
            L[prefix + 'CiLo'] = signed(pc.get('ci_lo'))
            L[prefix + 'CiHi'] = signed(pc.get('ci_hi'))
        p2 = p.get('R2_col')
        if p2:
            L[prefix + 'Drsq'] = signed(p2.get('mean_diff'))
            L[prefix + 'PRsq'] = pval(p2.get('wilcoxon_p'))
        p3 = p.get('rmse')
        if p3:
            L[prefix + 'Drmse'] = signed(-p3.get('mean_diff'))
            L[prefix + 'Prmse'] = pval(p3.get('wilcoxon_p'))

    AUX = ['FrontalVsGazeOnly', 'TempStructVsGazeOnly', 'PermEEGVsGazeOnly',
           'LSTMvsGazeOnly', 'TransformerVsGazeOnly', 'LTCvsGazeOnly',
           'WaveletVsGazeOnly', 'RidgeVsGazeOnly', 'NoMaskVsGazeOnly']
    for k in AUX:
        for suffix in ['Dr', 'P', 'CiLo', 'CiHi']:
            L.setdefault(k + suffix, MISSING)
    auxd = s.get('paired_aux', {})
    for k in AUX:
        v = auxd.get(k)
        if v:
            L[k + 'Dr'] = signed(v.get('mean_diff'))
            L[k + 'P'] = pval(v.get('wilcoxon_p'))
            L[k + 'CiLo'] = signed(v.get('ci_lo'))
            L[k + 'CiHi'] = signed(v.get('ci_hi'))

    for k in ['MaxBlockedAttnFixed', 'MaxBlockedAttnLearned', 'LearnedDelayNorm',
              'LearnedDelayMs', 'LearnedBandwidth', 'SynDeltaMin', 'SynDeltaMax',
              'VerEpochs', 'InitDelay', 'MeanRowMass', 'RepBandwidth']:
        L.setdefault(k, MISSING)
    L['InitDelay'] = '0.15'
    fm = ver.get('fixed_mask', {})
    lm = ver.get('learned_mask', {})
    if fm:
        L['MaxBlockedAttnFixed'] = f"{fm.get('max_attention_on_blocked', 0):.1e}"
        L['MeanRowMass'] = fmt(fm.get('mean_row_mass'), 3)
        L['VerEpochs'] = str(fm.get('train_epochs', ''))
    hist_path = os.path.join(RESULTS, 'learned_structure_history.json')
    if os.path.exists(hist_path):
        hist = json.load(open(hist_path))
        L['RepBandwidth'] = fmt(hist['bandwidth'][-1])
    if lm:
        L['MaxBlockedAttnLearned'] = f"{lm.get('max_attention_on_blocked', 0):.1e}"
        L['LearnedDelayNorm'] = fmt(lm.get('delay_normalized'))
        L['LearnedDelayMs'] = fmt(lm.get('delay_ms'), 0)
        L['LearnedBandwidth'] = fmt(lm.get('bandwidth'))
    syn = ver.get('synthetic', [])
    if syn:
        L['SynDeltaMin'] = fmt(min(x['delta_learned'] for x in syn))
        L['SynDeltaMax'] = fmt(max(x['delta_learned'] for x in syn))
        words = ['One', 'Two', 'Three', 'Four', 'Five']
        offsets = sorted({x['target_offset'] for x in syn})
        for word, off in zip(words, offsets):
            vals = [x['delta_learned'] for x in syn
                    if x['target_offset'] == off]
            bws = [x['bandwidth'] for x in syn if x['target_offset'] == off]
            true = off - 1.0
            learned = sum(vals) / len(vals)
            L['SynTrue' + word] = f'{true:.2f}'
            L['SynLearned' + word] = fmt(learned)
            L['SynError' + word] = fmt(abs(learned - true))
            L['SynBeta' + word] = fmt(sum(bws) / len(bws))

    import numpy as np
    for k in ['LagN', 'LagEegGazeMedian', 'LagEegGazeSig', 'LagGazeEegMedian',
              'LagGazeEegSig', 'LagLags', 'LagHz']:
        L.setdefault(k, MISSING)
    L['LagLags'] = '4'
    L['LagHz'] = '16'
    if lag:
        imp1 = np.array([r['eeg_to_gaze_imp'] for r in lag])
        p1 = np.array([r['eeg_to_gaze_p'] for r in lag])
        imp2 = np.array([r['gaze_to_eeg_imp'] for r in lag])
        p2 = np.array([r['gaze_to_eeg_p'] for r in lag])
        L['LagN'] = str(len(lag))
        L['LagEegGazeMedian'] = fmt(100 * np.median(imp1), 1)
        L['LagEegGazeSig'] = str(int((p1 < 0.05).sum()))
        L['LagGazeEegMedian'] = fmt(100 * np.median(imp2), 1)
        L['LagGazeEegSig'] = str(int((p2 < 0.05).sum()))

    for k in ['ChOne', 'ChTwo', 'ChThree', 'ChLeast', 'ChOneVal', 'ChTwoVal',
              'ChThreeVal']:
        L.setdefault(k, MISSING)
    if chz:
        rank = chz['ranking']
        L['ChOne'], L['ChTwo'], L['ChThree'] = rank[0], rank[1], rank[2]
        L['ChLeast'] = rank[-1]
        vals = chz['mean_delta_mse']
        L['ChOneVal'] = fmt(vals[chz['channels'].index(rank[0])], 4)
        L['ChTwoVal'] = fmt(vals[chz['channels'].index(rank[1])], 4)
        L['ChThreeVal'] = fmt(vals[chz['channels'].index(rank[2])], 4)

    L.setdefault('MeanPPRcol', MISSING)
    pp = s['models'].get('TempAlignEEGNet', {}).get('per_participant', [])
    if pp:
        L['MeanPPRcol'] = fmt(float(np.mean([x['r_col'] for x in pp])))
        L['NPP'] = str(len(pp))
    L.setdefault('NPP', MISSING)

    # participant-level mean for the gaze-only variant (for the aggregation note)
    L.setdefault('GazeOnlyPPRcol', MISSING)
    gpp = s['models'].get('GazeOnly', {}).get('per_participant', [])
    if gpp:
        L['GazeOnlyPPRcol'] = fmt(float(np.mean([x['r_col'] for x in gpp])))

    for k in ['GrpAdhdN', 'GrpCtrlN', 'GrpUndetN', 'GrpAdhdR', 'GrpCtrlR',
              'GrpMwp', 'GrpOlsAdhdP', 'GrpOlsAgeP', 'GrpOlsAdhdCoef',
              'GrpOlsAgeCoef']:
        L.setdefault(k, MISSING)
    if grp:
        d = grp.get('descriptives', {})
        L['GrpAdhdN'] = str(d.get('yes', {}).get('n', ''))
        L['GrpCtrlN'] = str(d.get('no', {}).get('n', ''))
        L['GrpUndetN'] = str(d.get('undetermined', {}).get('n', ''))
        L['GrpAdhdR'] = fmt(d.get('yes', {}).get('r_col_mean'))
        L['GrpCtrlR'] = fmt(d.get('no', {}).get('r_col_mean'))
        L['GrpMwp'] = fmt(grp['mannwhitney_u']['p'])
        L['GrpOlsAdhdP'] = fmt(grp['ols_r_col']['p_adhd'])
        L['GrpOlsAgeP'] = fmt(grp['ols_r_col']['p_age'])
        L['GrpOlsAdhdCoef'] = signed(grp['ols_r_col']['coef_adhd'])
        L['GrpOlsAgeCoef'] = signed(grp['ols_r_col']['coef_age'])

    out = os.path.join(RESULTS, 'numbers.tex')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('% AUTO-GENERATED by emit_numbers.py - do not edit\n')
        for k in sorted(L):
            f.write(f'\\newcommand{{\\num{k}}}{{{L[k]}}}\n')
    print(f'wrote {out} with {len(L)} macros')

    # optional audit against a local manuscript tree
    if not os.path.isdir(MANU):
        return
    used = set()
    for path in glob.glob(os.path.join(MANU, '*.tex')):
        if path.endswith('numbers.tex'):
            continue
        with open(path, encoding='utf-8') as f:
            for m in re.finditer(r'\\num([A-Za-z]+)', f.read()):
                used.add(m.group(1))
    missing = sorted(u for u in used if u not in set(L))
    unused = sorted(d for d in L if d not in used)
    if missing:
        print('TEXT USES UNDEFINED MACROS:', ', '.join(missing))
    print('defined-but-unused (ok):', len(unused))


if __name__ == '__main__':
    main()
