# hrrmonie_cw_udp.py
# Extraction RR + HR via radar uRAD 24GHz CW + envoi UDP vers Meta Quest
# Raspberry Pi 5 — Projet Hrrmonie

import sys
import os
import signal
import math
import socket
import json
import numpy as np
from collections import deque
from time import sleep, time
from scipy.signal import butter, sosfiltfilt, find_peaks

sys.path.insert(0, '/home/xrlab-3il')
import uRAD_Pi5 as radar

# ─────────────────────────────────────────────────────────
# CONFIGURATION RADAR CW
# ─────────────────────────────────────────────────────────
MODE    = 1
F0      = 125
BW      = 240
NS      = 200
NTAR    = 3
RMAX    = 100
MTI     = 0
MTH     = 0
ALPHA   = 10
LAMBDA  = 3e8 / 24.125e9

# ─────────────────────────────────────────────────────────
# CONFIGURATION UDP
# ─────────────────────────────────────────────────────────
QUEST_IP   = "METTEZ_IP_DU_QUEST_ICI"  # ex: "192.168.1.42"
QUEST_PORT = 5005
UDP_RATE   = 0.1   # envoyer toutes les 100ms

# ─────────────────────────────────────────────────────────
# PARAMÈTRES PHYSIOLOGIQUES
# ─────────────────────────────────────────────────────────
RR_MIN_HZ  = 0.10
RR_MAX_HZ  = 0.50
HR_MIN_HZ  = 0.80
HR_MAX_HZ  = 2.50

# ─────────────────────────────────────────────────────────
# PARAMÈTRES TRAITEMENT
# ─────────────────────────────────────────────────────────
WINDOW_S         = 30.0
MIN_SAMPLES      = 150
DISPLAY_EVERY    = 1.0
COMPUTE_EVERY    = 2.0
PRESENCE_WIN_S   = 4.0
PRESENCE_VAR_THR = 1e-5
HR_MEDIAN_LEN    = 7
HR_EMA_ALPHA     = 0.20
HR_MAX_JUMP_HZ   = 0.35
BETA_FS          = 0.05
BETA_DC          = 0.01

# Bin lock
SMOOTH_ALPHA    = 0.15
REACQ_RATIO     = 1.45
SWITCH_RATIO    = 1.18
SWITCH_HOLD     = 4
LOCAL_RADIUS    = 2
DROP_RATIO      = 0.70

# ─────────────────────────────────────────────────────────
# ÉTAT GLOBAL
# ─────────────────────────────────────────────────────────
running         = True
human_present   = False
current_rr      = 0.0
current_hr      = 0.0
status_msg      = "Initialisation..."

buf_t           = deque()
buf_phi         = deque()
buf_x           = deque()

hr_history      = deque(maxlen=HR_MEDIAN_LEN)
hr_last_hz      = np.nan
hr_last_out     = np.nan
hr_invalid_cnt  = 0

fs_est          = None
idx_lock        = None
score_bins      = None
switch_cand_idx = None
switch_cand_cnt = 0

mean_I_phase    = None
mean_Q_phase    = None
phi_unwrapped   = 0.0
phi_prev        = None
drift_base      = None
DRIFT_ALPHA     = None

t_last_udp      = 0.0
udp_sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ─────────────────────────────────────────────────────────
# RESET COMPLET
# ─────────────────────────────────────────────────────────
def reset_all():
    global buf_t, buf_phi, buf_x
    global hr_history, hr_last_hz, hr_last_out, hr_invalid_cnt
    global phi_prev, phi_unwrapped, drift_base, DRIFT_ALPHA
    global mean_I_phase, mean_Q_phase
    global idx_lock, score_bins, switch_cand_idx, switch_cand_cnt
    global current_rr, current_hr

    buf_t.clear()
    buf_phi.clear()
    buf_x.clear()
    hr_history.clear()
    hr_last_hz      = np.nan
    hr_last_out     = np.nan
    hr_invalid_cnt  = 0
    phi_prev        = None
    phi_unwrapped   = 0.0
    drift_base      = None
    DRIFT_ALPHA     = None
    mean_I_phase    = None
    mean_Q_phase    = None
    idx_lock        = None
    score_bins      = None
    switch_cand_idx = None
    switch_cand_cnt = 0
    current_rr      = 0.0
    current_hr      = 0.0

# ─────────────────────────────────────────────────────────
# ARRÊT PROPRE
# ─────────────────────────────────────────────────────────
def cleanup(sig=None, frame=None):
    global running
    running = False
    print("\n\nArrêt en cours...")
    try:
        radar.turnOFF()
    except:
        pass
    try:
        udp_sock.close()
    except:
        pass
    print("Radar éteint. GPIO libérés.")
    sys.exit(0)

signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ─────────────────────────────────────────────────────────
# ENVOI UDP
# ─────────────────────────────────────────────────────────
def send_udp():
    try:
        payload = json.dumps({
            "rr":      round(current_rr, 1),
            "hr":      round(current_hr, 1),
            "present": human_present
        }).encode('utf-8')
        udp_sock.sendto(payload, (QUEST_IP, QUEST_PORT))
    except:
        pass

# ─────────────────────────────────────────────────────────
# BIN LOCK
# ─────────────────────────────────────────────────────────
def choose_bin(z_bin):
    global idx_lock, score_bins, switch_cand_idx, switch_cand_cnt

    amp = np.abs(z_bin)
    n   = len(amp)

    if score_bins is None or len(score_bins) != n:
        score_bins = np.zeros(n, dtype=np.float64)

    score_bins = (1.0 - SMOOTH_ALPHA) * score_bins + SMOOTH_ALPHA * amp

    if idx_lock is None:
        idx_lock        = int(np.argmax(score_bins))
        switch_cand_idx = None
        switch_cand_cnt = 0
        return idx_lock

    idx_lock   = int(np.clip(idx_lock, 0, n - 1))
    prev_score = float(score_bins[idx_lock]) + 1e-12

    lo        = max(0, idx_lock - LOCAL_RADIUS)
    hi        = min(n, idx_lock + LOCAL_RADIUS + 1)
    idx_local = lo + int(np.argmax(score_bins[lo:hi]))
    loc_score = float(score_bins[idx_local])

    if idx_local != idx_lock and loc_score >= SWITCH_RATIO * prev_score:
        idx_lock   = idx_local
        prev_score = loc_score

    idx_global  = int(np.argmax(score_bins))
    glob_score  = float(score_bins[idx_global])
    prev_raw    = float(amp[idx_lock]) + 1e-12
    glob_raw    = float(amp[idx_global]) + 1e-12

    strong = (
        abs(idx_global - idx_lock) > LOCAL_RADIUS and
        glob_score >= REACQ_RATIO * prev_score and
        prev_raw <= DROP_RATIO * glob_raw
    )

    if strong:
        if switch_cand_idx == idx_global:
            switch_cand_cnt += 1
        else:
            switch_cand_idx = idx_global
            switch_cand_cnt = 1
        if switch_cand_cnt >= SWITCH_HOLD:
            idx_lock        = idx_global
            switch_cand_idx = None
            switch_cand_cnt = 0
    else:
        switch_cand_idx = None
        switch_cand_cnt = 0

    return idx_lock

# ─────────────────────────────────────────────────────────
# DC REMOVAL
# ─────────────────────────────────────────────────────────
def dc_remove(I_in, Q_in):
    global mean_I_phase, mean_Q_phase

    I_in = float(I_in)
    Q_in = float(Q_in)

    if mean_I_phase is None:
        mean_I_phase = I_in
        mean_Q_phase = Q_in

    mean_I_phase = (1 - BETA_DC) * mean_I_phase + BETA_DC * I_in
    mean_Q_phase = (1 - BETA_DC) * mean_Q_phase + BETA_DC * Q_in

    return I_in - mean_I_phase, Q_in - mean_Q_phase

# ─────────────────────────────────────────────────────────
# UNWRAP INCRÉMENTAL
# ─────────────────────────────────────────────────────────
def unwrap_inc(phi):
    global phi_unwrapped, phi_prev

    phi = float(phi)
    if phi_prev is None:
        phi_prev      = phi
        phi_unwrapped = phi
        return phi_unwrapped

    d = phi - phi_prev
    if d >  math.pi:  d -= 2 * math.pi
    elif d < -math.pi: d += 2 * math.pi

    phi_unwrapped += d
    phi_prev       = phi
    return phi_unwrapped

# ─────────────────────────────────────────────────────────
# SUPPRESSION DÉRIVE
# ─────────────────────────────────────────────────────────
def suppress_drift(x):
    global drift_base, DRIFT_ALPHA

    x = float(x)

    if drift_base is None or DRIFT_ALPHA is None:
        drift_base  = x
        DRIFT_ALPHA = 1.0 / max(1.0, 10.0 * fs_est) if fs_est else 0.001
        return 0.0

    drift_base = (1.0 - DRIFT_ALPHA) * drift_base + DRIFT_ALPHA * x
    return x - drift_base

# ─────────────────────────────────────────────────────────
# FILTRES
# ─────────────────────────────────────────────────────────
def bandpass(sig, f_lo, f_hi, fs, order=4):
    sig = np.asarray(sig, dtype=np.float64)
    nyq = fs / 2.0
    lo  = np.clip(f_lo / nyq, 0.001, 0.999)
    hi  = np.clip(f_hi / nyq, 0.001, 0.999)
    if lo >= hi or len(sig) < 3 * (2 * order) + 1:
        return sig
    sos = butter(order, [lo, hi], btype='band', output='sos')
    return sosfiltfilt(sos, sig)

def highpass(sig, fc, fs, order=4):
    sig = np.asarray(sig, dtype=np.float64)
    nyq = fs / 2.0
    f   = np.clip(fc / nyq, 0.001, 0.999)
    if len(sig) < 3 * (2 * order) + 1:
        return sig
    sos = butter(order, f, btype='high', output='sos')
    return sosfiltfilt(sos, sig)

def notch_harmonics(sig, fs, f_rr, max_harm=5, bw=0.06):
    x   = np.asarray(sig, dtype=np.float64).copy()
    nyq = fs / 2.0
    if not np.isfinite(f_rr) or f_rr <= 0:
        return x
    for k in range(2, max_harm + 1):
        f0 = k * f_rr
        lo = f0 - bw
        hi = f0 + bw
        if lo <= 0 or hi >= nyq:
            continue
        try:
            x = x - bandpass(x, lo, hi, fs, order=2)
        except:
            pass
    return x

# ─────────────────────────────────────────────────────────
# ESTIMATION FRÉQUENCE
# ─────────────────────────────────────────────────────────
def estimate_freq(sig, fs, f_lo, f_hi):
    sig = np.asarray(sig, dtype=np.float64)
    n   = len(sig)

    if n < int(fs * 5):
        return np.nan, 0.0

    sig  = sig - np.mean(sig)
    win  = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(max(n, 1))))
    X    = np.fft.rfft(sig * win, n=nfft)
    P    = np.abs(X) ** 2
    f    = np.fft.rfftfreq(nfft, d=1.0/fs)

    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return np.nan, 0.0

    P_b  = P[mask]
    f_b  = f[mask]
    pmin = 0.05 * np.max(P_b)
    peaks, _ = find_peaks(P_b, prominence=pmin)

    if len(peaks) == 0:
        k = int(np.argmax(P_b))
    else:
        k = int(peaks[np.argmax(P_b[peaks])])

    peak_f = float(f_b[k])
    peak_p = float(P_b[k])
    noise  = float(np.median(P_b)) + 1e-12
    snr    = peak_p / noise

    if snr < 2.0:
        return np.nan, snr

    return peak_f, snr

# ─────────────────────────────────────────────────────────
# DÉTECTION DE PRÉSENCE
# ─────────────────────────────────────────────────────────
def check_presence():
    global status_msg

    if fs_est is None or len(buf_x) < 5:
        status_msg = f"Initialisation... ({len(buf_phi)}/{MIN_SAMPLES})"
        return False

    n_win  = max(5, min(int(PRESENCE_WIN_S * fs_est), len(buf_x)))
    recent = np.array(list(buf_x)[-n_win:])
    var    = float(np.var(recent))

    if var > PRESENCE_VAR_THR:
        status_msg = f"Présence détectée (var={var:.2e})"
        return True
    else:
        status_msg = f"Aucune présence (var={var:.2e})"
        return False

# ─────────────────────────────────────────────────────────
# CALCUL SIGNES VITAUX
# ─────────────────────────────────────────────────────────
def compute_vitals():
    global current_rr, current_hr
    global hr_history, hr_last_hz, hr_last_out, hr_invalid_cnt

    fs   = fs_est
    data = np.array(list(buf_phi), dtype=np.float64)
    n    = len(data)

    if n < int(fs * 8):
        return

    phi_hp = highpass(data, 0.05, fs)

    # ── RESPIRATION ──────────────────────────────────────
    sig_rr        = bandpass(phi_hp, RR_MIN_HZ, RR_MAX_HZ, fs)
    rr_hz, rr_snr = estimate_freq(sig_rr, fs, RR_MIN_HZ, RR_MAX_HZ)

    if np.isfinite(rr_hz):
        current_rr = rr_hz * 60.0

    # ── DÉRIVÉE DE PHASE ─────────────────────────────────
    phi_diff = np.diff(phi_hp, prepend=phi_hp[0])

    # ── FRÉQUENCE CARDIAQUE ───────────────────────────────
    if np.isfinite(hr_last_hz):
        hr_lo = max(HR_MIN_HZ, hr_last_hz - HR_MAX_JUMP_HZ)
        hr_hi = min(HR_MAX_HZ, hr_last_hz + HR_MAX_JUMP_HZ)
    else:
        hr_lo = HR_MIN_HZ
        hr_hi = HR_MAX_HZ

    sig_hr = bandpass(phi_diff, hr_lo, hr_hi, fs)

    if np.isfinite(rr_hz):
        sig_hr = notch_harmonics(sig_hr, fs, rr_hz)

    hr_hz, hr_snr = estimate_freq(sig_hr, fs, hr_lo, hr_hi)

    if not np.isfinite(hr_hz):
        sig_hr2       = bandpass(phi_diff, HR_MIN_HZ, HR_MAX_HZ, fs)
        if np.isfinite(rr_hz):
            sig_hr2   = notch_harmonics(sig_hr2, fs, rr_hz)
        hr_hz, hr_snr = estimate_freq(sig_hr2, fs, HR_MIN_HZ, HR_MAX_HZ)

    if np.isfinite(hr_hz):
        hr_invalid_cnt = 0
        hr_history.append(hr_hz * 60.0)
        hr_last_hz = hr_hz
        vals = [v for v in hr_history if np.isfinite(v)]
        if vals:
            hr_med = float(np.median(vals))
            if np.isfinite(hr_last_out):
                hr_last_out = ((1.0 - HR_EMA_ALPHA) * hr_last_out
                               + HR_EMA_ALPHA * hr_med)
            else:
                hr_last_out = hr_med
            current_hr = hr_last_out
    else:
        hr_invalid_cnt += 1
        if hr_invalid_cnt >= 5:
            hr_history.clear()
            hr_last_hz  = np.nan
            hr_last_out = np.nan

# ─────────────────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────────────────
def display():
    os.system('clear')
    buf_s = len(buf_phi) / fs_est if fs_est else 0

    print("=" * 56)
    print("      HRRMONIE — Signes Vitaux + UDP → Quest")
    print("=" * 56)
    print(f"  Quest IP : {QUEST_IP}:{QUEST_PORT}")
    print("-" * 56)

    if not human_present:
        print(f"\n  🔴  {status_msg}")
        print("\n  Fréquence Respiratoire :    ---   breaths/min")
        print("  Fréquence Cardiaque    :    ---   BPM")
    else:
        rr_s = f"{current_rr:.1f}" if current_rr > 0 else "calcul..."
        hr_s = f"{current_hr:.1f}" if current_hr > 0 else "calcul..."
        print(f"\n  🟢  {status_msg}")
        print(f"\n  Fréquence Respiratoire :  {rr_s:>7}  breaths/min")
        print(f"  Fréquence Cardiaque    :  {hr_s:>7}  BPM")

    print(f"\n  Buffer  : {buf_s:.1f}s / {WINDOW_S:.0f}s")
    if fs_est:
        print(f"  Fs réel : {fs_est:.1f} Hz")
    print("\n" + "=" * 56)
    print("  Ctrl+C pour arrêter")
    print("=" * 56)

# ─────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────
def main():
    global human_present, fs_est, running
    global t_last_udp

    print("Démarrage radar uRAD CW...")
    ret = radar.turnON()
    if ret != 0:
        print(f"ERREUR turnON: {ret}")
        sys.exit(1)
    sleep(0.5)

    ret = radar.loadConfiguration(
        mode=MODE, f0=F0, BW=BW, Ns=NS, Ntar=NTAR,
        Rmax=RMAX, MTI=MTI, Mth=MTH, Alpha=ALPHA,
        distance_true=False, velocity_true=False,
        SNR_true=False, I_true=True, Q_true=True,
        movement_true=False
    )
    if ret != 0:
        print(f"ERREUR config: {ret}")
        radar.turnOFF()
        sys.exit(1)

    print(f"Radar prêt — envoi UDP vers {QUEST_IP}:{QUEST_PORT}\n")
    sleep(0.3)

    t_last_display  = time()
    t_last_compute  = time()
    t_last_presence = time()
    t_last_udp      = time()
    t_prev          = time()

    while running:

        # ── Acquisition ──────────────────────────────────
        ret, _, IQ = radar.detection()

        if ret == 0 and len(IQ[0]) > 0:
            I_raw = np.asarray(IQ[0], dtype=np.float64)
            Q_raw = np.asarray(IQ[1], dtype=np.float64)
            z_bin = I_raw + 1j * Q_raw

            idx = choose_bin(z_bin)
            lo  = max(0, idx - 1)
            hi  = min(len(z_bin), idx + 2)
            z_s = np.mean(z_bin[lo:hi])

            I_c, Q_c = dc_remove(float(np.real(z_s)),
                                  float(np.imag(z_s)))

            phi   = math.atan2(Q_c, I_c)
            phi_u = unwrap_inc(phi)
            x_m   = (LAMBDA / (4 * math.pi)) * phi_u

            now    = time()
            dt     = now - t_prev
            t_prev = now

            if dt > 0:
                fs_inst = 1.0 / dt
                fs_est  = fs_inst if fs_est is None else \
                          (1 - BETA_FS) * fs_est + BETA_FS * fs_inst

            x_hp = suppress_drift(x_m)

            buf_t.append(now)
            buf_phi.append(phi_u)
            buf_x.append(x_hp)

            while len(buf_t) > 1 and \
                  (buf_t[-1] - buf_t[0]) > WINDOW_S:
                buf_t.popleft()
                buf_phi.popleft()
                buf_x.popleft()

        now = time()

        # ── Présence (chaque seconde) ─────────────────────
        if (now - t_last_presence) >= 1.0:
            was_present   = human_present
            human_present = check_presence()
            if was_present and not human_present:
                reset_all()
            t_last_presence = now

        # ── Calcul signes vitaux (toutes les 2s) ──────────
        if (now - t_last_compute) >= COMPUTE_EVERY \
                and human_present \
                and fs_est is not None \
                and len(buf_phi) >= MIN_SAMPLES:
            compute_vitals()
            t_last_compute = now

        # ── Envoi UDP (toutes les 100ms) ──────────────────
        if (now - t_last_udp) >= UDP_RATE:
            send_udp()
            t_last_udp = now

        # ── Affichage (chaque seconde) ────────────────────
        if (now - t_last_display) >= DISPLAY_EVERY:
            display()
            t_last_display = now

if __name__ == "__main__":
    main()
