import socket
import json
import time
import math

# ─── CONFIG ───────────────────────────────────────────────────────────────────
UNITY_IP   = "192.168.x.x"  # ← Remplace par l'IP de ton PC Unity
UNITY_PORT = 6000
INTERVAL   = 0.1             # Envoi toutes les 100ms
# ──────────────────────────────────────────────────────────────────────────────

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[TEST] Envoi UDP vers {UNITY_IP}:{UNITY_PORT}")
    print("[TEST] Ctrl+C pour arrêter\n")

    t = 0.0
    try:
        while True:
            # Simule une respiration oscillante entre 12 et 20 rpm
            rr_simule = 16 + 4 * math.sin(t)
            hr_simule = 70 + 5 * math.sin(t * 0.5)

            data = {
                "respiration": rr_simule,
                "rr":          rr_simule,
                "hr":          hr_simule,
                "hr_brut":     hr_simule,
                "snr_hr":      10.0,
                "prom_hr":     1.0,
                "hr_ref_hz":   1.2,
                "bin":         3
            }

            payload = json.dumps(data).encode("utf-8")
            sock.sendto(payload, (UNITY_IP, UNITY_PORT))

            print(f"  → RR: {rr_simule:.1f} rpm | HR: {hr_simule:.1f} bpm")

            t += INTERVAL
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n[TEST] Arrêté.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
