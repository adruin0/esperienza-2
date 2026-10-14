import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.stats import norm


def getData(filepath):
    df = pd.read_excel(filepath)
    x, y, dx, dy = df.to_numpy(dtype=float).T
    labels = list(df.columns)
    return x, y, dx, dy, labels


def retta(params, x):
    m, q = params
    return m * x + q


def pulisci(nome):
    return nome.split("=")[-1].strip()


def chiQuadroLinare(x, y, m, q, dx, dy):
    residui = (y - m * x - q) ** 2
    errori = dy**2 + m**2 * dx**2
    return np.sum(residui / errori)


def parametriFitODR(x, y, dx, dy, stima_iniziale=(-1.1, 7.5)):
    modello = Model(retta)
    sx = np.where(dx > 0, dx, 1e-12)
    sy = np.where(dy > 0, dy, 1e-12)
    dati = RealData(x, y, sx=sx, sy=sy)
    odr = ODR(dati, modello, beta0=stima_iniziale)
    risultato = odr.run()

    m, q = risultato.beta
    dm, dq = np.sqrt(np.diag(risultato.cov_beta))
    cov_mq = risultato.cov_beta[0, 1]

    return m, q, dm, dq, cov_mq


def residui(x, y, dx, dy, m, q):
    r = y - retta((m, q), x)
    sr = np.sqrt(dy**2 + m**2 * dx**2)
    return r, sr


def consistenzaStatistica(a, b, da, db, k):
    discrepanza = abs(a - b)
    sigma_d = np.sqrt(da**2 + db**2)
    return discrepanza <= k * sigma_d


if __name__ == "__main__":
    # Lettura dati: x = I_R [mA], y = V_R [V], dx = dI_R [mA], dy = dV_R [V]
    # N.B. Con I_R in mA e V_R in V, la resistenza R_T risulta espressa in kOhm.
    x, y, dx, dy, nomi = getData(sys.argv[1])
    k = 3

    nomi = [pulisci(n) for n in nomi]
    print(f"Caricati {len(x)} punti da {sys.argv[1]}")

    # ==========================================================================
    # PARAMETRI SPERIMENTALI DEL PARTITORE (da adattare ai valori di laboratorio)
    # ==========================================================================
    V0, dV0 = 15.0, 0.01    # f.e.m. del generatore V0 [V]
    R1, dR1 = 2.193, 0.006    # R1 del partitore [kOhm]
    R2, dR2 = 2.164, 0.006    # R2 del partitore [kOhm]
    unita_R = "kΩ"
    # ==========================================================================

    # Previsioni teoriche dal teorema di Thévenin:
    # V' = E_T_teo = V0 * R2 / (R1 + R2)
    E_teo = V0 * R2 / (R1 + R2)
    dE_teo = np.sqrt(
        (R2 / (R1 + R2) * dV0) ** 2
        + (V0 * R2 / (R1 + R2) ** 2 * dR1) ** 2
        + (V0 * R1 / (R1 + R2) ** 2 * dR2) ** 2
    )

    # R0 = R_T_teo = (R1 * R2) / (R1 + R2)
    R_teo = (R1 * R2) / (R1 + R2)
    dR_teo = np.sqrt(
        ((R2 / (R1 + R2)) ** 2 * dR1) ** 2
        + ((R1 / (R1 + R2)) ** 2 * dR2) ** 2
    )

    m0, q0 = np.polyfit(x, y, 1)
    m, q, dm, dq, cov_mq = parametriFitODR(x, y, dx, dy, stima_iniziale=(m0, q0))

    # Da V_R = E_T - R_T * I_R  -->  E_T = q, R_T = -m
    E_T, dE_T = q, dq
    R_T, dR_T = -m, dm

    # Corrente di corto circuito estrapolata: I_cc = E_T / R_T = -q / m
    I_cc = E_T / R_T
    var_Icc = (1.0 / m**2) * (dq**2) + (q**2 / m**4) * (dm**2) - 2.0 * (q / m**3) * cov_mq
    dI_cc = np.sqrt(max(var_Icc, 0.0))

    z_E = abs(E_T - E_teo) / np.sqrt(dE_T**2 + dE_teo**2)
    z_R = abs(R_T - R_teo) / np.sqrt(dR_T**2 + dR_teo**2)

    print("=========================================")
    print("Thévenin - Partitore resistivo:")
    print(f"E_T (fit)         = {E_T:.4f} ± {dE_T:.4f} V")
    print(f"V'  (teorico)     = {E_teo:.4f} ± {dE_teo:.4f} V")
    print(f"R_T (fit)         = {R_T:.4f} ± {dR_T:.4f} {unita_R}")
    print(f"R0  (teorico)     = {R_teo:.4f} ± {dR_teo:.4f} {unita_R}")
    print(f"I_cc (estrapolata)= {I_cc:.3f} ± {dI_cc:.3f} mA")
    print("=========================================")
    print(f"Discrepanza E_T vs V': {z_E:.2f} sigma")
    if consistenzaStatistica(E_T, E_teo, dE_T, dE_teo, k):
        print(f"-> E_T e V' risultano compatibili entro {k} sigma")
    else:
        print(f"-> E_T e V' NON risultano compatibili entro {k} sigma")

    print(f"Discrepanza R_T vs R0: {z_R:.2f} sigma")
    if consistenzaStatistica(R_T, R_teo, dR_T, dR_teo, k):
        print(f"-> R_T e R0 risultano compatibili entro {k} sigma")
    else:
        print(f"-> R_T e R0 NON risultano compatibili entro {k} sigma")
    print("=========================================")

    chi2 = chiQuadroLinare(x, y, m, q, dx, dy)
    gdl = x.size - 2
    print(f"chi2: {chi2:.3f}  chi2/gdl: {chi2 / gdl:.3f}")

    r, sr = residui(x, y, dx, dy, m, q)

    # Curva di fit
    x_fit = np.linspace(0, x.max() * 1.05, 200)
    y_fit = retta((m, q), x_fit)

    stile = dict(capsize=2, elinewidth=0.8, markersize=5, alpha=0.85)

    # ===== Finestra 1: dati e retta di fit =====
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        x, y, xerr=dx, yerr=dy,
        fmt="o", color="blue",
        label="Dati sperimentali $(I_R, V_R)$", **stile
    )
    ax.plot(
        x_fit, y_fit, "-", color="red", linewidth=1.5,
        label=f"Fit: $\mathcal{{E}}_T$ = {E_T:.3f} ± {dE_T:.3f} V\n$R_T$ = {R_T:.3f} ± {dR_T:.3f} {unita_R}"
    )

    ax.set_xlabel(nomi[0])
    ax.set_ylabel(nomi[1])
    ax.set_title("Curva caratteristica - Partitore resistivo (Thévenin)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    # ===== Finestra 2: residui normalizzati + istogramma a lato =====
    zn = r / sr
    edges = np.linspace(-3, 3, 7)
    y_g = np.linspace(-3.5, 3.5, 400)

    fig_r, (ax_s, ax_h) = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=True, width_ratios=[4, 1], layout="constrained"
    )

    ax_s.axhspan(-1, 1, color="gray", alpha=0.15, lw=0)
    ax_s.axhspan(-2, 2, color="gray", alpha=0.08, lw=0)
    ax_s.axhline(0, color="black", linewidth=0.8)
    ax_s.plot(
        x, zn, "o", color="blue",
        label=f"Residui ($\chi^2$/gdl = {chi2 / gdl:.2f})"
    )
    ax_s.set_ylabel(r"$r_i / \sigma_{r_i}$")
    ax_s.set_xlabel(nomi[0])
    ax_s.set_ylim(-3.5, 3.5)
    ax_s.set_title("Residui normalizzati - Partitore resistivo")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper left")

    ax_h.hist(
        zn, bins=edges, orientation="horizontal",
        color="blue", alpha=0.5, edgecolor="black"
    )
    ax_h.plot(len(zn) * (edges[1] - edges[0]) * norm.pdf(y_g), y_g, "k-", linewidth=1)
    ax_h.set_xlabel("conteggi")
    ax_h.grid(True, alpha=0.3)

    plt.show()
