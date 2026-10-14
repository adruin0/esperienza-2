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


def parametriFitODR(x, y, dx, dy, stima_iniziale=(1.0, 0.0)):
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
    # Lettura dati: x = R3 [Ohm], y = I [uA], dx = dR3 [Ohm], dy = dI [uA]
    x_all, y_all, dx_all, dy_all, nomi = getData(sys.argv[1])
    k = 3

    nomi = [pulisci(n) for n in nomi]
    print(f"Caricati {len(x_all)} punti da {sys.argv[1]}")

    # ==========================================================================
    # PARAMETRI SPERIMENTALI (da adattare ai valori misurati in laboratorio)
    # ==========================================================================
    R1, dR1 = 220.0, 0.6            # R1 [Ohm]
    R2, dR2 = 118.9, 0.4            # R2 [Ohm]
    Rx_mult, dRx_mult = 149.0, 0.5  # Rx = R4 misurata col multimetro [Ohm]

    # Intervallo di R3 [Ohm] attorno a I ~ 0 per il fit lineare
    # Impostare a None, None se il file xlsx contiene già solo la regione lineare
    R3_min_fit, R3_max_fit = None, None
    # ==========================================================================

    mask = np.ones_like(x_all, dtype=bool)
    if R3_min_fit is not None:
        mask &= (x_all >= R3_min_fit)
    if R3_max_fit is not None:
        mask &= (x_all <= R3_max_fit)

    x, y, dx, dy = x_all[mask], y_all[mask], dx_all[mask], dy_all[mask]
    usa_sottoinsieme = len(x) < len(x_all)
    if usa_sottoinsieme:
        print(f"Selezionati {len(x)} punti nella regione di interesse (I ~ 0) per il fit.")

    A0, B0 = np.polyfit(x, y, 1)
    A, B, dA, dB, covAB = parametriFitODR(x, y, dx, dy, stima_iniziale=(A0, B0))

    # Calcolo di R3 a I = 0: R3 = -B / A e propagazione con covarianza
    R3_0 = -B / A
    var_R3 = (1.0 / A**2) * (dB**2) + (B**2 / A**4) * (dA**2) - 2.0 * (B / A**3) * covAB
    dR3_0 = np.sqrt(max(var_R3, 0.0))

    # Stima di Rx = R4 = (R2 * R3) / R1
    Rx = (R2 * R3_0) / R1
    dRx = Rx * np.sqrt((dR1 / R1) ** 2 + (dR2 / R2) ** 2 + (dR3_0 / R3_0) ** 2)

    z = abs(Rx - Rx_mult) / np.sqrt(dRx**2 + dRx_mult**2)

    print("=========================================")
    print("Ponte di Wheatstone:")
    print(f"A  = {A:.4f} ± {dA:.4f} μA/Ω")
    print(f"B  = {B:.2f} ± {dB:.2f} μA")
    print(f"cov(A, B) = {covAB:.4e}")
    print(f"R3 (I=0)  = {R3_0:.2f} ± {dR3_0:.2f} Ω")
    print(f"Rx (fit)  = {Rx:.2f} ± {dRx:.2f} Ω")
    print(f"Rx (mult) = {Rx_mult:.2f} ± {dRx_mult:.2f} Ω")
    print("=========================================")
    print(f"Discrepanza: {z:.2f} sigma")
    if consistenzaStatistica(Rx, Rx_mult, dRx, dRx_mult, k):
        print(f"Le stime di Rx risultano compatibili entro {k} sigma")
    else:
        print(f"Le stime di Rx NON risultano compatibili entro {k} sigma")
    print("=========================================")

    chi2 = chiQuadroLinare(x, y, A, B, dx, dy)
    gdl = x.size - 2
    print(f"chi2: {chi2:.3f}  chi2/gdl: {chi2 / gdl:.3f}")

    r, sr = residui(x, y, dx, dy, A, B)

    # Curva di fit
    margin = 0.05 * (x.max() - x.min()) if x.max() != x.min() else 1.0
    x_fit = np.linspace(x.min() - margin, x.max() + margin, 200)
    y_fit = retta((A, B), x_fit)

    stile = dict(capsize=2, elinewidth=0.8, markersize=5, alpha=0.85)

    # ===== Finestra 1: dati e retta di fit =====
    fig, ax = plt.subplots(figsize=(8, 6))

    if usa_sottoinsieme:
        ax.errorbar(
            x_all, y_all, xerr=dx_all, yerr=dy_all,
            fmt="o", color="gray", alpha=0.4,
            label="Dati completi (esclusi dal fit)",
            capsize=2, elinewidth=0.6, markersize=4
        )

    ax.errorbar(
        x, y, xerr=dx, yerr=dy,
        fmt="o", color="blue",
        label="Dati (regione di fit)", **stile
    )
    ax.plot(
        x_fit, y_fit, "-", color="red", linewidth=1.5,
        label=f"Fit: A = {A:.3f} ± {dA:.3f} μA/Ω, B = {B:.1f} ± {dB:.1f} μA\n$R_3(0)$ = {R3_0:.1f} ± {dR3_0:.1f} $\Omega$"
    )
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")

    ax.set_xlabel(nomi[0])
    ax.set_ylabel(nomi[1])
    ax.set_title("I vs $R_3$ - Ponte di Wheatstone")
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
    ax_s.set_title("Residui normalizzati del fit lineare - Ponte di Wheatstone")
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
