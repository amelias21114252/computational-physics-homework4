"""
Computational Physics - Homework 4
Fall 2026

Problem 1:
    Exercise 8.18 (Brusselator) solved with Bulirsch-Stoer.

Problem 2:
    Supermassive black-hole binary solved with a fourth-order
    Runge-Kutta method and adaptive timesteps.

The adaptive RK4 error tolerance DELTA is an error tolerance per unit
time.  It is calibrated in Problem 2(a) using a no-friction orbit with
r_peri = 1e-7 for ten orbital periods.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Global plotting settings
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})

# ============================================================
# Problem 1: Brusselator
# ============================================================

def brusselator(t, r, a=1.0, b=3.0):
    x, y = r
    return np.array([
        1.0 - (b + 1.0)*x + a*x*x*y,
        b*x - a*x*x*y
    ])


def modified_midpoint(f, t, r, H, n):
    h = H/n
    r1 = r.copy()
    r2 = r + h*f(t, r)

    for m in range(1, n):
        r1, r2 = r2, r1 + 2.0*h*f(t + m*h, r2)

    return 0.5*(r1 + r2 + h*f(t + H, r2))


def bs_step(f, t, r, H, delta=1e-10, nmax=8):
    ns = list(range(2, nmax + 1, 2))
    table = []
    target = delta*abs(H)

    for k, n in enumerate(ns):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            estimate = modified_midpoint(f, t, r, H, n)

        if not np.all(np.isfinite(estimate)):
            return r.copy(), False

        row = [estimate]

        for j in range(1, k + 1):
            factor = (ns[k]/ns[k-j])**2 - 1.0
            value = row[j-1] + (row[j-1] - table[k-1][j-1])/factor

            if not np.all(np.isfinite(value)):
                return r.copy(), False

            row.append(value)

        table.append(row)

        if k > 0:
            err = np.max(np.abs(table[k][k] - table[k-1][k-1]))
            if np.isfinite(err) and err <= target:
                return table[k][k], True

    return table[-1][-1], False


def bulirsch_stoer(f, t0, t1, r0, delta=1e-10, nmax=8):
    times = [t0]
    states = [np.array(r0, dtype=float)]
    boundaries = [t0]

    def advance(t, r, H):
        rnew, ok = bs_step(f, t, r, H, delta, nmax)

        if ok:
            times.append(t + H)
            states.append(rnew)
            boundaries.append(t + H)
            return rnew

        if abs(H) < 1e-12:
            raise RuntimeError("Bulirsch-Stoer interval became too small.")

        rmid = advance(t, r, H/2.0)
        return advance(t + H/2.0, rmid, H/2.0)

    advance(t0, np.array(r0, dtype=float), t1 - t0)

    order = np.argsort(times)

    return (
        np.asarray(times)[order],
        np.asarray(states)[order],
        np.asarray(boundaries)
    )


# ============================================================
# Problem 2: black-hole binary
# ============================================================

G = 1.0
M = 1.0
MU = G*M/4.0
R_S = 1e-7

# Calibrated in Problem 2(a)
DELTA = 1e-8


def bh_rhs(t, state, A=0.0, B=1.0):
    """Right-hand side for one BH.

    d2 r/dt2 = -(GM/4r^3) r - [A/(v^3+B)] v
    """

    x, y, vx, vy = state

    r = max(np.hypot(x, y), 1e-15)
    v = np.hypot(vx, vy)

    ax = -MU*x/r**3
    ay = -MU*y/r**3

    if A != 0.0:
        denom = v**3 + B
        ax -= A*vx/denom
        ay -= A*vy/denom

    return np.array([vx, vy, ax, ay])


def rk4_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    k2 = f(t + h/2.0, y + h*k1/2.0, *args)
    k3 = f(t + h/2.0, y + h*k2/2.0, *args)
    k4 = f(t + h, y + h*k3, *args)

    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6.0


def adaptive_rk4(
    A,
    B,
    vfrac=0.8,
    delta=DELTA,
    tmax=100.0,
    h0=1e-3,
    initial_state=None,
    stop_radius=R_S,
    save_every=1,
    max_attempts=2_000_000,
    max_seconds=None,
    progress_every=15000
):
    """Adaptive fourth-order RK using step doubling.

    One step of length 2h is compared with two steps of length h.

    For RK4, the position error estimate per unit time is

        epsilon = |r_small - r_big| / (30 h).

    The step is accepted when epsilon <= delta.

    Because epsilon scales approximately as h^4, the timestep
    controller uses the fourth root.
    """

    if initial_state is None:
        vcirc = np.sqrt(MU)
        state = np.array([1.0, 0.0, 0.0, vfrac*vcirc], dtype=float)
    else:
        state = np.array(initial_state, dtype=float)

    t = 0.0
    h = h0

    T = [t]
    Y = [state.copy()]

    accepted = 0
    attempted = 0
    started = time.monotonic()

    while t < tmax:
        radius = np.hypot(state[0], state[1])

        if stop_radius > 0 and radius <= stop_radius:
            break

        if attempted >= max_attempts:
            print(f"  LIMIT: {max_attempts} attempts at t={t:.5g}, r={radius:.4g}; not claiming merger.", flush=True)
            break
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            print(f"  TIME LIMIT: {max_seconds}s at t={t:.5g}, r={radius:.4g}; not claiming merger.", flush=True)
            break
        if attempted and attempted % progress_every == 0:
            print(f"    progress: t={t:.5g}/{tmax:g}, r={radius:.3g}, attempts={attempted}", flush=True)

        # Do not step past the requested final time.
        h = min(h, (tmax - t)/2.0)

        # Anticipate the shorter orbital timescale near pericenter.
        local_dyn = np.sqrt(max(radius, R_S)**3/MU)
        h = min(h, 0.25*local_dyn + 1e-14)

        if h <= 1e-14:
            print("  WARNING: minimum timestep reached.")
            break

        attempted += 1

        # One step of length 2h
        y_big = rk4_step(bh_rhs, t, state, 2.0*h, A, B)

        # Two steps of length h
        y_half = rk4_step(bh_rhs, t, state, h, A, B)
        y_small = rk4_step(bh_rhs, t + h, y_half, h, A, B)

        if not (np.all(np.isfinite(y_big)) and np.all(np.isfinite(y_small))):
            h *= 0.2
            continue

        # Position error only, as required for orbit accuracy.
        position_difference = np.linalg.norm(y_small[:2] - y_big[:2])

        # Error per unit time.
        error_per_time = position_difference/(30.0*h)

        if error_per_time <= delta:
            old_state = state.copy()
            old_t = t
            old_r = radius

            # Richardson correction.
            state = y_small + (y_small - y_big)/15.0
            t += 2.0*h
            accepted += 1

            new_r = np.hypot(state[0], state[1])

            # Interpolate the crossing of r = r_s.
            if stop_radius > 0 and old_r > stop_radius and new_r <= stop_radius:
                denom = old_r - new_r

                if denom != 0:
                    frac = (old_r - stop_radius)/denom
                    frac = np.clip(frac, 0.0, 1.0)
                    state = old_state + frac*(state - old_state)
                    t = old_t + frac*(t - old_t)

                T.append(t)
                Y.append(state.copy())
                break

            if accepted % save_every == 0:
                T.append(t)
                Y.append(state.copy())

            if error_per_time == 0:
                factor = 2.0
            else:
                factor = 0.9*(delta/error_per_time)**0.25
                factor = min(2.5, max(0.5, factor))

            h *= factor

        else:
            factor = 0.9*(delta/error_per_time)**0.25
            h *= max(0.1, min(0.5, factor))

    if T[-1] != t:
        T.append(t)
        Y.append(state.copy())

    return np.asarray(T), np.asarray(Y), accepted, attempted


def physical_time_unit_myr():
    """Convert one code-time unit to Myr.

    Distance unit = 100 pc.
    r_s = 1e-7 code units.

    Thus the physical Schwarzschild radius is fixed.  Using
    r_s = 2GM/c^2 determines the implied physical mass, after
    which t_unit = sqrt(L^3/GM).
    """

    G_SI = 6.67430e-11
    c = 299792458.0
    pc = 3.085677581491367e16

    length_unit = 100.0*pc
    rs_physical = R_S*length_unit

    mass_physical = rs_physical*c*c/(2.0*G_SI)

    seconds = np.sqrt(
        length_unit**3/(G_SI*mass_physical)
    )

    myr = seconds/(1e6*365.25*86400.0)

    return myr, mass_physical


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
    print(f"  saved {filename}")


def reaches_rs(Y):
    return np.hypot(Y[-1, 0], Y[-1, 1]) <= R_S*(1.0 + 1e-9)


def main():

    time_unit, mass_physical = physical_time_unit_myr()

    print("Computational Physics Homework 4")
    print("--------------------------------")
    print(f"Adaptive RK4 delta = {DELTA:.1e}")
    print(f"1 code time unit = {time_unit:.6f} Myr")
    print(f"Implied BH mass = {mass_physical/1.98847e30:.3e} solar masses")


    # ========================================================
    # Problem 1
    # ========================================================

    print("\nProblem 1: Brusselator")

    t, z, boundaries = bulirsch_stoer(
        brusselator,
        0.0,
        20.0,
        [0.0, 0.0],
        delta=1e-10,
        nmax=8
    )

    plt.figure(figsize=(8, 5))

    plt.plot(t, z[:, 0], "o-", ms=3, label=r"$x(t)$")
    plt.plot(t, z[:, 1], "o-", ms=3, label=r"$y(t)$")

    for q in boundaries:
        plt.axvline(q, alpha=0.10, lw=0.6)

    plt.title("Brusselator Solution with Adaptive Bulirsch-Stoer")
    plt.xlabel(r"Time $t$")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid(alpha=0.2)

    save_plot("problem1_brusselator.png")

    print(f"  accepted Bulirsch-Stoer intervals: {len(t)-1}")


    # ========================================================
    # Problem 2(a)
    # ========================================================

    print("\nProblem 2(a): tolerance calibration")

    r_ap = 1.0
    r_peri = R_S

    # Vis-viva relation at apocenter.
    v_ap = np.sqrt(
        2.0*MU*r_peri /
        (r_ap*(r_ap + r_peri))
    )

    semi_major = 0.5*(r_ap + r_peri)

    period = 2.0*np.pi*np.sqrt(
        semi_major**3/MU
    )

    T, Y, nacc, natt = adaptive_rk4(
        A=0.0,
        B=1.0,
        delta=DELTA,
        tmax=10.0*period,
        h0=1e-3,
        initial_state=[1.0, 0.0, 0.0, v_ap],
        stop_radius=0.0,
        save_every=1
    )

    r = np.hypot(Y[:, 0], Y[:, 1])

    energy = (
        0.5*(Y[:, 2]**2 + Y[:, 3]**2)
        - MU/r
    )

    relative_energy_error = (
        (energy - energy[0])/abs(energy[0])
    )

    max_energy_error = np.max(
        np.abs(relative_energy_error)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        T/period,
        relative_energy_error,
        label="Relative energy error"
    )

    plt.title("Energy Conservation Over 10 Orbits")
    plt.xlabel("Orbital periods")
    plt.ylabel(r"$(E-E_0)/|E_0|$")
    plt.legend()
    plt.grid(alpha=0.2)

    save_plot("problem2a_energy_accuracy.png")

    print(f"  target r_peri = {r_peri:.3e}")
    print(f"  minimum sampled radius = {np.min(r):.8e}")
    print(f"  initial apocenter speed = {v_ap:.8e}")
    print(f"  max relative energy error = {max_energy_error:.3e}")
    print(f"  accepted/attempted steps = {nacc}/{natt}")


    # ========================================================
    # Problem 2(b)
    # ========================================================

    print("\nProblem 2(b): A = B = 1")

    T_b, Y_b, nacc, natt = adaptive_rk4(
        A=1.0,
        B=1.0,
        vfrac=0.8,
        delta=DELTA,
        tmax=100.0,
        h0=2e-3,
        stop_radius=R_S,
        save_every=1
    )

    r_b = np.hypot(Y_b[:, 0], Y_b[:, 1])

    plt.figure(figsize=(7, 6))

    plt.plot(
        Y_b[:, 0],
        Y_b[:, 1],
        label=r"$A=B=1$"
    )

    plt.title(r"BH Orbit with Dynamical Friction ($A=B=1$)")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.axis("equal")
    plt.legend()
    plt.grid(alpha=0.2)

    save_plot("problem2b_orbit.png")

    plt.figure(figsize=(8, 5))

    plt.plot(
        T_b,
        np.log10(np.maximum(r_b, R_S)),
        label=r"$\log_{10}(r)$"
    )

    plt.axhline(
        np.log10(R_S),
        ls="--",
        alpha=0.6,
        label=r"$r_s=10^{-7}$"
    )

    plt.title(r"Orbital Radius vs. Time ($A=B=1$)")
    plt.xlabel("Time [code units]")
    plt.ylabel(r"$\log_{10}(r)$")
    plt.legend()
    plt.grid(alpha=0.2)

    save_plot("problem2b_logr.png")

    print(f"  reached r_s: {reaches_rs(Y_b)}")
    print(f"  final time = {T_b[-1]:.8f} code units")
    print(f"  final time = {T_b[-1]*time_unit:.8f} Myr")
    print(f"  accepted/attempted steps = {nacc}/{natt}")


    # ========================================================
    # Problem 2(c)
    # ========================================================

    print("\nProblem 2(c): dependence on B/A")

    # Hold A=1 and vary B across the requested range.  Then B/A=B.
    # Part (d) separately tests whether the individual magnitudes matter.
    A_fixed = 1.0
    B_values = np.array([0.5, 1.0, 2.0, 5.0, 10.0])

    results = []

    for count, B in enumerate(B_values, start=1):
        A = A_fixed

        print(
            f"  {count}/{len(B_values)}: "
            f"A={A:g}, B={B:g}, B/A={B/A:.4g}",
            flush=True
        )

        T_c, Y_c, nacc, natt = adaptive_rk4(
            A=A,
            B=B,
            vfrac=0.8,
            delta=DELTA,
            tmax=350.0,
            h0=3e-3,
            stop_radius=R_S,
            save_every=20,
            max_seconds=None,
            max_attempts=2_000_000
        )

        reached = reaches_rs(Y_c)

        results.append({
            "A": A,
            "B": B,
            "ratio": B/A,
            "time": T_c[-1],
            "reached": reached
        })

        status = "reached r_s" if reached else "not reached (integration stopped; plotted time is a lower bound)"

        print(
            f"       {status}; "
            f"t={T_c[-1]*time_unit:.4f} Myr"
        )


    ratios_reached = [
        q["ratio"] for q in results if q["reached"]
    ]

    times_reached = [
        q["time"]*time_unit for q in results if q["reached"]
    ]

    ratios_limit = [
        q["ratio"] for q in results if not q["reached"]
    ]

    times_limit = [
        q["time"]*time_unit for q in results if not q["reached"]
    ]

    plt.figure(figsize=(8, 5))

    plt.scatter(
        ratios_reached,
        times_reached,
        label=r"Reached $r_s$"
    )

    if len(ratios_limit) > 0:
        plt.scatter(
            ratios_limit,
            times_limit,
            marker="x",
            s=70,
            label=r"Not reached (lower bound)"
        )

    plt.xscale("log")

    plt.title(r"Time to Schwarzschild Radius vs. $B/A$")
    plt.xlabel(r"Ratio $B/A$")
    plt.ylabel(r"Time to $r_s$ [Myr]")
    plt.legend()
    plt.grid(alpha=0.2)

    save_plot("problem2c_ratio.png")


    # ========================================================
    # Problem 2(d): initial velocity
    # ========================================================

    print("\nProblem 2(d): initial velocity dependence")

    velocity_fractions = np.array([
        0.5, 0.65, 0.8, 0.95, 1.0
    ])

    velocity_times = []

    for vf in velocity_fractions:

        T_d, Y_d, nacc, natt = adaptive_rk4(
            A=1.0,
            B=1.0,
            vfrac=vf,
            delta=DELTA,
            tmax=100.0,
            h0=3e-3,
            stop_radius=R_S,
            save_every=2,
            max_seconds=None
        )

        velocity_times.append(
            T_d[-1]*time_unit
        )

        print(
            f"  v0/vcirc={vf:.2f}: "
            f"{T_d[-1]*time_unit:.4f} Myr"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        velocity_fractions,
        velocity_times,
        "o-",
        label=r"$A=B=1$"
    )

    plt.title("Dependence on Initial Velocity")
    plt.xlabel(r"Initial velocity $v_0/v_{\rm circ}$")
    plt.ylabel(r"Time to $r_s$ [Myr]")
    plt.legend()
    plt.grid(alpha=0.2)

    save_plot("problem2d_velocity.png")


    # ========================================================
    # Problem 2(d): same ratio, different A and B
    # ========================================================

    print("\nProblem 2(d): fixed B/A = 1")

    scales = np.array([
        0.5, 1.0, 2.0, 5.0, 10.0
    ])

    scale_times = []

    for q in scales:

        T_s, Y_s, nacc, natt = adaptive_rk4(
            A=q,
            B=q,
            vfrac=0.8,
            delta=DELTA,
            tmax=100.0,
            h0=3e-3,
            stop_radius=R_S,
            save_every=2,
            max_seconds=None
        )

        scale_times.append(
            T_s[-1]*time_unit
        )

        print(
            f"  A=B={q:g}: "
            f"{T_s[-1]*time_unit:.4f} Myr"
        )

    plt.figure(figsize=(8, 5))

    plt.loglog(
        scales,
        scale_times,
        "o-",
        label=r"Fixed $B/A=1$"
    )

    plt.title(r"Dependence on $A,B$ at Fixed $B/A=1$")
    plt.xlabel(r"$A=B$")
    plt.ylabel(r"Time to $r_s$ [Myr]")
    plt.legend()
    plt.grid(alpha=0.2)

    save_plot("problem2d_scale.png")


    # ========================================================
    # Save numerical summary for the report
    # ========================================================

    with open("homework4_results.txt", "w") as f:

        f.write("Computational Physics Homework 4 Results\n")
        f.write("========================================\n\n")

        f.write(f"DELTA = {DELTA:.8e}\n")
        f.write(f"time_unit_Myr = {time_unit:.12e}\n")
        f.write(
            f"BH_mass_solar = "
            f"{mass_physical/1.98847e30:.12e}\n\n"
        )

        f.write("Problem 2a\n")
        f.write(f"period = {period:.12e}\n")
        f.write(f"v_ap = {v_ap:.12e}\n")
        f.write(f"min_radius = {np.min(r):.12e}\n")
        f.write(f"max_relative_energy_error = {max_energy_error:.12e}\n\n")

        f.write("Problem 2b\n")
        f.write(f"time_code = {T_b[-1]:.12e}\n")
        f.write(f"time_Myr = {T_b[-1]*time_unit:.12e}\n")
        f.write(f"reached_rs = {reaches_rs(Y_b)}\n\n")

        f.write("Problem 2c\n")
        for q in results:
            f.write(
                f"A={q['A']:g}, B={q['B']:g}, "
                f"B/A={q['ratio']:.8g}, "
                f"time_Myr={q['time']*time_unit:.8g}, "
                f"reached={q['reached']}\n"
            )

        f.write("\nProblem 2d velocity\n")
        for vf, tm in zip(velocity_fractions, velocity_times):
            f.write(
                f"v0/vcirc={vf:.8g}, time_Myr={tm:.8g}\n"
            )

        f.write("\nProblem 2d fixed ratio\n")
        for q, tm in zip(scales, scale_times):
            f.write(
                f"A=B={q:.8g}, B/A=1, time_Myr={tm:.8g}\n"
            )

    print("\nAll calculations complete.")
    print("Generated:")
    print("  problem1_brusselator.png")
    print("  problem2a_energy_accuracy.png")
    print("  problem2b_orbit.png")
    print("  problem2b_logr.png")
    print("  problem2c_ratio.png")
    print("  problem2d_velocity.png")
    print("  problem2d_scale.png")
    print("  homework4_results.txt")


if __name__ == "__main__":
    main()
