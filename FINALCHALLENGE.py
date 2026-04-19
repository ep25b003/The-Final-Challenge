
# STANDARD LIBRARY IMPORTS

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import CubicSpline
from scipy.signal      import butter, filtfilt
from scipy.optimize    import minimize, Bounds, brentq



#  PHASE 1 — DATA PIPELINE
#  - Route profile (GPS + elevation + slope)
#  - Solar irradiance model



# SECTION 1A: CONSTANTS RELATED TO ROUTE

# GPS coordinates of the two cities
SASOLBURG = (-26.8149, 27.8321)   # (latitude, longitude)
ZEERUST   = (-25.5444, 26.0832)

TOTAL_ROUTE_KM  = 280.0   # road distance Sasolburg → Zeerust
SPACING_KM      = 0.1     # 100 m resolution = 0.1 km


# SECTION 1B: HELPER GEOMETRY FUNCTIONS


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Great-circle distance [km] between two GPS points.
    Formula:
        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        d = 2R · arcsin(√a)    where R = 6371 km
    """
    R    = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_bearing(lat1, lon1, lat2, lon2):
  
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x    = np.sin(dlon) * np.cos(lat2)
    y    = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

# SECTION 1C: ROUTE GENERATION

def generate_route(spacing_km=SPACING_KM):
    

    # ── Step 1: Distance array ────────────────────────────────
    n_points = int(TOTAL_ROUTE_KM / spacing_km) + 1
    dist     = np.linspace(0, TOTAL_ROUTE_KM, n_points)   # km
    print(f"  Step 1: Created {n_points} distance points (0 to {TOTAL_ROUTE_KM} km)")

    # ── Step 2: Base elevation via cubic spline ───────────────
    ctrl_km  = np.array([  0,   40,   80,  120,  160,  200,  240,  280])
    ctrl_elv = np.array([1480, 1510, 1430, 1350, 1380, 1300, 1240, 1200])

    cs       = CubicSpline(ctrl_km, ctrl_elv)
    elv_base = cs(dist)
    print(f"  Step 2: Cubic spline interpolated through {len(ctrl_km)} control points")
    print(f"          Elevation range: {elv_base.min():.0f} m – {elv_base.max():.0f} m")

    # ── Step 3: Add GPS noise ─────────────────────────────────
    rng      = np.random.default_rng(seed=42)   # fixed seed = reproducible
    noise    = rng.normal(loc=0, scale=5.0, size=n_points)   # σ = 5 m
    elv_raw  = elv_base + noise
    print(f"  Step 3: Added GPS noise (σ=5 m). Noise RMS = {np.std(noise):.2f} m")

    # ── Step 4: Butterworth low-pass filter ───────────────────
    fs      = 1.0 / spacing_km       # samples per km = 10
    f_cut   = 1.0 / 0.5              # 2 cycles/km (500 m cutoff)
    W_n     = f_cut / (fs / 2.0)    # normalised: 0.4
    W_n     = np.clip(W_n, 1e-4, 0.999)

    b, a         = butter(N=4, Wn=W_n, btype='low', analog=False)
    elv_filtered = filtfilt(b, a, elv_raw)   # zero-phase forward+backward

    noise_removed = np.std(elv_raw - elv_filtered)
    print(f"  Step 4: Butterworth LPF applied (4th order, cutoff=500m)")
    print(f"          Noise removed: {noise_removed:.2f} m RMS")

    # ── Step 5: Slope from filtered elevation ─────────────────
    # gradient gives dE/dx in units of m/km (since dist is in km)
    # We need dimensionless (m/m), so divide by 1000
    delv_per_m = np.gradient(elv_filtered, dist * 1000.0)   # m/m = dimensionless
    slope_rad  = np.arctan(delv_per_m)                       # radians
    slope_deg  = np.degrees(slope_rad)
    print(f"  Step 5: Slope computed. Range: {slope_deg.min():.2f}° to {slope_deg.max():.2f}°")

    # ── Step 6: Lat/lon and bearing ──────────────────────────
    frac      = dist / TOTAL_ROUTE_KM
    lats      = SASOLBURG[0] + frac * (ZEERUST[0] - SASOLBURG[0])
    lons      = SASOLBURG[1] + frac * (ZEERUST[1] - SASOLBURG[1])
    bearings  = np.zeros(n_points)
    for i in range(n_points - 1):
        bearings[i] = compute_bearing(lats[i], lons[i], lats[i+1], lons[i+1])
    bearings[-1] = bearings[-2]

    df = pd.DataFrame({
        "dist_km"        : np.round(dist, 4),
        "lat"            : np.round(lats, 6),
        "lon"            : np.round(lons, 6),
        "elevation_m"    : np.round(elv_filtered, 2),
        "elevation_raw_m": np.round(elv_raw, 2),
        "slope_rad"      : np.round(slope_rad, 6),
        "slope_deg"      : np.round(slope_deg, 4),
        "bearing_deg"    : np.round(bearings, 2),
        "segment_km"     : spacing_km,
    })

    print(f"  Step 6: DataFrame built — {len(df)} rows × {len(df.columns)} columns")
    print(f"  Route profile complete.\n")
    return df



# SECTION 1D: SOLAR MODEL

# Solar constants (given in problem)
I_PEAK   = 1073.0           # W/m²  peak irradiance at solar noon
T_NOON   = 12 * 3600        # 43200 seconds from midnight = 12:00 PM
SIGMA    = 11600.0          # seconds — std deviation of Gaussian daylight curve

# Panel parameters (from Agnirath orientation session)
ETA_PANEL      = 0.24      # 20% panel efficiency (monocrystalline silicon)
A_PANEL        = 6.0        # m² total active panel area
P_SOLAR_PEAK   = I_PEAK * ETA_PANEL * A_PANEL   # W ≈ 1288 W at noon


def solar_irradiance(t_sec):
    return I_PEAK * np.exp(-0.5 * ((t_sec - T_NOON) / SIGMA) ** 2)


def solar_power(t_sec):
    """
    Electrical power [W] delivered by the solar array.

    P_solar(t) = I(t) × η_panel × A_panel

    Peak at noon: 1073 × 0.24 × 6.0 = 1545.12 W ≈ 1545 W
    
    """
    return solar_irradiance(t_sec) * ETA_PANEL * A_PANEL


def solar_energy_window(t_start, t_end, n=2000):
    """
    Total solar energy [Wh] between t_start and t_end (both in seconds).

    Uses the trapezoidal rule for numerical integration:
        E = ∫ P_solar(t) dt  [W·s] → divide by 3600 → [Wh]

    n=2000 integration steps → error < 0.01 Wh for a 9-hour window.
    """
    t   = np.linspace(t_start, t_end, n)
    P   = solar_power(t)
    return np.trapezoid(P, t) / 3600.0


def avg_solar_power(t_start, t_end):
    """
    Time-averaged solar power [W] over a window.
    = total_energy_Wh / duration_hours
    """
    dur_h = (t_end - t_start) / 3600.0
    if dur_h <= 0:
        return 0.0
    return solar_energy_window(t_start, t_end) / dur_h


#  PHASE 2 — OPTIMISATION ENGINE
#  - Base route speed optimiser
#  - Loop count + speed optimiser

# SECTION 2A: PHYSICAL CONSTANTS


# ── Aerodynamic drag ──────────────────────────────────────────
# P_aero = k × v³
# k derived from: k_SI = ½ρ·Cd·A = ½×1.2×0.13×1.54 = 0.12012 W/(m/s)³
# Convert to km/h: k = k_SI / 3.6³ = 0.12012 / 46.656
RHO    = 1.2    # kg/m³  air density at 25°C sea level
CD     = 0.13   # drag coefficient (teardrop solar car body)
A_CAR  = 1.54   # m²    frontal area
K_SI   = 0.5 * RHO * CD * A_CAR       # W/(m/s)³ = 0.12012
K      = K_SI / (3.6 ** 3)            # W/(km/h)³ = 2.575×10⁻³

# ── Rolling resistance + electronics (P_losses) ───────────────
# At reference speed v_ref = 75 km/h, mass = 250 kg:
#   P_roll = μ_r × m × g × v = 0.003 × 250 × 9.81 × (75/3.6) = 153 W
#   P_elec = 47 W  (BMS, telemetry, cooling fans, data loggers)
#   P_losses = 200 W
# Treated as constant — valid within ±30 km/h of reference speed.
MU_R       = 0.003
MASS       = 250.0    # kg
G          = 9.81     # m/s²
V_REF      = 75.0     # km/h
P_ROLL_REF = MU_R * MASS * G * (V_REF / 3.6)   # ≈ 153 W
P_ELEC     = 47.0                               # W
P_LOSSES   = P_ROLL_REF + P_ELEC               # ≈ 200 W

# ── Drivetrain efficiencies ───────────────────────────────────
ETA_MOTOR = 0.90   # battery → wheel (motoring)   [UPDATED: was 0.95]
ETA_REGEN = 0.70   # wheel → battery (regen)

# ── Battery ───────────────────────────────────────────────────
BAT_TOTAL_WH = 5000.0   # Wh total capacity
SOC_INIT     = 0.80     # 80% at race start = 4000 Wh
SOC_MIN      = 0.20     # 20% hard floor (Day 3 survival)
E_INIT       = SOC_INIT * BAT_TOTAL_WH   # 4000 Wh
E_MIN        = SOC_MIN  * BAT_TOTAL_WH   # 1000 Wh

# ── Race schedule (all in seconds from midnight) ──────────────
T_START   = 8  * 3600   # 08:00 AM race start
T_END     = 17 * 3600   # 05:00 PM hard cutoff
T_CTRL    = 30 * 60     # 30 min mandatory control stop at Zeerust
T_INT     = 5  * 60     # 5 min between loops
L_LOOP    = 35.0        # km per loop (straight-line physics)

# ── Legal speed limits (South Africa, open road) ─────────────
V_MIN  = 60.0    # km/h minimum
V_MAX  = 120.0   # km/h maximum

# ── Optimiser: number of macro speed blocks ───────────────────
N_BLOCKS = 10

# SECTION 2B: POWER EQUATIONS

def p_mech(v_kmh, slope_rad=0.0):
    """
    Mechanical power at the wheels [W].

    FULL EQUATION:
        P_mech = k·v³  +  P_losses  +  m·g·sin(θ)·v
    """
    v_ms = v_kmh / 3.6
    return K * v_kmh**3 + P_LOSSES + MASS * G * np.sin(slope_rad) * v_ms


def p_battery_draw(v_kmh, slope_rad=0.0):
    """
    Two cases:
        Motoring (P_mech ≥ 0):
            P_draw = P_mech / η_motor        
        Regen braking (P_mech < 0):
            P_draw = P_mech × η_regen   (negative = charging)
    """
    pm = p_mech(v_kmh, slope_rad)
    if np.isscalar(pm):
        return pm / ETA_MOTOR if pm >= 0 else pm * ETA_REGEN
    return np.where(pm >= 0, pm / ETA_MOTOR, pm * ETA_REGEN)


def energy_segment(v_kmh, slope_rad, dist_km, t_start_sec):
    """
    Net battery energy consumed [Wh] for one route segment.

    FORMULA:
        E_segment = (P_battery - P_solar_avg) × duration_hours
    """
    t_end     = t_start_sec + (dist_km / v_kmh) * 3600.0
    p_sol_avg = avg_solar_power(t_start_sec, t_end)
    p_bat     = p_battery_draw(v_kmh, slope_rad)
    p_net     = p_bat - p_sol_avg
    return p_net * (dist_km / v_kmh)   # Wh

# SECTION 2C: BASE ROUTE OPTIMISER

def assign_blocks(route_df):
    n = len(route_df)
    return np.floor(np.linspace(0, N_BLOCKS - 1e-9, n)).astype(int)


def simulate_base_route(v_blocks, route_df, block_ids):
    E         = E_INIT
    t_elapsed = 0.0
    t_clock   = float(T_START)
    soc_list  = [E / BAT_TOTAL_WH]

    segs_d = route_df["segment_km"].values
    segs_s = route_df["slope_rad"].values

    for i in range(len(route_df)):
        v    = float(v_blocks[block_ids[i]])
        d    = segs_d[i]
        slp  = segs_s[i]

        dE        = energy_segment(v, slp, d, t_clock)
        dt_h      = d / v

        E         -= dE
        t_elapsed += dt_h
        t_clock   += dt_h * 3600.0
        soc_list.append(E / BAT_TOTAL_WH)

    return {
        "t_arrival_h" : t_elapsed,
        "E_at_arrival": E,
        "soc_profile" : np.array(soc_list),
    }


def optimise_base_route(route_df):
    """
    Find the 10 block speeds that minimise travel time while
    keeping SOC ≥ 20 % throughout.

    OPTIMISATION METHOD: SLSQP
    (Sequential Least Squares Programming)
  
    OBJECTIVE (what we minimise):
        f(v) = total travel time = Σ (dist_k / v_block_k)
        Minimising travel time = arriving at Zeerust earlier
        = more time available for loops = more distance covered

    CONSTRAINTS (what must always be true):
        1. soc_profile[i] ≥ 0.20  for all i  (SOC never below 20%)
           → returned as array of values that must all be ≥ 0
        2. E_at_arrival ≥ E_MIN + 600 Wh
           → ensures at least some energy for loops after arriving

    BOUNDS:
        60 km/h ≤ v_k ≤ 120 km/h  for each block k

    INITIAL GUESS:
        v0 = 90 km/h for all blocks (midpoint of legal range)
        A good starting point helps SLSQP converge faster.

    Returns dict with optimal speeds and simulation results.
    """
    print(f"  Variables : {N_BLOCKS} speed blocks")
    print(f"  Bounds    : [{V_MIN}, {V_MAX}] km/h")

    block_ids = assign_blocks(route_df)

    def objective(v_blocks):
        res = simulate_base_route(v_blocks, route_df, block_ids)
        return res["t_arrival_h"]

    def constraint_soc(v_blocks):
        # Must all be ≥ 0 for constraint to be satisfied
        res = simulate_base_route(v_blocks, route_df, block_ids)
        return res["soc_profile"] - SOC_MIN

    def constraint_reserve(v_blocks):
        res = simulate_base_route(v_blocks, route_df, block_ids)
        return res["E_at_arrival"] - E_MIN - 600.0   # keep 600 Wh for loops

    constraints = [
        {"type": "ineq", "fun": constraint_soc},
        {"type": "ineq", "fun": constraint_reserve},
    ]
    bounds = Bounds([V_MIN]*N_BLOCKS, [V_MAX]*N_BLOCKS)
    v0     = np.full(N_BLOCKS, 90.0)

    result = minimize(
        objective,
        v0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"maxiter": 500, "ftol": 1e-6, "disp": True},
    )

    v_opt = result.x
    sim   = simulate_base_route(v_opt, route_df, block_ids)

    # Build per-segment speed array (one speed value per 100 m segment)
    v_per_seg = np.array([float(v_opt[block_ids[i]])
                           for i in range(len(route_df))])

    t_arr_sec = T_START + sim["t_arrival_h"] * 3600
    hh = int(t_arr_sec // 3600)
    mm = int((t_arr_sec % 3600) // 60)

    print(f"\n  [RESULT]")
    print(f"  Block speeds  : {np.round(v_opt, 1)} km/h")
    print(f"  Arrival time  : {hh:02d}:{mm:02d}")
    print(f"  Battery arr.  : {sim['E_at_arrival']:.0f} Wh  "
          f"(SOC {sim['E_at_arrival']/BAT_TOTAL_WH*100:.1f}%)")
    return {
        "v_per_seg"    : v_per_seg,
        "block_speeds" : v_opt,
        "t_arrival_h"  : sim["t_arrival_h"],
        "t_arrival_sec": t_arr_sec,
        "E_at_arrival" : sim["E_at_arrival"],
        "soc_profile"  : sim["soc_profile"],
        "block_ids"    : block_ids,
    }

# SECTION 2D: LOOP OPTIMISER

def energy_one_loop(v_kmh, t_start_sec):
    """
    Net battery energy [Wh] for one 35 km loop at constant speed.

    Loops are FLAT (straight-line physics) → slope = 0.
    Solar still charges during the loop.

    P_net = P_battery(v, 0) - P_solar_avg(t_start, t_end)
    E     = P_net × duration_hours
    """
    dur_h    = L_LOOP / v_kmh
    t_end    = t_start_sec + dur_h * 3600.0
    p_sol    = avg_solar_power(t_start_sec, t_end)
    p_bat    = p_battery_draw(v_kmh, slope_rad=0.0)
    return (p_bat - p_sol) * dur_h


def energy_N_loops(v_kmh, N, t_start_sec):
    """
    Total net battery energy [Wh] for N consecutive loops.
    Between loops: 5-minute mandatory stop.
    During the stop the car is stationary:
        P_battery = 0 (not driving)
        P_solar   > 0 (panels still generating)
    → Battery GAINS energy during stops.
    This is accounted for by subtracting the solar harvest during stops.

    Parameters
    v_kmh      : loop speed [km/h]
    N          : number of loops
    t_start_sec: clock time at start of first loop [s from midnight]
    """
    E_total = 0.0
    t_cur   = t_start_sec

    for i in range(N):
        # Energy cost for this loop
        E_total += energy_one_loop(v_kmh, t_cur)
        t_cur   += (L_LOOP / v_kmh) * 3600.0

        # 5-minute charging stop between loops (not after the last one)
        if i < N - 1:
            t_stop_end = t_cur + T_INT
            p_sol_stop = avg_solar_power(t_cur, t_stop_end)
        # Solar charges battery during stop → subtract from total cost
            E_total   -= p_sol_stop * (T_INT / 3600.0)
            t_cur      = t_stop_end

    return E_total


def v_min_time(N, T_remaining_sec):
    """
    Minimum speed [km/h] to finish N loops before the deadline.
    If the required speed exceeds V_MAX, these N loops are infeasible.
    """
    n_stops       = max(0, N - 1)
    T_driving_sec = T_remaining_sec - n_stops * T_INT
    if T_driving_sec <= 0:
        return float("inf")
    return N * L_LOOP / (T_driving_sec / 3600.0)


def optimise_loops(E_at_zeerust, t_arrival_sec):
    """
    Find the maximum number of complete loops N and the target speed.
    """

    t_loops_start = t_arrival_sec + T_CTRL
    T_remaining   = T_END - t_loops_start
    E_available   = E_at_zeerust - E_MIN   # usable energy for loops

    hh = int(t_loops_start // 3600)
    mm = int((t_loops_start % 3600) // 60)
    print(f"  Loops start at      : {hh:02d}:{mm:02d}")
    print(f"  Time for loops      : {T_remaining/3600:.3f} h = {T_remaining/60:.1f} min")
    print(f"  Energy for loops    : {E_available:.0f} Wh")

    N_max_possible = int(T_remaining / ((L_LOOP / V_MIN) * 3600))
    N_max_possible = min(N_max_possible, 25)   # safety cap

    print(f"\n  {'N':>3}  {'v_min(km/h)':>12}  {'v_max(km/h)':>12}  "
          f"{'E_used(Wh)':>11}  {'Result':>8}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*11}  {'─'*8}")

    best = {"N": 0, "v": V_MIN, "v_max": V_MIN, "E_rem": E_at_zeerust}

    for N in range(N_max_possible, 0, -1):
        vmt = v_min_time(N, T_remaining)

        if vmt > V_MAX:
            print(f"  {N:>3}  {vmt:>12.2f}  {'—':>12}  {'—':>11}  ✗ speed>120")
            continue

        vmt = max(vmt, V_MIN)

        # Energy slack: positive = feasible, negative = too expensive
        def slack(v):
            return E_available - energy_N_loops(v, N, t_loops_start)

        s_at_vmin = slack(vmt)
        s_at_vmax = slack(V_MAX)

        E_at_vmin = energy_N_loops(vmt, N, t_loops_start)

        if s_at_vmin < 0:
            print(f"  {N:>3}  {vmt:>12.2f}  {'—':>12}  "
                  f"{E_at_vmin:>11.0f}  ✗ energy")
            continue

        # Find v_max_energy (largest v still within energy budget)
        if s_at_vmax >= 0:
            v_max_e = V_MAX
        else:
            try:
                v_max_e = brentq(slack, vmt, V_MAX, xtol=0.1)
            except ValueError:
                v_max_e = vmt

        E_remaining = E_at_zeerust - E_at_vmin

        print(f"  {N:>3}  {vmt:>12.2f}  {v_max_e:>12.2f}  "
              f"{E_at_vmin:>11.0f}  ✓ FEASIBLE")

        best = {"N": N, "v": vmt, "v_max": v_max_e,
                "E_rem": E_remaining}
        break

    N_opt = best["N"]
    v_opt = best["v"]

    print(f"\n  [RESULT]")
    print(f"  N_loops       = {N_opt}")
    print(f"  Target speed  = {v_opt:.2f} km/h  (= v_min_time)")
    print(f"  Speed range   = [{v_opt:.2f}, {best['v_max']:.2f}] km/h")
    print(f"  Battery at 5PM= {best['E_rem']:.0f} Wh  "
          f"(SOC {best['E_rem']/BAT_TOTAL_WH*100:.1f}%)")
    print(f"  Total dist    = {280 + N_opt*35:.0f} km")

    return {
        "N_loops"       : N_opt,
        "v_loop"        : v_opt,
        "v_max"         : best["v_max"],
        "E_remaining"   : best["E_rem"],
        "t_loops_start" : t_loops_start,
        "total_dist"    : 280.0 + N_opt * L_LOOP,
    }
#  PHASE 3 — VISUALISATION

def build_timeline(base_res, loop_res, route_df):
    """
    Build arrays of (time, velocity, SOC, distance) every 30 seconds.
        Fine enough to show stops and speed changes clearly.
        For a 9-hour race: 9×120 = 1080 steps → fast to compute.
    For each time step we compute:
        P_net = P_bat(v, slope) - P_solar(t)
        ΔE    = P_net × Δt_hours
        E    -= ΔE
        SOC   = E / 5000
    """
    print("\n[PHASE 3 — VIZ] Building full day timeline (30 s steps)...")

    DT     = 30.0   # seconds per step
    t      = float(T_START)
    E      = float(E_INIT)
    dist   = 0.0

    times, vels, socs, dists = [], [], [], []

    # ── Phase A: Base route ───────────────────────────────────
    v_arr  = base_res["v_per_seg"]
    seg_d  = route_df["segment_km"].values
    seg_s  = route_df["slope_rad"].values

    for i in range(len(route_df)):
        v         = float(v_arr[i])
        seg_sec   = (seg_d[i] / v) * 3600.0
        n_steps   = max(1, int(seg_sec / DT))
        dt_act    = seg_sec / n_steps

        for _ in range(n_steps):
            p_bat  = p_battery_draw(v, seg_s[i])
            p_sol  = solar_power(t)
            E     -= (p_bat - p_sol) * (dt_act / 3600.0)
            dist  += v * (dt_act / 3600.0)
            times.append(t); vels.append(v)
            socs.append(np.clip(E / BAT_TOTAL_WH, 0, 1))
            dists.append(dist)
            t += dt_act

    # ── Phase B: Control stop ─────────────────────────────────
    t_ctrl_end = t + T_CTRL
    while t < t_ctrl_end:
        p_sol  = solar_power(t)
        E     += p_sol * (DT / 3600.0)   # charging while stopped
        E      = min(E, BAT_TOTAL_WH)
        times.append(t); vels.append(0.0)
        socs.append(np.clip(E / BAT_TOTAL_WH, 0, 1))
        dists.append(dist)
        t += DT
    t = t_ctrl_end

    # ── Phase C: Loops ────────────────────────────────────────
    N      = loop_res["N_loops"]
    v_loop = loop_res["v_loop"]

    for loop_i in range(N):
        loop_sec  = (L_LOOP / v_loop) * 3600.0
        n_steps   = max(1, int(loop_sec / DT))
        dt_act    = loop_sec / n_steps

        for _ in range(n_steps):
            if t >= T_END:
                break
            p_bat  = p_battery_draw(v_loop, 0.0)
            p_sol  = solar_power(t)
            E     -= (p_bat - p_sol) * (dt_act / 3600.0)
            dist  += v_loop * (dt_act / 3600.0)
            times.append(t); vels.append(v_loop)
            socs.append(np.clip(E / BAT_TOTAL_WH, 0, 1))
            dists.append(dist)
            t += dt_act

        # 5-min stop between loops
        if loop_i < N - 1 and t < T_END:
            t_int_end = min(t + T_INT, T_END)
            while t < t_int_end:
                p_sol  = solar_power(t)
                E     += p_sol * (DT / 3600.0)
                E      = min(E, BAT_TOTAL_WH)
                times.append(t); vels.append(0.0)
                socs.append(np.clip(E / BAT_TOTAL_WH, 0, 1))
                dists.append(dist)
                t += DT
            t = t_int_end

    times = np.array(times)
    hours = (times - T_START) / 3600.0   # hours since race start (0=8AM)
    vels  = np.array(vels)
    socs  = np.array(socs)
    dists = np.array(dists)
    accs  = np.gradient(vels, hours + 1e-9)   # km/h per hour
    accs  = accs * (1000.0 / 3600.0**2)       # → m/s²

    print(f"  Timeline built: {len(times)} time steps covering "
          f"{hours[-1]:.2f} hours")
    return {"hours": hours, "vels": vels, "socs": socs,
            "dists": dists, "accs": accs, "times": times}


def clock_fmt(x, pos):
    """Format hours-since-8AM as HH:MM for x-axis labels."""
    total_min = int(8*60 + x*60)
    hh = total_min // 60
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"


def plot_all(base_res, loop_res, route_df, tl):
    """
    Plot 1 — Velocity Profile:

    Plot 2 — SOC Profile:

    Plot 3 — Acceleration Profile:
        
    Plot 4 — Solar Power:

    Plot 5 — Elevation Profile:
    """

    # ── Plot style ─────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "#0d0d0d", "axes.facecolor": "#1a1a1a",
        "axes.edgecolor": "#444", "axes.labelcolor": "white",
        "axes.titlecolor": "white", "xtick.color": "white",
        "ytick.color": "white", "grid.color": "#333",
        "grid.linestyle": "--", "text.color": "white",
        "legend.facecolor": "#1a1a1a", "legend.edgecolor": "#555",
    })

    OG = "#FF6B00"; CY = "#00D4FF"; GR = "#00FF88"
    RD = "#FF4444"; YL = "#FFD700"

    h   = tl["hours"]
    fmt = mticker.FuncFormatter(clock_fmt)

    t_arr_h   = base_res["t_arrival_h"]
    t_loops_h = (loop_res["t_loops_start"] - T_START) / 3600.0

    # ── PLOT 1: Velocity ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(h, tl["vels"], color=OG, lw=1.5, label="Speed (km/h)")
    ax.fill_between(h, tl["vels"], alpha=0.15, color=OG)
    ax.axvline(t_arr_h,   color=CY, lw=1.5, ls="--",
               label=f"Arrive Zeerust ({int(8+t_arr_h):02d}:{int((t_arr_h%1)*60):02d})")
    ax.axvline(t_loops_h, color=GR, lw=1.5, ls="--", label="Loops begin")
    ax.set_xlabel("Time of Day"); ax.set_ylabel("Speed [km/h]")
    ax.set_title("Velocity Profile – Full Race Day\n"
                 f"Base route ~{base_res['block_speeds'].mean():.1f} km/h → "
                 f"{loop_res['N_loops']} loops at {loop_res['v_loop']:.1f} km/h",
                 fontsize=13, weight="bold")
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlim(0, h[-1]); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("01_velocity_profile.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: 01_velocity_profile.png")

    # ── PLOT 2: SOC ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    soc_pct = tl["socs"] * 100
    ax.plot(h, soc_pct, color=CY, lw=2.0, label="Battery SOC (%)")
    ax.fill_between(h, soc_pct, 20, where=(soc_pct >= 20),
                    alpha=0.12, color=CY)
    ax.axhline(20, color=RD, lw=2.0, ls="--", label="SOC minimum = 20%")
    ax.axhline(80, color=GR, lw=1.0, ls=":",  label="Initial SOC = 80%")
    ax.axvline(t_arr_h,   color="#888", lw=1.0, ls="--")
    ax.axvline(t_loops_h, color="#888", lw=1.0, ls="--")
    final_soc = soc_pct[-1]
    ax.annotate(f"End SOC: {final_soc:.1f}%",
                xy=(h[-1], final_soc), xytext=(h[-1]-1.5, final_soc+5),
                color=YL, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=YL))
    ax.set_xlabel("Time of Day"); ax.set_ylabel("State of Charge [%]")
    ax.set_title("Battery SOC Profile – Full Race Day", fontsize=13, weight="bold")
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlim(0, h[-1]); ax.set_ylim(0, 100)
    ax.legend(loc="lower left"); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("02_soc_profile.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: 02_soc_profile.png")

    # ── PLOT 3: Acceleration ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    acc_clip = np.clip(tl["accs"], -3, 3)
    ax.plot(h, acc_clip, color=YL, lw=1.0, alpha=0.8)
    ax.fill_between(h, acc_clip, 0, where=(acc_clip>0),
                    alpha=0.2, color=GR, label="Accelerating")
    ax.fill_between(h, acc_clip, 0, where=(acc_clip<0),
                    alpha=0.2, color=RD, label="Decelerating")
    ax.axhline(0, color="#555", lw=0.8, ls=":")
    ax.set_xlabel("Time of Day"); ax.set_ylabel("Acceleration [m/s²]")
    ax.set_title("Acceleration Profile  (clipped to ±3 m/s²)",
                 fontsize=13, weight="bold")
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlim(0, h[-1]); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("03_acceleration_profile.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: 03_acceleration_profile.png")

    # ── PLOT 4: Solar Power ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    P_sol = solar_power(tl["times"])
    ax.plot(h, P_sol, color=YL, lw=2.5, label="P_solar [W]")
    ax.fill_between(h, P_sol, alpha=0.2, color=YL)
    ax.axhline(P_SOLAR_PEAK, color=OG, lw=1.0, ls=":",
               label=f"Peak = {P_SOLAR_PEAK:.0f} W  (12:00 PM)")
    ax.axvspan(t_loops_h, h[-1], alpha=0.07, color=GR,
               label="Loop phase (high solar)")
    ax.set_xlabel("Time of Day"); ax.set_ylabel("Solar Power [W]")
    ax.set_title(f"Solar Array Output — η={ETA_PANEL*100:.0f}%,  "
                 f"A={A_PANEL} m²,  Peak={P_SOLAR_PEAK:.0f} W",
                 fontsize=13, weight="bold")
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlim(0, h[-1]); ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("04_solar_profile.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: 04_solar_profile.png")

    # ── PLOT 5: Elevation + Slope ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    d    = route_df["dist_km"].values
    e    = route_df["elevation_m"].values
    e_raw= route_df["elevation_raw_m"].values
    s    = route_df["slope_deg"].values

    ax1.plot(d, e_raw, color="#555555", lw=0.6, alpha=0.7, label="Raw (GPS noise)")
    ax1.plot(d, e,     color=GR,        lw=1.8, label="Filtered (Butterworth LPF)")
    ax1.fill_between(d, e.min()-10, e, alpha=0.1, color=GR)
    ax1.set_ylabel("Elevation [m]")
    ax1.set_title("Base Route: Sasolburg → Zeerust\n"
                  "100 m resolution · 4th-order Butterworth LPF (500 m cutoff)",
                  fontsize=12, weight="bold")
    ax1.legend(loc="upper right"); ax1.grid(True, alpha=0.4)

    ax2.plot(d, s, color=OG, lw=0.8, alpha=0.9)
    ax2.axhline(0, color="#555", lw=0.8, ls=":")
    ax2.fill_between(d, s, 0, where=(np.array(s)>0), alpha=0.35,
                     color=RD, label="Uphill (costs energy)")
    ax2.fill_between(d, s, 0, where=(np.array(s)<0), alpha=0.35,
                     color=GR, label="Downhill (regen possible)")
    ax2.set_xlabel("Distance from Sasolburg [km]")
    ax2.set_ylabel("Road Slope [°]")
    ax2.legend(); ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("05_elevation_profile.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: 05_elevation_profile.png")


# =============================================================
#  MAIN — Orchestrates everything
# =============================================================

def main():
    print("=" * 62)
    print("   AGNIRATH SOLAR RACE STRATEGY — DAY 2 OPTIMISER")
    print("   Sasolburg → Zeerust | Sasol Solar Challenge Sim")
    print("=" * 62)

    # ── PHASE 1 ───────────────────────────────────────────────
    print("\n" + "─"*62)
    print(" PHASE 1: DATA PIPELINE")
    print("─"*62)

    route_df = generate_route(spacing_km=SPACING_KM)
    route_df.to_csv("route_data.csv", index=False)
    print(f"  Route saved: route_data.csv  ({len(route_df)} rows)")

    print(f"\n  Solar model (Gaussian irradiance):")
    print(f"    I_peak     = {I_PEAK} W/m²  at 12:00 PM")
    print(f"    σ          = {SIGMA/3600:.2f} hours  (daylight std dev)")
    print(f"    η_panel    = {ETA_PANEL*100:.0f}%")
    print(f"    A_panel    = {A_PANEL} m²")
    print(f"    Peak P_sol = {P_SOLAR_PEAK:.0f} W")
    print(f"    8AM–5PM E  = {solar_energy_window(T_START, T_END):.0f} Wh")

    # ── PHASE 2 ───────────────────────────────────────────────
    print("\n" + "─"*62)
    print(" PHASE 2: OPTIMISATION ENGINE")
    print("─"*62)

    base_res = optimise_base_route(route_df)
    loop_res = optimise_loops(
        E_at_zeerust  = base_res["E_at_arrival"],
        t_arrival_sec = base_res["t_arrival_sec"],
    )

    # ── PHASE 3 ───────────────────────────────────────────────
    print("\n" + "─"*62)
    print(" PHASE 3: VISUALISATION")
    print("─"*62)

    tl = build_timeline(base_res, loop_res, route_df)
    plot_all(base_res, loop_res, route_df, tl)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    t_arr = base_res["t_arrival_sec"]
    hh_a  = int(t_arr // 3600); mm_a = int((t_arr % 3600) // 60)
    t_lp  = loop_res["t_loops_start"]
    hh_l  = int(t_lp // 3600);  mm_l  = int((t_lp % 3600) // 60)

    print("\n" + "=" * 62)
    print("   FINAL RACE STRATEGY SUMMARY")
    print("=" * 62)
    print(f"\n  BASE ROUTE   :  Sasolburg → Zeerust  (280 km)")
    print(f"  Depart       :  08:00 AM")
    print(f"  Speed profile:  {np.round(base_res['block_speeds'],1)} km/h")
    print(f"  Arrive       :  {hh_a:02d}:{mm_a:02d}")
    print(f"  Battery arr. :  {base_res['E_at_arrival']:.0f} Wh  "
          f"(SOC {base_res['E_at_arrival']/BAT_TOTAL_WH*100:.1f}%)")

    print(f"\n  CONTROL STOP :  30 min at Zeerust  (solar charging)")

    print(f"\n  LOOP PHASE   :  starts {hh_l:02d}:{mm_l:02d}")
    print(f"  Loop length  :  {L_LOOP} km  (flat, straight-line)")
    print(f"  N loops      :  {loop_res['N_loops']}")
    print(f"  Loop speed   :  {loop_res['v_loop']:.2f} km/h  (= v_min_time)")
    print(f"  Speed range  :  [{loop_res['v_loop']:.2f}, {loop_res['v_max']:.2f}] km/h")
    print(f"  Battery 5PM  :  {loop_res['E_remaining']:.0f} Wh  "
          f"(SOC {loop_res['E_remaining']/BAT_TOTAL_WH*100:.1f}%)")

    print(f"\n  TOTAL DIST   :  {loop_res['total_dist']:.0f} km")
    print(f"  (280 base + {loop_res['N_loops']} × 35 km loops)")
    print("\n  Output files :")
    for f in ["route_data.csv","01_velocity_profile.png",
              "02_soc_profile.png","03_acceleration_profile.png",
              "04_solar_profile.png","05_elevation_profile.png"]:
        print(f"    {f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
