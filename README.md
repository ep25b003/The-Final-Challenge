## System Architecture
├── main.py                        ← orchestrator
├── phase1/
│   ├── route_fetcher.py           ← GPS + elevation pipeline 
│   └── solar_model.py             ← Gaussian irradiance model
├── phase2/
│   ├── physics.py                 ← car power equations + constants
│   ├── base_route_optimizer.py    ← SLSQP speed optimiser for base route
│   └── loop_optimizer.py          ← integer loop count + speed optimiser
├── phase3/
│   └── visualizer.py              ← all plots
├── data/
│   └── route_data.csv             ← 281-point route profile
└── outputs/
    ├── 01_velocity_profile.png
    ├── 02_soc_profile.png
    ├── 03_acceleration_profile.png
    ├── 04_solar_profile.png
    └── 05_elevation_profile.png
## Mathematical Assumptions
### Car Physics
 Parameter             Value                  Derivation  

| k (aero drag) | 2.575 × 10⁻³ W/(km/h)³ | k = ½ρCdA = ½×1.2×0.13×1.54 |
| ρ (air density) | 1.2 kg/m³            | Standard, 25°C, sea level |
| Cd            | 0.13                   | Typical solar car teardrop |
| A(frontal area)| 1.54 m²               | Typical solar car |
| P_losses      | 200 W                  | Rolling resistance at 75 km/h (153 W) + electronics (47 W) |
| η_motor       | 0.9                    | Battery → wheel efficiency |
| η_regen       | 0.70                   | Wheel → battery (regen braking) |
| Mass          | 250 kg                 | Car + driver |
| Battery       | 5 kWh                  | Standard|
| SOC_init      | 80%                    | 4000 Wh at race start |
| SOC_min       | 20%                    | 1000 Wh hard floor (Day 3 survival) |
Power equation:
P_mech(v, θ) = k·v³ + P_losses + m·g·sin(θ)·v
P_battery = P_mech / η_motor    (motoring, P_mech > 0)
P_battery = P_mech × η_regen    (regen,    P_mech < 0)
P_net = P_battery - P_solar(t)  (net battery drain)
### Solar Model
I(t) = 1073 × exp(-0.5 × ((t - 43200) / 11600)²)   W/m²
P_solar(t) = I(t) × η_panel × A_panel
           = I(t) × 0.24 × 6.0
| Parameter | Value | Source |
| I_peak | 1073 W/m² | Given |
| t_noon | 43200 s (12:00 PM) | Given |
| σ | 11600 s | Given |
| η_panel | 24% 
| A_panel | 6.0 m²
| Peak P_solar | 1545.12 W| At solar noon |
| Total energy 8AM–5PM | 6493 Wh | Integrated Gaussian |
### Spatial Resolution
Chosen: 100 m
Justification for 0.1 km resolution:
- A resolution of 1 km is too sparse and would smooth out steep hills that could overdraw the battery.
- A resolution of 10 meters generates too many nodes (over 3000 points for a 300 km route), which exponentially increases the computational load for the optimizer
- Sub-100 m resolution adds GPS noise, not strategic insight so we smooth the data using lowpass filter
- Optimiser convergence time stays under 2 minutes



## Optimisation Strategy
### Two-Model Approach
**Model 1 — Base Route (Sasolburg → Zeerust, 280 km)**
- Method: SLSQP (Sequential Least Squares Programming) via scipy
- Variables: 10 macro-block speeds (each ≈ 28 km block)
- Objective: minimise travel time (arrive early → more loop time)
- Constraints: SOC ≥ 20% throughout, legal speed limits [60, 120] km/h
**Model 2 — Loop Phase**
- Method: Brute-force search over integer N, binary search for v_max_energy
- Variables: N (integer loops), v_loop (continuous)
- Objective: maximise N (= maximise total distance)
- Constraints: finish by 5PM, SOC ≥ 20%, legal speed

### Why v_min as target loop speed

Energy consumed per loop:
E_loop(v) = (k·v³ + P_losses)/η_motor - P_solar(t)) × (L/v)
Since solar dominates base losses at low speed, E_loop increases with v
(shown both analytically and numerically). Therefore:
- **Minimum feasible speed = minimum energy consumed = maximum battery reserve**
- Any speed above v_min arrives at the same number of loops but wastes energy
- That wasted energy is unavailable for Day 3 — massive strategic disadvantage
## Results
BASE ROUTE:  8:00 AM → 11:38 AM  (3.64 h)
  Speed profile: ~77 km/h throughout
  Battery on arrival: 1600 Wh (SOC 32%)

CONTROL STOP: 11:38 → 12:08 (30 min, solar charging)

LOOP PHASE: 12:08 PM → 5:00 PM
  N_loops  = 7
  v_loop   = 60.0 km/h (= v_min_time)
  Feasible range: [60.0, 66.0] km/h
  Battery at 5PM: 1559 Wh (SOC 31.2%)
TOTAL DISTANCE: 525 km (280 base + 7 × 35 loops)
### Analytical Insights
1. Solar is the dominant energy source loops: at 60 km/h, P_net ≈ 40 W (nearly solar-neutral). The car is effectively powered by the sun.
2. The energy-time trade-off on base route: driving faster arrives earlier (more loop time) but drains battery faster. The SLSQP finds the Pareto-optimal balance — ~77 km/h is optimal because: below this, we lose loop time; above this, we lose too much energy.

3. Loop speed range is tight [60.0, 66.0 km/h] because the energy budget is limited (only 600 Wh available for loops). This narrow range confirms our Q2 analysis: the energy constraint is binding at the optimum.

4. Day 3 surplus: ending at SOC 31% gives a 1559 Wh buffer — enough for emergency driving even with zero solar the next morning.
APIs used:
- OSRM: https://router.project-osrm.org (road geometry)
- Open-Elevation: https://api.open-elevation.com (altitude)


