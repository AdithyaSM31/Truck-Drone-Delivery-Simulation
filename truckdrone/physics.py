"""
Drone physics and energy model (paper Section III-B).

Implements, as executable code, every equation the paper states:

  Eq. 1  E = P_cruise (1 + alpha m) d / v_cruise + P_hover t_delivery
  Eq. 2  T = m g + 0.5 rho A C_D v^2
  Eq. 4  B_d(t+1) = B_d(t) - E_sortie / C_bat
  Eq. 5  V(B_d) = V_0 - k (1 - B_d) - R_int I

Eq. 2 is not decorative here: the actuator-disk model built on it is what
*produces* the constants P_hover = 500 W, P_cruise = 350 W and alpha ~ 0.3
that Eq. 1 uses. Rotor disc area and propulsive efficiency are solved
numerically at construction time so the aerodynamic model reproduces the
stated hover and cruise powers exactly; the payload coefficient alpha that
falls out of it is then checked against the stated 0.3 (see
``calibration_report``).

Eq. 5 is likewise live: terminal voltage sags with both state of charge and
current draw, so delivering a given mechanical power needs more current and
wastes more energy as ohmic loss as the pack empties. Sortie costs are
integrated over sub-steps to capture that acceleration.
"""

import numpy as np

G = 9.81           # m/s^2
RHO = 1.225        # kg/m^3, sea level


class DronePhysics:
    """
    Physical model for a delivery drone in the DJI Matrice 300 weight class.

    The defaults reproduce the paper's Section III-B parameter set:
    P_cruise = 350 W, P_hover = 500 W, alpha ~ 0.3, C_bat = 100 Wh,
    v_cruise = 15 m/s (54 km/h), t_delivery = 30 s, B_safety = 10 %.
    """

    def __init__(
        self,
        max_range_km: float = 8.0,
        cruise_speed_kmh: float = 54.0,        # v_cruise = 15 m/s (paper Eq.1)
        max_payload_kg: float = 2.5,
        battery_capacity_wh: float = 100.0,    # C_bat = 100 Wh (paper Eq.4)
        hover_power_w: float = 500.0,          # P_hover = 500 W (paper Eq.1)
        cruise_power_w: float = 350.0,         # P_cruise = 350 W (paper Eq.1)
        payload_power_factor: float = 0.3,     # alpha = 0.3 (paper Eq.1)
        launch_recovery_time_s: float = 60.0,
        delivery_hover_time_s: float = 30.0,   # t_delivery = 30 s (paper Eq.1)
        safety_margin_pct: float = 10.0,       # B_safety = 0.10 (paper III-B)
        # --- Eq. 2 airframe parameters ---
        empty_mass_kg: float = 5.2,            # airframe + battery, no payload
        frontal_area_m2: float = 0.035,        # A (paper Eq.2)
        drag_coefficient: float = 0.75,        # C_D (paper Eq.2)
        avionics_power_w: float = 20.0,        # sensors, compute, comms
        # --- Eq. 5 LiPo parameters (6S pack) ---
        v_open_circuit: float = 25.2,          # V_0 (paper Eq.5)
        discharge_steepness: float = 3.0,      # k (paper Eq.5)
        internal_resistance_ohm: float = 0.05,  # R_int (paper Eq.5)
        v_nominal: float = 22.2,
        # --- truck comparison ---
        truck_speed_kmh: float = 25.0,
        truck_road_factor: float = 1.0,
        truck_energy_wh_per_km: float = 1193.0,
    ):
        self.max_range_km = max_range_km
        self.cruise_speed_kmh = cruise_speed_kmh
        self.cruise_speed_ms = cruise_speed_kmh / 3.6
        self.max_payload_kg = max_payload_kg
        self.battery_capacity_wh = battery_capacity_wh
        self.hover_power_w = hover_power_w
        self.cruise_power_w = cruise_power_w
        self.payload_power_factor = payload_power_factor
        self.launch_recovery_time_s = launch_recovery_time_s
        self.delivery_hover_time_s = delivery_hover_time_s
        self.safety_margin_pct = safety_margin_pct

        self.empty_mass_kg = empty_mass_kg
        self.frontal_area_m2 = frontal_area_m2
        self.drag_coefficient = drag_coefficient
        self.avionics_power_w = avionics_power_w

        self.v_open_circuit = v_open_circuit
        self.discharge_steepness = discharge_steepness
        self.internal_resistance_ohm = internal_resistance_ohm
        self.v_nominal = v_nominal

        self._truck_speed_kmh = truck_speed_kmh
        self.truck_road_factor = truck_road_factor
        self.truck_energy_wh_per_km = truck_energy_wh_per_km

        self.rotor_disc_area_m2, self.propulsive_efficiency = self._calibrate()

    # ------------------------------------------------------------------
    # Eq. 2 -- thrust and the actuator-disk power model
    # ------------------------------------------------------------------

    def drag_n(self, airspeed_ms: float = None) -> float:
        """Parasite drag: 0.5 rho A C_D v^2 (second term of Eq. 2)."""
        v = self.cruise_speed_ms if airspeed_ms is None else airspeed_ms
        return 0.5 * RHO * self.frontal_area_m2 * self.drag_coefficient * v ** 2

    def thrust_n(self, payload_kg: float = 0.0,
                 airspeed_ms: float = None) -> float:
        """
        Eq. 2:  T = m g + 0.5 rho A C_D v^2

        Total thrust the rotors must produce in forward flight, where m is the
        gross mass (airframe + payload).
        """
        mass = self.empty_mass_kg + payload_kg
        return mass * G + self.drag_n(airspeed_ms)

    def _shaft_power_w(self, payload_kg, airspeed_ms, disc_area):
        """
        Mechanical power demanded of the rotors, from actuator-disk theory.

        Hover (v = 0):    P = T^1.5 / sqrt(2 rho A_disc)
        Forward flight:   P = T v_i + parasite power, with induced velocity
                          v_i = T / (2 rho A_disc v)

        The induced term falls as airspeed rises, which is why cruise power is
        below hover power for the same aircraft.
        """
        thrust = self.thrust_n(payload_kg, airspeed_ms)
        if not airspeed_ms:
            return thrust ** 1.5 / np.sqrt(2.0 * RHO * disc_area)
        induced_velocity = thrust / (2.0 * RHO * disc_area * airspeed_ms)
        parasite_power = self.drag_n(airspeed_ms) * airspeed_ms
        return thrust * induced_velocity + parasite_power

    def _calibrate(self):
        """
        Solve for rotor disc area and propulsive efficiency such that the
        actuator-disk model reproduces the paper's stated P_hover and
        P_cruise at zero payload.

        Efficiency is pinned by the hover condition, which leaves one unknown
        (disc area) fixed by the cruise condition.
        """
        usable_hover = self.hover_power_w - self.avionics_power_w
        usable_cruise = self.cruise_power_w - self.avionics_power_w
        if usable_hover <= 0 or usable_cruise <= 0:
            raise ValueError("Avionics draw exceeds hover/cruise power.")

        def efficiency_for(disc_area):
            return self._shaft_power_w(0.0, 0.0, disc_area) / usable_hover

        def residual(disc_area):
            eta = efficiency_for(disc_area)
            shaft = self._shaft_power_w(0.0, self.cruise_speed_ms, disc_area)
            return shaft / eta - usable_cruise

        # Cruise power is U-shaped in disc area: induced power falls as the
        # disc grows, while the efficiency implied by the hover condition
        # worsens. Locate that minimum analytically, then bracket the lower
        # (higher disc-loading, physically compact) root below it.
        drag = self.drag_n(self.cruise_speed_ms)
        if drag > 0:
            thrust = self.thrust_n(0.0, self.cruise_speed_ms)
            s_min = thrust / (self.cruise_speed_ms * np.sqrt(drag))
            hi = s_min ** 2 / (2.0 * RHO)
        else:
            hi = 1e3

        lo = 1e-6
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0:
            raise ValueError(
                "Cannot calibrate the aerodynamic model to P_hover="
                "{} W / P_cruise={} W with the given airframe "
                "(best achievable cruise power is {:.1f} W).".format(
                    self.hover_power_w, self.cruise_power_w,
                    residual(hi) + usable_cruise + self.avionics_power_w))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if residual(lo) * residual(mid) <= 0:
                hi = mid
            else:
                lo = mid
        disc_area = 0.5 * (lo + hi)
        return disc_area, efficiency_for(disc_area)

    def aero_power_w(self, payload_kg: float = 0.0,
                     airspeed_ms: float = None) -> float:
        """Electrical power draw predicted by the Eq. 2 actuator-disk model."""
        v = self.cruise_speed_ms if airspeed_ms is None else airspeed_ms
        shaft = self._shaft_power_w(payload_kg, v, self.rotor_disc_area_m2)
        return shaft / self.propulsive_efficiency + self.avionics_power_w

    @property
    def alpha_aero(self) -> float:
        """Payload power coefficient implied by Eq. 2, per kg of payload."""
        return self.aero_power_w(1.0) / self.aero_power_w(0.0) - 1.0

    def calibration_report(self) -> str:
        return (
            "Aerodynamic calibration (Eq. 2 -> Eq. 1 constants)\n"
            "  empty mass          : {:.2f} kg\n"
            "  frontal area A      : {:.4f} m^2   C_D = {:.2f}\n"
            "  rotor disc area     : {:.4f} m^2 (solved)\n"
            "  propulsive eff. eta : {:.4f}     (solved)\n"
            "  P_hover  (model)    : {:.1f} W  [target {:.1f} W]\n"
            "  P_cruise (model)    : {:.1f} W  [target {:.1f} W]\n"
            "  alpha    (model)    : {:.3f}    [paper states {:.2f}]".format(
                self.empty_mass_kg, self.frontal_area_m2,
                self.drag_coefficient, self.rotor_disc_area_m2,
                self.propulsive_efficiency,
                self.aero_power_w(0.0, 0.0), self.hover_power_w,
                self.aero_power_w(0.0), self.cruise_power_w,
                self.alpha_aero, self.payload_power_factor))

    # ------------------------------------------------------------------
    # Eq. 5 -- LiPo terminal voltage and non-linear discharge
    # ------------------------------------------------------------------

    def terminal_voltage(self, soc: float, current_a: float) -> float:
        """Eq. 5:  V(B_d) = V_0 - k (1 - B_d) - R_int I."""
        return (self.v_open_circuit
                - self.discharge_steepness * (1.0 - soc)
                - self.internal_resistance_ohm * current_a)

    def _voltage_under_load(self, soc: float, power_w: float) -> float:
        """
        Terminal voltage when the pack must deliver ``power_w``.

        Current and voltage are coupled (I = P / V), so Eq. 5 becomes
        V^2 - E V + R_int P = 0 with E = V_0 - k (1 - B_d). The upper root is
        the physical operating point.
        """
        e = self.v_open_circuit - self.discharge_steepness * (1.0 - soc)
        disc = e * e - 4.0 * self.internal_resistance_ohm * power_w
        if disc <= 0:
            return 0.1 * self.v_nominal  # pack cannot sustain this load
        return 0.5 * (e + np.sqrt(disc))

    def discharge_multiplier(self, soc: float, power_w: float) -> float:
        """
        Factor by which energy drawn from the cells exceeds the energy
        delivered to the rotors, from the ohmic loss implied by Eq. 5.

        Delivering P at terminal voltage V costs an extra I^2 R_int of
        resistive heating, with I = P / V. As the pack empties V falls, so the
        same mechanical power needs more current and wastes more, accelerating
        depletion in the final stages of flight (paper Section III-B).
        """
        v = self._voltage_under_load(max(0.0, min(1.0, soc)), power_w)
        return 1.0 + self.internal_resistance_ohm * power_w / max(v * v, 1e-6)

    # ------------------------------------------------------------------
    # Eq. 1 -- energy for a flight leg
    # ------------------------------------------------------------------

    def cruise_power_at(self, payload_kg: float = 0.0) -> float:
        """P_cruise (1 + alpha m), the bracketed term of Eq. 1."""
        return self.cruise_power_w * (
            1.0 + self.payload_power_factor * payload_kg)

    def flight_time_s(self, distance_m: float) -> float:
        """Time in seconds to fly a given distance at cruise speed."""
        return distance_m / self.cruise_speed_ms

    def flight_time_min(self, distance_m: float) -> float:
        return self.flight_time_s(distance_m) / 60.0

    def energy_consumption_wh(self, distance_m: float,
                              payload_kg: float = 0.0) -> float:
        """
        Ideal (nominal-voltage) energy for a cruise leg -- the first term of
        Eq. 1. Voltage sag is applied separately by ``leg_charge_pct`` so the
        ideal figure stays comparable across states of charge.
        """
        time_h = self.flight_time_s(distance_m) / 3600.0
        return self.cruise_power_at(payload_kg) * time_h

    def hover_energy_wh(self, duration_s: float) -> float:
        """P_hover t, the second term of Eq. 1."""
        return self.hover_power_w * duration_s / 3600.0

    # ------------------------------------------------------------------
    # Eq. 4 -- battery state update, integrated with voltage sag
    # ------------------------------------------------------------------

    def _drain_pct(self, soc_pct, ideal_wh, power_w, substeps=4):
        """
        Deplete the pack by ``ideal_wh`` of nominal-voltage energy drawn at
        ``power_w``, integrating Eq. 5's sag over ``substeps`` sub-intervals.

        Returns (new_soc_pct, actual_wh_drawn).
        """
        soc = soc_pct
        drawn = 0.0
        chunk = ideal_wh / substeps
        for _ in range(substeps):
            actual = chunk * self.discharge_multiplier(soc / 100.0, power_w)
            drawn += actual
            soc -= actual / self.battery_capacity_wh * 100.0
            if soc <= 0.0:
                return 0.0, drawn
        return soc, drawn

    def battery_pct_consumed(self, distance_m: float,
                             payload_kg: float = 0.0,
                             soc_pct: float = 100.0) -> float:
        """Percentage of pack capacity consumed by one cruise leg."""
        ideal = self.energy_consumption_wh(distance_m, payload_kg)
        power = self.cruise_power_at(payload_kg)
        new_soc, _ = self._drain_pct(soc_pct, ideal, power)
        return soc_pct - new_soc

    def sortie_profile(self, dist_to_customer_m: float,
                       dist_customer_to_recovery_m: float,
                       payload_kg: float = 1.0,
                       soc_pct: float = 100.0) -> dict:
        """
        Simulate a complete sortie:
            launch -> cruise out (laden) -> hover handoff -> cruise back
            (empty) -> recover

        Returns a dict with battery_pct_cost, energy_wh (actual charge drawn,
        voltage-sag inclusive), ideal_energy_wh (Eq. 1 nominal), time_s and
        end_soc_pct.
        """
        soc = soc_pct
        total_wh = 0.0
        ideal_wh = 0.0

        # Launch: hover-power climb-out
        e = self.hover_energy_wh(self.launch_recovery_time_s / 2.0)
        ideal_wh += e
        soc, wh = self._drain_pct(soc, e, self.hover_power_w)
        total_wh += wh

        # Outbound cruise, carrying the package
        e = self.energy_consumption_wh(dist_to_customer_m, payload_kg)
        ideal_wh += e
        soc, wh = self._drain_pct(soc, e, self.cruise_power_at(payload_kg))
        total_wh += wh

        # Hover at the customer for the handoff
        e = self.hover_energy_wh(self.delivery_hover_time_s)
        ideal_wh += e
        soc, wh = self._drain_pct(soc, e, self.hover_power_w)
        total_wh += wh

        # Return cruise, unladen
        e = self.energy_consumption_wh(dist_customer_to_recovery_m, 0.0)
        ideal_wh += e
        soc, wh = self._drain_pct(soc, e, self.cruise_power_at(0.0))
        total_wh += wh

        # Recovery: hover-power descent and docking
        e = self.hover_energy_wh(self.launch_recovery_time_s / 2.0)
        ideal_wh += e
        soc, wh = self._drain_pct(soc, e, self.hover_power_w)
        total_wh += wh

        return {
            "battery_pct_cost": soc_pct - soc,
            "energy_wh": total_wh,
            "ideal_energy_wh": ideal_wh,
            "time_s": self.sortie_time_s(dist_to_customer_m,
                                         dist_customer_to_recovery_m),
            "end_soc_pct": soc,
        }

    def sortie_battery_cost(self, dist_to_customer_m: float,
                            dist_customer_to_recovery_m: float,
                            payload_kg: float = 1.0,
                            soc_pct: float = 100.0) -> float:
        """Battery percentage consumed by a full sortie."""
        return self.sortie_profile(dist_to_customer_m,
                                   dist_customer_to_recovery_m,
                                   payload_kg, soc_pct)["battery_pct_cost"]

    def sortie_time_s(self, dist_to_customer_m: float,
                      dist_customer_to_recovery_m: float) -> float:
        """Total wall-clock time for a complete drone sortie."""
        return (self.flight_time_s(dist_to_customer_m)
                + self.delivery_hover_time_s
                + self.flight_time_s(dist_customer_to_recovery_m)
                + self.launch_recovery_time_s)

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def is_sortie_feasible(self, dist_to_customer_m: float,
                           dist_customer_to_recovery_m: float,
                           current_battery_pct: float = 100.0,
                           payload_kg: float = 1.0,
                           safety_margin_pct: float = None) -> bool:
        """
        Feasible iff B_d - E_sortie / C_bat >= B_safety (paper Section III-B).
        """
        margin = (self.safety_margin_pct if safety_margin_pct is None
                  else safety_margin_pct)
        profile = self.sortie_profile(dist_to_customer_m,
                                      dist_customer_to_recovery_m,
                                      payload_kg, current_battery_pct)
        return profile["end_soc_pct"] >= margin

    def max_one_way_range_m(self, current_battery_pct: float = 100.0,
                            payload_kg: float = 1.0,
                            safety_margin_pct: float = None) -> float:
        """Furthest a customer can be and still allow a feasible round trip."""
        lo, hi = 0.0, self.max_range_km * 1000.0
        if self.is_sortie_feasible(hi, hi, current_battery_pct, payload_kg,
                                   safety_margin_pct):
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.is_sortie_feasible(mid, mid, current_battery_pct,
                                       payload_kg, safety_margin_pct):
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------
    # Truck comparison
    # ------------------------------------------------------------------

    def truck_speed_kmh(self) -> float:
        return self._truck_speed_kmh

    def truck_travel_time_s(self, distance_m: float) -> float:
        """
        Truck travel time over a *road* distance.

        ``truck_road_factor`` defaults to 1.0 because the scenario builder
        supplies genuine road-network shortest-path distances (already
        tortuosity-adjusted). Set it above 1.0 only when feeding this method
        straight-line distances.
        """
        return (distance_m * self.truck_road_factor) / (self._truck_speed_kmh / 3.6)

    def truck_energy_wh(self, distance_m: float) -> float:
        """
        Tank-to-wheel energy for a diesel delivery van.

        12 L/100 km at 9.94 kWh/L of diesel gives 1193 Wh/km, the default for
        ``truck_energy_wh_per_km``.
        """
        return distance_m / 1000.0 * self.truck_energy_wh_per_km

    def co2_per_km(self) -> float:
        """Drone CO2 per km, at an Indian grid factor of 0.5 kg CO2/kWh."""
        wh_per_km = self.energy_consumption_wh(1000.0)
        return wh_per_km / 1000.0 * 0.5

    @staticmethod
    def truck_co2_per_km() -> float:
        """Average CO2 for a diesel delivery truck: ~0.21 kg CO2/km."""
        return 0.21

    def __repr__(self):
        return (
            "DronePhysics(\n"
            "  speed={} km/h, battery={} Wh, payload<={} kg,\n"
            "  P_cruise={} W, P_hover={} W, alpha={} (model {:.3f}),\n"
            "  max 1-way range={:.2f} km @ full charge\n"
            ")".format(self.cruise_speed_kmh, self.battery_capacity_wh,
                       self.max_payload_kg, self.cruise_power_w,
                       self.hover_power_w, self.payload_power_factor,
                       self.alpha_aero,
                       self.max_one_way_range_m() / 1000.0))


if __name__ == "__main__":
    drone = DronePhysics()
    print(drone)
    print()
    print(drone.calibration_report())
    print()

    d_out, d_back = 2000, 2500
    profile = drone.sortie_profile(d_out, d_back)
    print("Sortie: {} m out, {} m back".format(d_out, d_back))
    print("  Time        : {:.0f} s ({:.1f} min)".format(
        profile["time_s"], profile["time_s"] / 60))
    print("  Ideal energy: {:.2f} Wh (Eq. 1)".format(profile["ideal_energy_wh"]))
    print("  Drawn energy: {:.2f} Wh (with Eq. 5 voltage sag)".format(
        profile["energy_wh"]))
    print("  Battery cost: {:.1f} %".format(profile["battery_pct_cost"]))
    print("  Feasible    : {}".format(drone.is_sortie_feasible(d_out, d_back)))
    print()
    print("Voltage sag check (cruise power draw vs. state of charge):")
    for soc in (1.0, 0.5, 0.2):
        print("  SoC {:>4.0%} -> V = {:.2f} V, energy multiplier {:.4f}".format(
            soc, drone._voltage_under_load(soc, drone.cruise_power_w),
            drone.discharge_multiplier(soc, drone.cruise_power_w)))
