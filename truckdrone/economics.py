"""
Money and emissions: turning kilometres, minutes and watt-hours into rupees
and kilograms of CO2.

The Review-1 proposal promised a dispatcher that balances time, energy *and
cost*. Time and energy fall out of the simulator directly; cost needs a model,
and every number in it is an assumption. They are collected here, each with
its source and reasoning, so that anyone disputing the cost results can see
exactly which input to change -- and so a sensitivity analysis is one edit,
not a search.

Prices are for urban India in 2026 and are deliberately round. The point of
the cost objective is the *trade-off* it creates against time and energy, not
a forecast to the rupee.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    # Driver plus van overhead, per hour on shift. A delivery driver in a
    # metro earns roughly Rs 20-25k a month; with vehicle overhead and
    # employer costs the loaded rate is around Rs 300/h.
    driver_inr_per_hour: float = 300.0

    # Diesel retail price and energy content. 9.94 kWh/L is the figure the
    # physics module already uses to derive 1193 Wh/km from 12 L/100 km.
    diesel_inr_per_l: float = 92.0
    diesel_kwh_per_l: float = 9.94

    # Commercial electricity tariff for charging drone packs on the truck.
    electricity_inr_per_kwh: float = 9.0

    # Battery wear, amortised per sortie: a 100 Wh pack costs about Rs 6,000
    # and is retired after about 500 full cycles, so roughly Rs 12 per cycle.
    # Maintenance is folded in at the same order of magnitude.
    drone_inr_per_sortie: float = 12.0

    # A diesel van idling while it waits for a drone burns about 0.8 L/h.
    # Engine off while the driver is at a customer's door.
    idle_l_per_hour: float = 0.8

    # Emission factors. Diesel: 2.68 kg CO2 per litre burned. Indian grid:
    # 0.71 kg CO2 per kWh (CEA baseline, 2023-24).
    diesel_kg_co2_per_l: float = 2.68
    grid_kg_co2_per_kwh: float = 0.71

    # ------------------------------------------------------------------
    def diesel_litres(self, truck_wh):
        return truck_wh / 1000.0 / self.diesel_kwh_per_l

    def idle_wh(self, idle_s):
        """Fuel energy (Wh) burned idling for ``idle_s`` seconds."""
        return idle_s / 3600.0 * self.idle_l_per_hour * self.diesel_kwh_per_l * 1000.0

    def cost_inr(self, time_s, truck_wh, drone_wh, n_sorties):
        return (time_s / 3600.0 * self.driver_inr_per_hour
                + self.diesel_litres(truck_wh) * self.diesel_inr_per_l
                + drone_wh / 1000.0 * self.electricity_inr_per_kwh
                + n_sorties * self.drone_inr_per_sortie)

    def co2_kg(self, truck_wh, drone_wh):
        return (self.diesel_litres(truck_wh) * self.diesel_kg_co2_per_l
                + drone_wh / 1000.0 * self.grid_kg_co2_per_kwh)


DEFAULT_COSTS = CostModel()
