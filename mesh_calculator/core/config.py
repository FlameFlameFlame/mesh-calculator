"""
Configuration dataclasses and constants for mesh calculator.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MeshConfig:
    """System configuration parameters for mesh network calculation."""

    # H3 grid parameters
    h3_resolution: int = 8

    # Radio frequency parameters
    frequency_hz: float = 868e6  # 868 MHz
    mast_height_m: float = 5.0  # Tower mast height in meters

    # Network topology parameters
    max_towers_per_route: int = 10  # Maximum towers per route
    gap_repair_rounds: int = 5     # Max gap repair rounds (0 = disabled)
    routing_k_ring: int = 2  # k-ring radius for routing graph neighbor search
    road_buffer_m: float = 100.0  # Buffer around road cells in meters (0 = road-only)
    optimizer_search_radius_m: Optional[float] = None  # Optional planner-only search radius override
    gap_repair_search_radius_ladder_m: list[float] = field(
        default_factory=lambda: [0.0, 300.0, 600.0, 900.0]
    )
    fallback_initial_search_radius_ladder_m: list[float] = field(
        default_factory=lambda: [300.0, 600.0, 900.0, 1200.0]
    )
    auto_refine_h3_on_gradient: bool = True
    gradient_refine_threshold_m_per_km: float = 100.0
    gradient_refine_percentile: float = 90.0
    auto_refine_h3_max_resolution: int = 10
    export_full_grid_cells: bool = True
    max_coverage_radius_m: float = 15000.0  # Max tower coverage search radius in meters
    coverage_receiver_height_m: float = 1.5  # RX height above ground for tower-coverage maps

    # Link budget parameters
    tx_power_mw: float = 500.0              # Transmit power in milliwatts
    antenna_gain_dbi: float = 2.0           # Antenna gain, applied at both TX and RX
    receiver_sensitivity_dbm: float = -137.0  # Minimum receivable signal (LoRa SF12)
    # Optional policy gate: if set, LOS requires clearance >= threshold.
    # None keeps visibility decision purely link-budget based.
    min_fresnel_clearance_m: Optional[float] = None
    los_dense_sample_step_m: float = 50.0
    los_dense_max_samples: int = 400
    cell_anchor_margin_m: float = 10.0

    # Physical constants
    earth_radius_m: float = 6371000.0  # Earth radius in meters
    effective_earth_radius_factor: float = 4.0 / 3.0  # For radio horizon
    speed_of_light_m_s: float = 299792458.0  # Speed of light in m/s

    @property
    def effective_earth_radius_m(self) -> float:
        """Effective earth radius accounting for radio refraction."""
        return self.earth_radius_m * self.effective_earth_radius_factor

    @property
    def max_visibility_m(self) -> float:
        """Physics-derived maximum LOS range: distance at which FSPL equals link budget."""
        import math
        # FSPL(d) = 20*log10(4*pi*d*f/c); solve for d given link_budget
        d = (self.speed_of_light_m_s / (4 * math.pi * self.frequency_hz)) * (
            10 ** (self.link_budget_db / 20)
        )
        # Cap at 200 km — beyond that, earth curvature dominates regardless
        return min(d, 200_000.0)

    @property
    def wavelength_m(self) -> float:
        """Radio wavelength in meters."""
        return self.speed_of_light_m_s / self.frequency_hz

    @property
    def frequency_mhz(self) -> float:
        """Frequency in MHz for FSPL calculations."""
        return self.frequency_hz / 1e6

    @property
    def tx_power_dbm(self) -> float:
        """TX power in dBm."""
        import math
        return 10.0 * math.log10(self.tx_power_mw)

    @property
    def link_budget_db(self) -> float:
        """Total one-way link budget: TX power + both antenna gains − sensitivity."""
        return self.tx_power_dbm + 2.0 * self.antenna_gain_dbi - self.receiver_sensitivity_dbm


@dataclass
class InputPaths:
    """Input file paths for mesh calculator."""

    boundary: str  # GeoJSON polygon
    elevation: str  # GeoTIFF elevation data
    roads: str  # GeoJSON road network
    target_sites: str  # GeoJSON sites with priorities
    existing_towers: Optional[str] = None  # Optional seed towers
    city_boundaries: Optional[str] = None  # City boundary polygons
    grid_bundle: Optional[str] = None  # Optional persisted multi-resolution grid bundle


@dataclass
class OutputPaths:
    """Output file paths for mesh calculator."""

    towers: str = "output/towers.geojson"
    coverage: str = "output/coverage.geojson"
    report: str = "output/report.json"
    visibility_edges: str = "output/visibility_edges.geojson"


@dataclass
class RouteSpec:
    """Specification for a single user-chosen route to process."""

    route_id: str
    features: list  # GeoJSON feature dicts for this route
    site1: dict     # {name, lat, lon, site_height_m?}
    site2: dict     # {name, lat, lon, site_height_m?}
    max_towers_per_route: int = 10  # per-route tower limit


@dataclass
class MeshCalculatorConfig:
    """Complete configuration for mesh calculator."""

    parameters: MeshConfig = field(default_factory=MeshConfig)
    inputs: Optional[InputPaths] = None
    outputs: OutputPaths = field(default_factory=OutputPaths)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MeshCalculatorConfig':
        """Create config from dictionary (loaded from YAML)."""
        raw_params = dict(config_dict.get('parameters', {}))
        raw_params.pop('max_visibility_m', None)  # removed; now a computed property
        params = MeshConfig(**raw_params)

        inputs_dict = config_dict.get('inputs', {})
        inputs = InputPaths(**inputs_dict) if inputs_dict else None

        outputs_dict = config_dict.get('outputs', {})
        outputs = OutputPaths(**{
            k: v for k, v in outputs_dict.items()
            if k in OutputPaths.__dataclass_fields__
        })

        return cls(parameters=params, inputs=inputs, outputs=outputs)
