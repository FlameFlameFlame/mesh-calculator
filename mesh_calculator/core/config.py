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
    mast_height_m: float = 28.0  # Tower mast height in meters

    # Visibility and spacing constraints
    max_visibility_m: float = 70000.0  # 70 km maximum LOS distance
    tower_separation_m: float = 5000.0  # 5 km minimum tower separation

    # Network topology parameters
    hop_limit: int = 7  # Maximum hops within a cluster
    max_nodes_per_road: int = 10  # Maximum nodes per road segment
    routing_k_ring: int = 2  # k-ring radius for routing graph neighbor search

    # Physical constants
    earth_radius_m: float = 6371000.0  # Earth radius in meters
    effective_earth_radius_factor: float = 4.0 / 3.0  # For radio horizon
    speed_of_light_m_s: float = 299792458.0  # Speed of light in m/s

    @property
    def effective_earth_radius_m(self) -> float:
        """Effective earth radius accounting for radio refraction."""
        return self.earth_radius_m * self.effective_earth_radius_factor

    @property
    def wavelength_m(self) -> float:
        """Radio wavelength in meters."""
        return self.speed_of_light_m_s / self.frequency_hz

    @property
    def frequency_mhz(self) -> float:
        """Frequency in MHz for FSPL calculations."""
        return self.frequency_hz / 1e6


@dataclass
class InputPaths:
    """Input file paths for mesh calculator."""

    boundary: str  # GeoJSON polygon
    elevation: str  # GeoTIFF elevation data
    roads: str  # GeoJSON road network
    target_sites: str  # GeoJSON sites with priorities
    existing_towers: Optional[str] = None  # Optional seed towers
    city_boundaries: Optional[str] = None  # City boundary polygons


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
    site1: dict     # {name, lat, lon}
    site2: dict     # {name, lat, lon}
    max_towers: int = 10  # per-route tower limit


@dataclass
class MeshCalculatorConfig:
    """Complete configuration for mesh calculator."""

    parameters: MeshConfig = field(default_factory=MeshConfig)
    inputs: Optional[InputPaths] = None
    outputs: OutputPaths = field(default_factory=OutputPaths)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MeshCalculatorConfig':
        """Create config from dictionary (loaded from YAML)."""
        params = MeshConfig(**config_dict.get('parameters', {}))

        inputs_dict = config_dict.get('inputs', {})
        inputs = InputPaths(**inputs_dict) if inputs_dict else None

        outputs_dict = config_dict.get('outputs', {})
        outputs = OutputPaths(**outputs_dict)

        return cls(parameters=params, inputs=inputs, outputs=outputs)
