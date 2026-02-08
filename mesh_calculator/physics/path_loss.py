"""
Path loss calculation (FSPL + knife-edge diffraction).

Based on h3_path_loss.sql from the original implementation.
"""
import math


def compute_path_loss(
    distance_m: float,
    frequency_hz: float,
    clearance_m: float,
    d1_m: float = None,
    d2_m: float = None
) -> float:
    """
    Calculate total path loss (FSPL + single knife-edge diffraction).

    Implements the formula from h3_path_loss.sql (lines 43-71).

    Args:
        distance_m: Total distance in meters
        frequency_hz: Radio frequency in Hz
        clearance_m: Worst-case Fresnel clearance in meters (negative if obstructed)
        d1_m: Distance from source to obstacle (optional)
        d2_m: Distance from obstacle to destination (optional)

    Returns:
        Total path loss in dB

    Reference:
        h3-mesh-placement/functions/h3_path_loss.sql:43-71
    """
    if distance_m <= 0:
        raise ValueError(f"distance_m must be positive (got {distance_m})")

    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive (got {frequency_hz})")

    # Convert to km and MHz for standard formulas
    distance_km = distance_m / 1000.0
    freq_mhz = frequency_hz / 1000000.0

    # Free-space path loss (FSPL) in dB
    # FSPL = 20*log10(d_km) + 20*log10(f_MHz) + 32.44
    fspl_db = 20 * math.log10(distance_km) + 20 * math.log10(freq_mhz) + 32.44

    # If clearance >= 0, no diffraction loss
    if clearance_m >= 0:
        return fspl_db

    # Calculate diffraction loss for obstructed path
    # Use provided obstacle distances or assume midpoint
    if d1_m is None or d1_m <= 0 or d2_m is None or d2_m <= 0:
        d1_km = distance_km / 2.0
        d2_km = distance_km / 2.0
    else:
        d1_km = d1_m / 1000.0
        d2_km = d2_m / 1000.0

    # First Fresnel zone radius at obstacle
    # r1 = 17.32 * sqrt(d1 * d2 / (f * (d1 + d2)))
    r1 = 17.32 * math.sqrt(d1_km * d2_km / (freq_mhz * (d1_km + d2_km)))

    if r1 <= 0:
        raise ValueError(f"Fresnel radius must be positive (got {r1})")

    # Fresnel-Kirchhoff diffraction parameter
    # nu = sqrt(2) * |clearance| / r1
    nu = math.sqrt(2) * abs(clearance_m) / r1

    # Knife-edge diffraction loss (ITU-R P.526)
    # L_d = 6.9 + 20*log10(sqrt((nu - 0.1)^2 + 1) + nu - 0.1)
    if nu > 0:
        diffraction_loss_db = 6.9 + 20 * math.log10(
            math.sqrt((nu - 0.1)**2 + 1) + nu - 0.1
        )
    else:
        diffraction_loss_db = 0.0

    # Total path loss
    total_loss_db = fspl_db + diffraction_loss_db

    return total_loss_db


def fspl_only(distance_m: float, frequency_hz: float) -> float:
    """
    Calculate free-space path loss only (no diffraction).

    Args:
        distance_m: Distance in meters
        frequency_hz: Frequency in Hz

    Returns:
        FSPL in dB
    """
    distance_km = distance_m / 1000.0
    freq_mhz = frequency_hz / 1000000.0
    return 20 * math.log10(distance_km) + 20 * math.log10(freq_mhz) + 32.44
