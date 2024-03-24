import math

def determine_direction(dx, dy, num_sectors):
    """
    Determines the directional label for a given dx, dy pair based on the specified number of polar sectors,
    with the first sector centered at 0 degrees (directly to the right).

    Parameters:
    - dx: Change in x-coordinate.
    - dy: Change in y-coordinate.
    - num_sectors: Number of sectors to divide the polar coordinate system into.

    Returns:
    - A vector of length num_sectors with a 1 at the index corresponding to the sector of the dx, dy pair,
      and 0s elsewhere.
    """
    if num_sectors < 1:
        raise ValueError("Number of sectors must be at least 1.")
    
    angle = math.atan2(dy, dx)
    angle_degrees = (math.degrees(angle) + 360) % 360
    degrees_per_sector = 360 / num_sectors
    adjusted_angle_degrees = (angle_degrees + degrees_per_sector / 2) % 360
    sector = int(adjusted_angle_degrees // degrees_per_sector)
    
    # Initialize a vector with zeros
    direction_vector = [0] * num_sectors
    # Set the appropriate sector to 1
    direction_vector[sector] = 1

    return direction_vector


