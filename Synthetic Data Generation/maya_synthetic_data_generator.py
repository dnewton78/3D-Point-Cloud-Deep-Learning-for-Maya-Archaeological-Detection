"""Python rewrite of `Maya Project_Synthetic Data Generator.cs`.

The original Grasshopper/Rhino C# geometry types are replaced with NumPy arrays:

- Point3d[,] -> ndarray shaped (rows, cols, 3)
- double[,]  -> ndarray shaped (rows, cols)
- int[,]     -> ndarray shaped (rows, cols)
- Line       -> Line dataclass storing two XYZ NumPy vectors

The generator preserves the original workflow and class codes:
  2 = base/ground
  6 = generated structure

CSV output has the same five columns and no header:
  X,Y,Z,Slope,Class
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np


@dataclass(frozen=True)
class Line:
    """Simple 3D line segment replacing Rhino.Geometry.Line."""

    start: np.ndarray
    end: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def point_at_length(self, distance: float) -> np.ndarray:
        length = self.length
        if length == 0.0:
            return self.start.copy()
        t = distance / length
        return self.start + t * (self.end - self.start)

    @property
    def midpoint(self) -> np.ndarray:
        return (self.start + self.end) / 2.0


def get_random_double(rng: np.random.Generator, minimum: float, maximum: float) -> float:
    """Equivalent in intent to min + NextDouble() * (max - min)."""
    return float(rng.uniform(minimum, maximum))


def generate_gaussian_noise(
    mean: float,
    std_dev: float,
    rng: np.random.Generator,
    size=None,
):
    """Generate Gaussian noise.

    The C# source uses Box-Muller one value at a time. NumPy's normal generator gives
    the same target distribution and allows efficient array generation.
    """
    return rng.normal(mean, std_dev, size=size)


def generate_point_grid(
    x_size: float,
    y_size: float,
    pt_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the base point grid, slopes array, and class array.

    Note: the original C# uses x_size / pt_num (not pt_num - 1). This rewrite
    intentionally keeps that behavior.
    """
    if pt_num <= 0:
        raise ValueError("pt_num must be positive")

    x_incr = x_size / float(pt_num)
    y_incr = y_size / float(pt_num)

    x = np.arange(pt_num, dtype=np.float64) * x_incr
    y = np.arange(pt_num, dtype=np.float64) * y_incr
    xx, yy = np.meshgrid(x, y, indexing="ij")

    points = np.zeros((pt_num, pt_num, 3), dtype=np.float64)
    points[..., 0] = xx
    points[..., 1] = yy

    slopes = np.zeros((pt_num, pt_num), dtype=np.float64)
    classifications = np.full((pt_num, pt_num), 2, dtype=np.int32)
    return points, slopes, classifications


def generate_topography(
    hill_num: int,
    max_length: float,
    min_length: float,
    max_point: np.ndarray,
    min_point: np.ndarray,
    rng: np.random.Generator,
) -> List[Line]:
    """Generate random line segments used as hill/topography centerlines."""
    lines: List[Line] = []

    xmin, ymin = float(min_point[0]), float(min_point[1])
    xmax, ymax = float(max_point[0]), float(max_point[1])

    for _ in range(hill_num):
        rand_x = get_random_double(rng, xmin, xmax)
        rand_y = get_random_double(rng, ymin, ymax)

        length = get_random_double(rng, min_length, max_length)
        rand_x2 = get_random_double(rng, rand_x - length, rand_x + length)
        rand_y2 = get_random_double(rng, rand_y - length, rand_y + length)

        p1 = np.array([rand_x, rand_y, 0.0], dtype=np.float64)
        p2 = np.array([rand_x2, rand_y2, 0.0], dtype=np.float64)
        lines.append(Line(p1, p2))

    return lines


def _rotated_rectangle_edges(
    center: np.ndarray,
    width: float,
    length: float,
    angle: float,
) -> List[Line]:
    """Build four rotated rectangle edges centered on *center*."""
    local = np.array(
        [
            [-width / 2.0, -length / 2.0],
            [ width / 2.0, -length / 2.0],
            [ width / 2.0,  length / 2.0],
            [-width / 2.0,  length / 2.0],
        ],
        dtype=np.float64,
    )

    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    xy = local @ rotation.T + center[:2]
    corners = np.column_stack([xy, np.zeros(4, dtype=np.float64)])

    return [
        Line(corners[0], corners[1]),
        Line(corners[1], corners[2]),
        Line(corners[2], corners[3]),
        Line(corners[3], corners[0]),
    ]


def generate_structures_from_selected_lines(
    percent_line: float,
    min_size: float,
    max_size: float,
    rng: np.random.Generator,
    selected_lines: Sequence[Line],
) -> List[Line]:
    """Generate line or rectangular structures around selected terrain lines.

    This is the overload actually used by the original RunScript method.
    """
    lines: List[Line] = []

    for source_line in selected_lines:
        center = source_line.midpoint

        if rng.random() <= percent_line:
            length = get_random_double(rng, min_size, max_size)
            rand_x2 = get_random_double(rng, center[0] - length / 2.0, center[0] + length / 2.0)
            rand_y2 = get_random_double(rng, center[1] - length / 2.0, center[1] + length / 2.0)

            p2 = np.array([rand_x2, rand_y2, 0.0], dtype=np.float64)
            p3 = center + (center - p2)
            p3 = np.array([p3[0], p3[1], 0.0], dtype=np.float64)
            lines.append(Line(p3, p2))
        else:
            width = get_random_double(rng, min_size, max_size)
            length = get_random_double(rng, min_size, max_size)
            angle = get_random_double(rng, 0.0, 2.0 * math.pi)
            lines.extend(_rotated_rectangle_edges(center, width, length, angle))

    return lines


def generate_structures_random(
    structure_number: int,
    percent_line: float,
    min_size: float,
    max_size: float,
    max_point: np.ndarray,
    min_point: np.ndarray,
    rng: np.random.Generator,
) -> List[Line]:
    """Translate the second, unused C# generateStructures overload.

    It independently positions structures anywhere inside the grid bounds.
    """
    lines: List[Line] = []
    xmin, ymin = float(min_point[0]), float(min_point[1])
    xmax, ymax = float(max_point[0]), float(max_point[1])

    for _ in range(structure_number):
        rand_x = get_random_double(rng, xmin, xmax)
        rand_y = get_random_double(rng, ymin, ymax)
        p1 = np.array([rand_x, rand_y, 0.0], dtype=np.float64)

        if rng.random() <= percent_line:
            length = get_random_double(rng, min_size, max_size)
            rand_x2 = get_random_double(rng, rand_x - length, rand_x + length)
            rand_y2 = get_random_double(rng, rand_y - length, rand_y + length)
            p2 = np.array([rand_x2, rand_y2, 0.0], dtype=np.float64)
            lines.append(Line(p1, p2))
        else:
            width = get_random_double(rng, min_size, max_size)
            length = get_random_double(rng, min_size, max_size)
            angle = get_random_double(rng, 0.0, 2.0 * math.pi)

            # Rhino's Rectangle3d(Plane, width, length) overload is corner-based.
            # Reproduce that geometry, then rotate around p1 as in the C# source.
            local = np.array(
                [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]],
                dtype=np.float64,
            )
            c, s = math.cos(angle), math.sin(angle)
            rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
            xy = local @ rotation.T + p1[:2]
            corners = np.column_stack([xy, np.zeros(4, dtype=np.float64)])
            lines.extend(
                [
                    Line(corners[0], corners[1]),
                    Line(corners[1], corners[2]),
                    Line(corners[2], corners[3]),
                    Line(corners[3], corners[0]),
                ]
            )

    return lines


def distance_line_to_points_xy(line: Line, points_xy: np.ndarray) -> np.ndarray:
    """Vectorized shortest distance from XY points to a finite line segment."""
    start = line.start[:2]
    end = line.end[:2]
    segment = end - start
    length_sq = float(np.dot(segment, segment))

    if length_sq == 0.0:
        return np.linalg.norm(points_xy - start, axis=-1)

    rel = points_xy - start
    t = np.sum(rel * segment, axis=-1) / length_sq
    t = np.clip(t, 0.0, 1.0)
    projection = start + t[..., None] * segment
    return np.linalg.norm(points_xy - projection, axis=-1)


def distance_line_to_point(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    """Scalar compatibility form of the original distanceLineToPoint method."""
    return float(
        distance_line_to_points_xy(
            Line(np.asarray(start, dtype=float), np.asarray(end, dtype=float)),
            np.asarray(point, dtype=float)[:2],
        )
    )


def move_points_with_gaussian(
    height: float,
    center: float,
    std_dev: float,
    is_topography: bool,
    buffer: float,
    lines: Sequence[Line],
    points: np.ndarray,
    classifications: np.ndarray,
) -> None:
    """Raise grid points around line segments using the source Gaussian profile."""
    if std_dev <= 0.0:
        raise ValueError("std_dev must be > 0")

    points_xy = points[..., :2]

    # Keep the C# line-by-line update order. Terrain additions accumulate because
    # class remains 2. Once a structure changes a point to class 6, later structure
    # lines no longer modify it, matching the original condition.
    for line in lines:
        distance = distance_line_to_points_xy(line, points_xy)
        mask = (distance <= buffer) & (classifications == 2)
        if not np.any(mask):
            continue

        z_dist = height * np.exp(-((distance - center) ** 2) / (2.0 * std_dev**2))
        points[..., 2][mask] += z_dist[mask]

        if not is_topography:
            classifications[mask] = 6


def add_gaussian_noise(
    mean: float,
    std_dev: float,
    points: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Add independent Gaussian noise to X, Y, and Z, in place."""
    if std_dev < 0.0:
        raise ValueError("std_dev must be >= 0")
    points += generate_gaussian_noise(mean, std_dev, rng, size=points.shape)


def calculate_slope(points: np.ndarray) -> np.ndarray:
    """Calculate slope in degrees from the same eight-neighbor rule as the C# code."""
    rows, cols, _ = points.shape
    slope_sum = np.zeros((rows, cols), dtype=np.float64)
    count = np.zeros((rows, cols), dtype=np.float64)

    # The C# checks these eight offsets in this order. Since only sums are kept,
    # vectorizing them is mathematically equivalent while being far faster.
    offsets = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]

    for di, dj in offsets:
        i0 = max(0, -di)
        i1 = min(rows, rows - di)
        j0 = max(0, -dj)
        j1 = min(cols, cols - dj)

        ni0, ni1 = i0 + di, i1 + di
        nj0, nj1 = j0 + dj, j1 + dj

        current = points[i0:i1, j0:j1]
        neighbor = points[ni0:ni1, nj0:nj1]

        dx = neighbor[..., 0] - current[..., 0]
        dy = neighbor[..., 1] - current[..., 1]
        run = np.hypot(dx, dy)
        height_difference = current[..., 2] - neighbor[..., 2]

        # A zero XY run is extraordinarily unlikely after Gaussian noise, but the
        # guard avoids inf/nan if two coordinates ever coincide exactly.
        contribution = np.divide(
            np.abs(height_difference),
            run,
            out=np.zeros_like(run),
            where=run != 0.0,
        )

        slope_sum[i0:i1, j0:j1] += contribution
        count[i0:i1, j0:j1] += 1.0

    mean_rise_run = np.divide(
        slope_sum,
        count,
        out=np.zeros_like(slope_sum),
        where=count != 0.0,
    )
    return np.round(np.degrees(np.arctan(mean_rise_run)), 3)


def write_points_to_csv(
    points: np.ndarray,
    slopes: np.ndarray,
    classifications: np.ndarray,
    path: str | Path,
) -> None:
    """Write X,Y,Z,Slope,Class with no header, like the C# source."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = np.column_stack(
        [
            points[..., 0].ravel(),
            points[..., 1].ravel(),
            points[..., 2].ravel(),
            slopes.ravel(),
            classifications.ravel(),
        ]
    )

    # Use explicit row formatting so class IDs remain integers and coordinates
    # match the original three-decimal CSV representation.
    with path.open("w", encoding="utf-8", newline="") as file:
        for x, y, z, slope, cls in flat:
            file.write(f"{x:.3f},{y:.3f},{z:.3f},{slope:.3f},{int(cls)}\n")


def write_to_csv(
    datasets_points: Sequence[np.ndarray],
    datasets_slopes: Sequence[np.ndarray],
    datasets_classes: Sequence[np.ndarray],
    file_name: str,
    file_path: str | Path,
) -> None:
    """Write each generated dataset as `<index><file_name>.csv`."""
    output_dir = Path(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (points, slopes, classes) in enumerate(
        zip(datasets_points, datasets_slopes, datasets_classes)
    ):
        write_points_to_csv(
            points,
            slopes,
            classes,
            output_dir / f"{index}{file_name}.csv",
        )


def run_script(
    data_num: int,
    size: float,
    divisions: int,
    max_height: float,
    min_height: float,
    std_dev_max: float,
    std_dev_min: float,
    max_structure: int,
    min_structure: int,
    max_struct_size: float,
    min_struct_size: float,
    line_square_balance: float,
    hill_num: int,
    hill_max_h: float,
    hill_min_h: float,
    hill_max_w: float,
    hill_min_w: float,
    hill_max_l: float,
    hill_min_l: float,
    noise_mean: float,
    noise_sdv: float,
    csv_export: bool = False,
    file_path: str | Path = ".",
    file_name: str = "synthetic_data",
    seed: Optional[int] = None,
):
    """Python equivalent of the C# RunScript method.

    Returns a dictionary containing both the first dataset (matching the original
    Grasshopper outputs) and every generated dataset.
    """
    if data_num < 1:
        raise ValueError("data_num must be at least 1")
    if divisions < 0:
        raise ValueError("divisions must be >= 0")
    if not 0.0 <= line_square_balance <= 1.0:
        raise ValueError("line_square_balance must be between 0 and 1")
    if hill_num < 0:
        raise ValueError("hill_num must be >= 0")
    if min_structure < 0 or max_structure < min_structure:
        raise ValueError("structure bounds are invalid")

    rng = np.random.default_rng(seed)

    datasets_points: List[np.ndarray] = []
    datasets_slopes: List[np.ndarray] = []
    datasets_classes: List[np.ndarray] = []

    last_topography_lines: List[Line] = []
    last_structure_lines: List[Line] = []

    for _ in range(data_num):
        pt_num = divisions + 1
        points, slopes, classifications = generate_point_grid(size, size, pt_num)

        max_point = points[pt_num - 1, pt_num - 1].copy()
        min_point = points[0, 0].copy()

        # Generate topography.
        topography_lines = generate_topography(
            hill_num,
            hill_max_l,
            hill_min_l,
            max_point,
            min_point,
            rng,
        )

        hill_height = get_random_double(rng, hill_min_h, hill_max_h)
        hill_std_dev = get_random_double(rng, hill_min_w, hill_max_w)
        move_points_with_gaussian(
            hill_height,
            0.0,
            hill_std_dev,
            True,
            hill_std_dev * 3.0,
            topography_lines,
            points,
            classifications,
        )

        # The C# chooses sNum by casting a random double to int, making max_structure
        # effectively exclusive for normal positive ranges. Preserve that behavior.
        structure_num = int(get_random_double(rng, float(min_structure), float(max_structure)))

        selected_lines: List[Line] = []
        if structure_num > 0:
            if not topography_lines:
                raise ValueError(
                    "The original algorithm cannot select structures when hill_num is 0. "
                    "Use hill_num > 0 or set structure count to 0."
                )
            indices = rng.integers(0, len(topography_lines), size=structure_num)
            selected_lines = [topography_lines[int(i)] for i in indices]

        structure_lines = generate_structures_from_selected_lines(
            line_square_balance,
            min_struct_size,
            max_struct_size,
            rng,
            selected_lines,
        )

        structure_height = get_random_double(rng, min_height, max_height)
        structure_std_dev = get_random_double(rng, std_dev_min, std_dev_max)
        move_points_with_gaussian(
            structure_height,
            0.0,
            structure_std_dev,
            False,
            structure_std_dev * 2.0,
            structure_lines,
            points,
            classifications,
        )

        add_gaussian_noise(noise_mean, noise_sdv, points, rng)
        slopes = calculate_slope(points)

        datasets_points.append(points)
        datasets_slopes.append(slopes)
        datasets_classes.append(classifications)

        last_topography_lines = topography_lines
        last_structure_lines = structure_lines

    if csv_export:
        write_to_csv(
            datasets_points,
            datasets_slopes,
            datasets_classes,
            file_name,
            file_path,
        )

    return {
        # Equivalent to C# outputs points/classifications/slopes/A/B.
        "points": datasets_points[0],
        "classifications": datasets_classes[0],
        "slopes": datasets_slopes[0],
        "A": last_topography_lines,
        "B": last_structure_lines,
        # Extra convenience access to all generated datasets.
        "all_points": datasets_points,
        "all_classifications": datasets_classes,
        "all_slopes": datasets_slopes,
    }


# Optional example. Change these numbers to the values from your Grasshopper inputs.
if __name__ == "__main__":
    result = run_script(
        data_num=1,
        size=100.0,
        divisions=100,
        max_height=3.0,
        min_height=1.0,
        std_dev_max=2.0,
        std_dev_min=0.5,
        max_structure=8,
        min_structure=3,
        max_struct_size=15.0,
        min_struct_size=5.0,
        line_square_balance=0.5,
        hill_num=5,
        hill_max_h=8.0,
        hill_min_h=2.0,
        hill_max_w=15.0,
        hill_min_w=5.0,
        hill_max_l=30.0,
        hill_min_l=10.0,
        noise_mean=0.0,
        noise_sdv=0.05,
        csv_export=True,
        file_path="synthetic_output",
        file_name="maya_synthetic",
        seed=42,
    )

    print("Points shape:", result["points"].shape)
    print("Unique classes:", np.unique(result["classifications"]))
    print("Slope range:", result["slopes"].min(), "to", result["slopes"].max())
