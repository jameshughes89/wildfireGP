import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from wildfireGP.network import (
    FUEL,
    FUEL_MOISTURE,
    SLOPE,
    STATE,
    WIND_DIRECTION,
    WIND_SPEED,
    NodeState,
    create_grid,
    from_fbp_raster,
    from_landfire_raster,
    reset_states,
    set_fuel_moisture,
    set_wind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_geotiff(path, data):
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0]),
    ) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------------
# create_grid
# ---------------------------------------------------------------------------


def test_create_grid_node_count():
    graph = create_grid(4, 5)
    assert graph.number_of_nodes() == 20


def test_create_grid_all_unburned():
    graph = create_grid(4, 5)
    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)


def test_create_grid_fuel_in_range():
    graph = create_grid(10, 10)
    fuels = [graph.nodes[n][FUEL] for n in graph.nodes]
    assert all(0.0 <= f <= 1.0 for f in fuels)


def test_create_grid_slope_in_range():
    graph = create_grid(10, 10)
    slopes = [graph.nodes[n][SLOPE] for n in graph.nodes]
    assert all(0.0 <= s <= 1.0 for s in slopes)


def test_create_grid_reproducible():
    g1 = create_grid(5, 5, seed=42)
    g2 = create_grid(5, 5, seed=42)
    for node in g1.nodes:
        assert g1.nodes[node][FUEL] == g2.nodes[node][FUEL]
        assert g1.nodes[node][SLOPE] == g2.nodes[node][SLOPE]


def test_create_grid_different_seeds_differ():
    g1 = create_grid(5, 5, seed=0)
    g2 = create_grid(5, 5, seed=1)
    fuels_g1 = [g1.nodes[n][FUEL] for n in g1.nodes]
    fuels_g2 = [g2.nodes[n][FUEL] for n in g2.nodes]
    assert fuels_g1 != fuels_g2


def test_create_grid_no_wind_or_moisture_set():
    graph = create_grid(3, 3)
    assert WIND_SPEED not in graph.graph
    assert WIND_DIRECTION not in graph.graph
    assert FUEL_MOISTURE not in graph.graph


# ---------------------------------------------------------------------------
# set_wind / set_fuel_moisture
# ---------------------------------------------------------------------------


def test_set_wind():
    graph = create_grid(3, 3)
    set_wind(graph, speed=30.0, direction=180.0)
    assert graph.graph[WIND_SPEED] == 30.0
    assert graph.graph[WIND_DIRECTION] == 180.0


def test_set_fuel_moisture():
    graph = create_grid(3, 3)
    set_fuel_moisture(graph, 0.25)
    assert graph.graph[FUEL_MOISTURE] == 0.25


# ---------------------------------------------------------------------------
# reset_states
# ---------------------------------------------------------------------------


def test_reset_states():
    graph = create_grid(3, 3)
    for node in list(graph.nodes)[:3]:
        graph.nodes[node][STATE] = NodeState.BURNING
    for node in list(graph.nodes)[3:5]:
        graph.nodes[node][STATE] = NodeState.BURNED

    reset_states(graph)

    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)


# ---------------------------------------------------------------------------
# from_fbp_raster
# ---------------------------------------------------------------------------


def test_from_fbp_raster_node_count(tmp_path):
    rows, cols = 3, 4
    fuel_data = np.ones((rows, cols), dtype=np.int32)
    dem_data = np.zeros((rows, cols), dtype=np.float32)
    _write_geotiff(tmp_path / "fuel.tif", fuel_data)
    _write_geotiff(tmp_path / "dem.tif", dem_data)

    graph = from_fbp_raster(str(tmp_path / "fuel.tif"), str(tmp_path / "dem.tif"), fuel_map={1: 0.5})
    assert graph.number_of_nodes() == rows * cols


def test_from_fbp_raster_all_unburned(tmp_path):
    rows, cols = 3, 4
    _write_geotiff(tmp_path / "fuel.tif", np.ones((rows, cols), dtype=np.int32))
    _write_geotiff(tmp_path / "dem.tif", np.zeros((rows, cols), dtype=np.float32))

    graph = from_fbp_raster(str(tmp_path / "fuel.tif"), str(tmp_path / "dem.tif"), fuel_map={1: 0.8})
    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)


def test_from_fbp_raster_fuel_in_range(tmp_path):
    rows, cols = 4, 4
    fuel_data = np.array([[1, 2, 3, 4]] * rows, dtype=np.int32)
    dem_data = np.zeros((rows, cols), dtype=np.float32)
    _write_geotiff(tmp_path / "fuel.tif", fuel_data)
    _write_geotiff(tmp_path / "dem.tif", dem_data)

    fuel_map = {1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0}
    graph = from_fbp_raster(str(tmp_path / "fuel.tif"), str(tmp_path / "dem.tif"), fuel_map=fuel_map)
    assert all(0.0 <= graph.nodes[n][FUEL] <= 1.0 for n in graph.nodes)


def test_from_fbp_raster_unknown_code_maps_to_zero(tmp_path):
    rows, cols = 2, 2
    fuel_data = np.array([[999, 999], [999, 999]], dtype=np.int32)
    dem_data = np.zeros((rows, cols), dtype=np.float32)
    _write_geotiff(tmp_path / "fuel.tif", fuel_data)
    _write_geotiff(tmp_path / "dem.tif", dem_data)

    graph = from_fbp_raster(str(tmp_path / "fuel.tif"), str(tmp_path / "dem.tif"), fuel_map={1: 0.5})
    assert all(graph.nodes[n][FUEL] == 0.0 for n in graph.nodes)


# ---------------------------------------------------------------------------
# from_landfire_raster
# ---------------------------------------------------------------------------


def test_from_landfire_raster_node_count(tmp_path):
    rows, cols = 5, 3
    _write_geotiff(tmp_path / "fuel.tif", np.full((rows, cols), 101, dtype=np.int32))
    _write_geotiff(tmp_path / "dem.tif", np.zeros((rows, cols), dtype=np.float32))

    graph = from_landfire_raster(str(tmp_path / "fuel.tif"), str(tmp_path / "dem.tif"), fuel_map={101: 0.4})
    assert graph.number_of_nodes() == rows * cols


def test_from_landfire_raster_all_unburned(tmp_path):
    rows, cols = 3, 3
    _write_geotiff(tmp_path / "fuel.tif", np.full((rows, cols), 181, dtype=np.int32))
    _write_geotiff(tmp_path / "dem.tif", np.zeros((rows, cols), dtype=np.float32))

    graph = from_landfire_raster(str(tmp_path / "fuel.tif"), str(tmp_path / "dem.tif"), fuel_map={181: 0.7})
    assert all(graph.nodes[n][STATE] == NodeState.UNBURNED for n in graph.nodes)


def test_from_landfire_raster_fuel_in_range(tmp_path):
    rows, cols = 3, 3
    fuel_data = np.array([[91, 101, 141], [161, 181, 201], [99, 102, 182]], dtype=np.int32)
    dem_data = np.zeros((rows, cols), dtype=np.float32)
    _write_geotiff(tmp_path / "fuel.tif", fuel_data)
    _write_geotiff(tmp_path / "dem.tif", dem_data)

    graph = from_landfire_raster(str(tmp_path / "fuel.tif"), str(tmp_path / "dem.tif"))
    assert all(0.0 <= graph.nodes[n][FUEL] <= 1.0 for n in graph.nodes)


def test_from_landfire_raster_slope_in_range(tmp_path):
    rows, cols = 4, 4
    dem_data = np.array([[100, 150, 200, 250],
                         [110, 160, 210, 260],
                         [120, 170, 220, 270],
                         [130, 180, 230, 280]], dtype=np.float32)
    _write_geotiff(tmp_path / "fuel.tif", np.full((rows, cols), 101, dtype=np.int32))
    _write_geotiff(tmp_path / "dem.tif", dem_data)

    graph = from_landfire_raster(str(tmp_path / "fuel.tif"), str(tmp_path / "dem.tif"), fuel_map={101: 0.4})
    assert all(0.0 <= graph.nodes[n][SLOPE] <= 1.0 for n in graph.nodes)
