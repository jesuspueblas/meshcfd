import numpy as np

from meshcfd.core.flow_visualization import (
    contour_segments,
    interpolate_to_regular_grid,
    streamline_polylines,
    vector_samples,
)
from meshcfd.core.model import FiniteVolumeMesh, MeshPatch


def regular_mesh():
    points = tuple((float(x), float(y)) for y in range(3) for x in range(4))
    cells = []
    for row in range(2):
        for column in range(3):
            a = row * 4 + column
            cells.append((a, a + 1, a + 5, a + 4))
    # Reuse the production constructor helper convention through a minimal valid mesh.
    faces, owners, neighbours, patches = [], [], [], []
    edge_map = {}
    for owner, cell in enumerate(cells):
        for edge in zip(cell, cell[1:] + cell[:1]):
            key = tuple(sorted(edge))
            if key in edge_map:
                neighbours[edge_map[key]] = owner
            else:
                edge_map[key] = len(faces)
                faces.append(edge)
                owners.append(owner)
                neighbours.append(-1)
                patches.append(0)
    cell_faces = [[] for _ in cells]
    cell_signs = [[] for _ in cells]
    for face_id, (owner, neighbour) in enumerate(zip(owners, neighbours)):
        cell_faces[owner].append(face_id)
        cell_signs[owner].append(1)
        if neighbour >= 0:
            cell_faces[neighbour].append(face_id)
            cell_signs[neighbour].append(-1)
    return FiniteVolumeMesh(
        points=points,
        cells=tuple(cells),
        faces=tuple(faces),
        face_owners=tuple(owners),
        face_neighbours=tuple(neighbours),
        face_patch_ids=tuple(-1 if neighbour >= 0 else 0 for neighbour in neighbours),
        patches=(MeshPatch(0, "boundary", tuple(i for i, n in enumerate(neighbours) if n < 0)),),
        cell_face_ids=tuple(tuple(ids) for ids in cell_faces),
        cell_face_signs=tuple(tuple(signs) for signs in cell_signs),
    )


def test_regular_grid_preserves_uniform_velocity_and_masks_domain():
    mesh = regular_mesh()
    grid = interpolate_to_regular_grid(mesh, (0, 1, 2, 0, 1, 2), (3,) * 6, (0,) * 6, (24, 16))
    assert np.allclose(grid.velocity_x[grid.valid], 3.0)
    assert np.allclose(grid.velocity_y[grid.valid], 0.0)
    assert grid.valid.any()


def test_contours_vectors_and_streamlines_are_generated():
    mesh = regular_mesh()
    grid = interpolate_to_regular_grid(mesh, (0, 1, 2, 0, 1, 2), (2,) * 6, (0,) * 6, (30, 20))
    assert contour_segments(grid, 5)
    assert vector_samples(grid, 2)
    lines = streamline_polylines(grid, 2)
    assert lines
    assert all(len(line) >= 3 for line in lines)
