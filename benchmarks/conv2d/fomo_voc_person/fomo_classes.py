"""Per-cell VOC class targets for multiclass FOMO.

The shared dataset format is x (N,96,96,3) uint8, y (N,12,12) uint8,
centers (M,2) float32, image_of_center (M,) int32, center_class (M,) int32,
and class_names (K,) str. Zero is background; class_names follows the order
of classes and image indices are zero-based. This module returns per-image
targets; it does not write a dataset.
"""
import numpy as np

SIZE, CELL = 96, 8
GRID = SIZE // CELL
VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')


def grid_from_objects(objects, w0, h0, classes):
    """Return the class grid and all retained centers and their class IDs."""
    class_ids = {}
    for name in classes:
        if name not in VOC_CLASSES:
            raise ValueError(f"unknown VOC class: {name}")
        if name in class_ids:
            raise ValueError(f"duplicate VOC class: {name}")
        class_ids[name] = len(class_ids) + 1

    grid = np.zeros((GRID, GRID), dtype=np.uint8)
    best_distance = np.full((GRID, GRID), np.inf)
    centers, center_class = [], []
    if isinstance(objects, dict):
        objects = [objects]
    for o in objects:
        if o['name'] not in class_ids or o.get('difficult', '0') == '1':
            continue
        b = o['bndbox']
        cx = (float(b['xmin']) + float(b['xmax'])) / 2 * SIZE / w0
        cy = (float(b['ymin']) + float(b['ymax'])) / 2 * SIZE / h0
        gx, gy = min(int(cx // CELL), GRID - 1), min(int(cy // CELL), GRID - 1)
        class_id = class_ids[o['name']]
        centers.append((cx, cy))
        center_class.append(class_id)
        d2 = (cx - (gx + 0.5) * CELL) ** 2 + (cy - (gy + 0.5) * CELL) ** 2
        if d2 < best_distance[gy, gx]:
            best_distance[gy, gx] = d2
            grid[gy, gx] = class_id
    return grid, np.asarray(centers, dtype=np.float32).reshape(-1, 2), np.asarray(center_class, dtype=np.int32)
