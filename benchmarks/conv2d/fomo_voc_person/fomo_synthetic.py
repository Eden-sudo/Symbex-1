"""Generate small, download-free multiclass FOMO datasets using NumPy."""

import argparse
import itertools
import os

import numpy as np

SIZE = 96
CELL = 8
GRID = 12


def _palette(n_classes):
    levels = [64, 94, 124, 154, 184, 214, 244]
    candidates = np.array(list(itertools.product(levels, repeat=3)), dtype=np.int32)
    chosen = []
    distances = np.full(len(candidates), np.iinfo(np.int32).max, dtype=np.int32)
    index = int(np.flatnonzero(np.all(candidates == [244, 64, 64], axis=1))[0])
    for _ in range(n_classes):
        color = candidates[index]
        chosen.append(color)
        distances = np.minimum(distances, np.sum((candidates - color) ** 2, axis=1))
        index = int(np.argmax(distances))
    return np.asarray(chosen, dtype=np.uint8)


def _select_objects(objects):
    winners = {}
    for index, obj in enumerate(objects):
        left, top, side, _ = obj
        cx, cy = left + side / 2, top + side / 2
        gx, gy = min(int(cx // CELL), GRID - 1), min(int(cy // CELL), GRID - 1)
        distance = (cx - (gx + .5) * CELL) ** 2 + (cy - (gy + .5) * CELL) ** 2
        key = (gx, gy)
        if key not in winners or distance < winners[key][0]:
            winners[key] = (distance, index)
    return [objects[index] for _, index in sorted(winners.values(), key=lambda item: item[1])]


def _integer(name, value, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is out of range')
    return value


def make(n_images, n_classes, seed, max_objects=3):
    """Generate data; colliding candidates lose to the closest cell center.

    Losing candidates are neither drawn nor included in the center metadata.
    """
    n_images = _integer('n_images', n_images, 0, np.iinfo(np.int32).max)
    n_classes = _integer('n_classes', n_classes, 1, 255)
    seed = _integer('seed', seed, 0)
    max_objects = _integer('max_objects', max_objects, 0, np.iinfo(np.int64).max - 1)
    rng = np.random.default_rng(seed)
    palette = _palette(n_classes)
    x = np.empty((n_images, SIZE, SIZE, 3), dtype=np.uint8)
    y = np.zeros((n_images, GRID, GRID), dtype=np.uint8)
    centers, owners, classes = [], [], []
    for image in range(n_images):
        x[image] = rng.integers(0, 32, size=(SIZE, SIZE, 3), dtype=np.uint8)
        objects = []
        for _ in range(rng.integers(0, max_objects + 1)):
            side = int(rng.integers(10, 21))
            class_id = int(rng.integers(1, n_classes + 1))
            left = int(rng.integers(0, SIZE - side + 1))
            top = int(rng.integers(0, SIZE - side + 1))
            objects.append((left, top, side, class_id))
        for left, top, side, class_id in _select_objects(objects):
            x[image, top:top + side, left:left + side] = palette[class_id - 1]
            cx, cy = left + side / 2, top + side / 2
            gx, gy = min(int(cx // CELL), GRID - 1), min(int(cy // CELL), GRID - 1)
            y[image, gy, gx] = class_id
            centers.append((cx, cy))
            owners.append(image)
            classes.append(class_id)
    return {
        'x': x, 'y': y,
        'centers': np.asarray(centers, dtype=np.float32).reshape(-1, 2),
        'image_of_center': np.asarray(owners, dtype=np.int32),
        'center_class': np.asarray(classes, dtype=np.int32),
        'class_names': np.asarray([f'class_{i}' for i in range(1, n_classes + 1)]),
    }


def save(path, data):
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'wb') as file:
        np.savez_compressed(file, **data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=int, required=True)
    parser.add_argument('--classes', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-objects', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        data = make(args.images, args.classes, args.seed, args.max_objects)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    save(args.out, data)


if __name__ == '__main__':
    main()
