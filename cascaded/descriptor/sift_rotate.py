from descriptor import Descriptor

import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage.feature import hog


class DescriptorSiftRotate(Descriptor):
    def __init__(self):
        Descriptor.__init__(self)

    # def _extract(self, images, coords, mapping, args):
    #     assert images.shape[1] == images.shape[2]
    #     n_inst = coords.shape[0]
    #
    #     nb = args.get('num_bins', 8)
    #     rotations = args.get('rotations', np.zeros((n_inst,), dtype=np.float32))
    #     win_sizes = args.get('window_sizes', 32)
    #     win_sizes = win_sizes if isinstance(win_sizes, np.ndarray) else np.ones((n_inst,), dtype=np.int32) * win_sizes
    #
    #     # Prepare descriptors
    #     descriptors = np.zeros(tuple(coords.shape[:2])+(nb*4*4,), dtype=np.float32)
    #
    #     # Fill descriptors
    #     coords, vis = np.copy(coords) - images.shape[1] / 2.0, np.empty(coords.shape[:2], dtype=np.bool)
    #     for i, (c, r, mp, ws) in enumerate(zip(coords, rotations, mapping, win_sizes)):
    #         hsize, qsize = ws/2, ws/4
    #
    #         # Get maximum window half-size, rotate and pad image
    #         im = np.pad(
    #             rotate(images[mp, ...], 57.2957*r),
    #             ((hsize, hsize), (hsize, hsize)),
    #             'constant', constant_values=0
    #         )
    #
    #         # Rotate geometry, set landmarks visibility
    #         ims = im.shape[0] - hsize
    #         c = np.dot(c, np.array([[np.cos(r), np.sin(r)], [-np.sin(r),  np.cos(r)]])) + im.shape[0] / 2.0
    #         vis[i, :] = (c[:, 0] >= hsize) & (c[:, 1] >= hsize) & (c[:, 0] < ims) & (c[:, 1] < ims)
    #
    #         # Extract descriptors from each interest window
    #         for j, (jc, jv) in enumerate(zip(c, vis[i, :])):
    #             descriptors[i, j, :] = hog(
    #                 im[jc[0]-hsize:jc[0]+hsize, jc[1]-hsize:jc[1]+hsize],
    #                 orientations=nb,
    #                 pixels_per_cell=(qsize, qsize),
    #                 cells_per_block=(1, 1)
    #             ) if jv else 0
    #
    #     # Normalize descriptors, return extracted information
    #     return descriptors.reshape((len(mapping), -1)), vis

    def _extract(self, images, coords, mapping, args):
        assert images.shape[1] == images.shape[2]
        n_inst = coords.shape[0]

        nb = args.get('num_bins', 8)
        rotations = args.get('rotations', np.zeros((n_inst,), dtype=np.float32))
        win_sizes = args.get('window_sizes', 32)
        win_sizes = win_sizes if isinstance(win_sizes, np.ndarray) else np.ones((n_inst,), dtype=np.int32) * win_sizes

        # Prepare descriptors
        descriptors = np.zeros(tuple(coords.shape[:2]) + (nb * 4 * 4,), dtype=np.float32)

        # Fill descriptors
        coords, vis = np.copy(coords) - images.shape[1] / 2.0, np.empty(coords.shape[:2], dtype=np.bool)
        for i, (c, r, mp, ws) in enumerate(zip(coords, rotations, mapping, win_sizes)):
            hsize, qsize = ws / 2, ws / 4

            # Get maximum window half-size, rotate and pad image
            im = np.pad(
                np.float32(rotate(images[mp, ...], 57.2957 * r)),
                ((hsize+1, hsize+1), (hsize+1, hsize+1)),
                'constant', constant_values=0
            )

            # Rotate geometry, set landmarks visibility
            ims = im.shape[0] - hsize - 2
            c = np.round(np.dot(c, np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])) + im.shape[0] / 2.0 - 1)
            vis[i, :] = (c[:, 0] >= hsize) & (c[:, 1] >= hsize) & (c[:, 0] < ims) & (c[:, 1] < ims)

            dx, dy = im[2:, 1:-1] - im[:-2, 1:-1], im[1:-1, 2:] - im[1:-1, :-2]
            mag = np.sqrt(dx ** 2 + dy ** 2)
            bin = np.cast[np.uint8]((np.arctan2(dy, dx) + 3.14159) // (6.2831853 / nb))
            for j, (jc, jv) in enumerate(zip(c, vis[i, :])):
                if jv:
                    t_mag = mag[jc[0] - hsize:jc[0] + hsize, jc[1] - hsize:jc[1] + hsize]
                    t_bin = bin[jc[0] - hsize:jc[0] + hsize, jc[1] - hsize:jc[1] + hsize]
                    t_mag = t_mag.reshape((4, qsize, 4, qsize)).swapaxes(1, 2).reshape((16, -1))
                    t_bin = t_bin.reshape((4, qsize, 4, qsize)).swapaxes(1, 2).reshape((16, -1))
                    descriptors[i, j, :] = np.concatenate([
                        np.bincount(b, weights=m, minlength=nb) for m, b in
                        zip(t_mag, t_bin)
                    ])

        # Normalize descriptors, return extracted information
        return descriptors.reshape((len(mapping), -1)), vis
