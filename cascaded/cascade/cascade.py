from ..regressor.linear import RegressorLinear as RegressorLinear
from ..regressor.gausslinear import RegressorGausslinear as RegressorGausslinear
from ..regressor.metalinear import RegressorMetalinear as RegressorMetalinear
from ..regressor.metalinear2 import RegressorMetalinear2 as RegressorMetalinear2
from ..regressor.metalinear_fs import RegressorMetalinearFs as RegressorMetalinearFs
from ..regressor.metalinear_fsen import RegressorMetalinearFsen as RegressorMetalinearFsen
from ..descriptor.sift import DescriptorSift as DescriptorSift
from ..descriptor.sift_rotate import DescriptorSiftRotate as DescriptorSiftRotate
from ..toolkit.procrustes import procrustes
import cPickle
import numpy as np


class Cascade:
    def __init__(self, regressor, descriptor):
        c_reg = ''.join([p.capitalize() for p in regressor.split('_')])
        c_des = ''.join([p.capitalize() for p in descriptor.split('_')])
        self.num_steps, self.num_landmarks, self.num_dimensions = (0, None, None)
        self.steps = []

        # Initialize storage for intermediate training results
        self._train_params, self._train_mapping = None, None

        # Import regressor
        if 'Regressor' + c_reg in globals():
            self.regressor = globals()['Regressor' + c_reg]
        else:
            raise ImportError("There is no implementation for the regressor '" + str(regressor) + "'!")

        # Import descriptor
        if 'Descriptor' + c_des in globals():
            self.descriptor = globals()['Descriptor' + c_des]
        else:
            raise ImportError("There is no implementation for the descriptor '" + str(descriptor) + "'!")

    def clear_training_data(self):
        self._train_params, self._train_mapping = None, None

    def train(self, images, ground_truth, n_steps=5, augmenter=None, args=None, save_as=None, continue_previous=True):
        # Capture training function
        f_learn = getattr(self, '_train_step')

        # Convert ground truth to float32 if required
        ground_truth = ground_truth if ground_truth.dtype == np.float32 else ground_truth.astype(np.float32)

        # If no previous training, perform initialization
        if self.num_steps == 0:
            self.num_landmarks, self.num_dimensions = ground_truth.shape[1], ground_truth.shape[2]
            getattr(self, '_initialize_method')(images, ground_truth)

        # Encode ground truth
        ground_truth = getattr(self, 'encoder').encode_parameters(ground_truth)

        # Apply already learned cascade steps (if intermediate results not present or not used)
        if not continue_previous or self._train_params is None:
            self._train_params, self._train_mapping = self.align(
                images,
                augmenter=augmenter,
                args={'target': ground_truth},
            )

        # Start learning cascade
        for i in range(self.num_steps, n_steps):
            # Train cascade step
            regressor, preds = f_learn(images, ground_truth, self._train_params, self._train_mapping, i, args)
            self.steps.append(regressor)
            self._train_params += preds
            self.num_steps += 1

            # Free memory
            del preds

            # Output train error, save snapshot
            print 'Step ' + str(i+1) + ' MSE: ' + str(np.mean((ground_truth[self._train_mapping, ...] - self._train_params) ** 2))
            if save_as is not None:
                s_file = save_as[i] if isinstance(save_as, list) else save_as
                cPickle.dump(self, open(s_file, 'wb'), cPickle.HIGHEST_PROTOCOL)

    def align(self, images, num_steps=None, save_all=False, augmenter=None, args=None):
        n_steps = self.num_steps if num_steps is None else num_steps
        args = {} if args is None else args

        # Capture description and alignment functions
        f_init = getattr(self, '_initialize_instances')
        f_align = getattr(self, '_align_step')

        # Capture parameters and mapping between parameters and images
        params, mapping = (
            f_init(images),
            range(images.shape[0])
        ) if augmenter is None else ((
            np.cast[np.float32](augmenter),
            range(images.shape[0])
        ) if isinstance(augmenter, np.ndarray) else augmenter(
            images,
            f_init(images),
            self
        ))
        params = (params,)

        # Apply current cascade steps
        ret = [None] * n_steps
        for i in range(n_steps):
            params = f_align(images, params[0], mapping, i, args=args)
            params = params if isinstance(params, tuple) else (params,)
            ret[i] = (np.copy(params[0]),)+params[1:] if save_all else None

        # Return results
        return (ret if save_all else params[0]), mapping

    # Abstract methods
    # --------------------------------------------------

    def _initialize_method(self, images, ground_truth):
        raise NotImplementedError('_initialize_method not implemented for the selected cascaded method!')

    def _initialize_instances(self, images):
        raise NotImplementedError('_initialize_instances not implemented for the selected cascaded method!')

    def _train_step(self, images, ground_truth, params, mapping, i):
        raise NotImplementedError('_train_step not implemented for the selected cascaded method!')

    def _align_step(self, images, params, mapping, i, features=None, args=None):
        raise NotImplementedError('_align_step not implemented for the selected cascaded method!')

    # Helper methods
    # --------------------------------------------------

    def _get_angles(self, shapes):
        # Prepare unit vector to rotate
        vector = np.zeros((1, shapes.shape[2]), dtype=np.float32)
        vector[0, 0] = 1

        # Calculate rotation angle of the face wrt. the screeen plane, for each shape
        angles = np.empty((shapes.shape[0],), dtype=np.float32)
        for i, s in enumerate(shapes):
            _, _, tfm = procrustes(self.mean_shape, s - np.mean(s, axis=0, keepdims=True), reflection=False)
            vect = np.dot(vector, tfm['rotation'][:, :2])
            angles[i] = np.arctan2(vect[0, 1], vect[0, 0])

        return angles

    def _apply_rotations(self, shapes, angles, center=True):
        means = np.mean(shapes, axis=1)[:, None, :]
        shapes = np.copy(shapes - means if center else shapes)

        for i, (s, r) in enumerate(zip(shapes, angles)):
            shapes[i, ...] = np.dot(s, np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]]))

        return shapes
