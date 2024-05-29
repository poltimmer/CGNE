import os
import random

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.misc import ceildiv, param_list


class SequenceHDF5Dataset(Dataset):
    def __init__(self, data_dir, attachment_only=True, transform=None, specific_classes=None, sequence_length=30,
                 overlap=1, stride=1, use_step_number=True):
        self.data_dir = data_dir
        self.transform = transform
        self.attachment_only = attachment_only
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.stride = stride
        self.use_step_number = use_step_number
        self.params_transform = ParamsTransform()

        self.classes = np.array(sorted(os.listdir(data_dir)))

        if specific_classes is not None:
            self.classes = np.array([f for f in self.classes if f in specific_classes])

        # if os.path.join(data_dir, self.classes[0]) is a file and not a directory:
        if len(self.classes) > 0 and os.path.isfile(os.path.join(data_dir, self.classes[0])):
            self.h5_files = self.classes
            self.files_in_class = np.array([[f] for f in self.classes])
        else:
            self.h5_files = []
            self.files_in_class = []
            for c in self.classes:
                files_in_class = [os.path.join(c, file) for file in sorted(os.listdir(os.path.join(data_dir, c)))]
                self.h5_files += files_in_class
                self.files_in_class.append(files_in_class)
            self.h5_files = np.array(self.h5_files)
            self.files_in_class = np.array(self.files_in_class)

        self._calc_n_steps()

    def _calc_n_steps(self):
        self.file_to_steps = {}
        self.total_steps = 0
        for file in self.h5_files:
            with h5py.File(os.path.join(self.data_dir, file), 'r', swmr=True) as f:
                steps = len([k for k in f.keys() if 'step_' in k])
                self.file_to_steps[file] = steps
                self.total_steps += ceildiv(steps - 1, ((self.sequence_length - self.overlap) * self.stride))

    def __len__(self):
        return self.total_steps

    def _get_file_and_step(self, index):
        for f, steps in self.file_to_steps.items():
            num_sequences = ceildiv(steps - 1, ((self.sequence_length - self.overlap) * self.stride))
            if index < num_sequences:
                file = f
                step_index = index * ((self.sequence_length - self.overlap) * self.stride)
                return file, step_index
            index -= num_sequences

    def __getitem__(self, index):
        file, start_step_idx = self._get_file_and_step(index)
        return self._get_sequence_from_file(file, start_step_idx)

    def _get_sequence_from_file(self, file, start_step_idx, sequence_length=None, stride=None, first_and_last=False):
        if sequence_length is None:
            sequence_length = self.sequence_length
        if stride is None:
            stride = self.stride

        # Open the hdf5 file and extract the step
        with h5py.File(os.path.join(self.data_dir, file), 'r', swmr=True) as f:
            step_names = sorted([k for k in f.keys() if 'step_' in k], key=lambda x: int(x.split('_')[1]))
            if first_and_last:
                step_names = step_names[::len(step_names) - 1]
            else:
                step_names = step_names[start_step_idx:start_step_idx + sequence_length * stride:stride]
            steps = []
            t = []
            for i, step_name in enumerate(step_names):
                if self.use_step_number:
                    step_idx = int(step_name.split('_')[1])
                    t.append(step_idx)
                else:
                    t.append(i + (start_step_idx // stride))
                step = torch.tensor(f[step_name]['attachment'], dtype=torch.float32)
                step = step.unsqueeze(0)
                if not self.attachment_only:
                    step = torch.stack([step, torch.tensor(f[step_name]['ice'], dtype=torch.float32)], dim=0)
                steps.append(step)

            params = torch.tensor([f.attrs[p] for p in param_list])
            steps = torch.stack(steps)
            t = torch.tensor(t, dtype=torch.float32)

        # Apply the transform if specified
        if self.transform:
            steps = self.transform(steps)

        params = self.params_transform(params)

        return steps, params, t

    def get_sequence_from_class(self, class_idx, start_step_idx, file_idx=None, sequence_length=None, stride=None):
        if file_idx is None:
            file_idx = random.randint(0, len(self.files_in_class[class_idx]) - 1)
        file = self.files_in_class[class_idx, file_idx]
        return self._get_sequence_from_file(file, start_step_idx, sequence_length, stride)


class FirstAndLastHDF5Dataset(SequenceHDF5Dataset):
    def __init__(self, data_dir, attachment_only=True, transform=None, specific_classes=None, use_step_number=True):
        super().__init__(data_dir=data_dir, attachment_only=attachment_only, transform=transform,
                         specific_classes=specific_classes, sequence_length=2, overlap=1, stride=1,
                         use_step_number=use_step_number)

    def _calc_n_steps(self):
        self.file_to_steps = {}
        self.total_steps = len(self.h5_files)

    def __getitem__(self, index):
        file = self.h5_files[index]
        return self._get_sequence_from_file(file, 0, first_and_last=True)


class FirstStepHDF5Dataset(SequenceHDF5Dataset):
    def __init__(self, data_dir, attachment_only=True, transform=None, specific_classes=None, stride=1,
                 use_step_number=True):
        super().__init__(data_dir=data_dir, attachment_only=attachment_only, transform=transform,
                         specific_classes=specific_classes, sequence_length=2, overlap=0, stride=stride,
                         use_step_number=use_step_number)

    def _calc_n_steps(self):
        self.file_to_steps = {}
        self.total_steps = len(self.h5_files)

    def __getitem__(self, index):
        file = self.h5_files[index]
        return self._get_sequence_from_file(file, 0)


class ParamsTransform:
    def __init__(self):
        # Parameters for standardization
        # 'alpha', 'beta', 'gamma', 'theta', 'kappa', 'mu', 'rho'
        self.means = torch.tensor([1.8481e-01, 1.2362e-01, 7.5704e-05, -2.1965, 1.2412e-02, 6.5471e-02, -2.3105e-01],
                                  dtype=torch.float64)
        self.stds = torch.tensor([1.0171e-01, 0.0449, 1.3514e-05, 0.3401, 7.2223e-03, 4.0388e-02, 0.0125],
                                 dtype=torch.float64)

        # Parameters for Box-Cox
        self.box_cox_lambdas = torch.tensor([-4.37, 3.83, 0.270], dtype=torch.float32)  # beta, rho, theta

    def __call__(self, params):
        # Apply transformations
        # Box-Cox for beta, rho, theta
        i_beta = param_list.index('beta')
        i_rho = param_list.index('rho')
        i_theta = param_list.index('theta')
        params[i_beta] = self.box_cox_transform(params[i_beta], self.box_cox_lambdas[0])
        params[i_rho] = self.box_cox_transform(params[i_rho], self.box_cox_lambdas[1])
        params[i_theta] = self.box_cox_transform(params[i_theta], self.box_cox_lambdas[2])
        # Standardization
        params = (params - self.means) / self.stds

        return params.to(torch.float32)

    @staticmethod
    def box_cox_transform(y, lmbda):
        return torch.log(y) if lmbda == 0 else (torch.pow(y, lmbda) - 1) / lmbda


def reverse_box_cox_transform(y, lmbda):
    return torch.exp(y) if lmbda == 0 else torch.pow(lmbda * y + 1, 1 / lmbda)


def reverse_transform_params(params_transformed):
    params_transformed = params_transformed.clone().cpu().to(torch.float64)
    # Parameters for de-standardization
    means = torch.tensor([1.8481e-01, 1.2362e-01, 7.5704e-05, -2.1965, 1.2412e-02, 6.5471e-02, -2.3105e-01],
                         dtype=torch.float64, device=params_transformed.device)
    stds = torch.tensor([1.0171e-01, 0.0449, 1.3514e-05, 0.3401, 7.2223e-03, 4.0388e-02, 0.0125],
                        dtype=torch.float64, device=params_transformed.device)

    # Parameters for reverse Box-Cox beta, rho, theta
    box_cox_lambdas = torch.tensor([-4.37, 3.83, 0.270], dtype=torch.float32, device=params_transformed.device)

    # Reverse transformations
    # De-standardization
    params_transformed = (params_transformed * stds) + means
    i_beta = param_list.index('beta')
    i_rho = param_list.index('rho')
    i_theta = param_list.index('theta')
    # Reverse Box-Cox for beta, rho, theta
    params_transformed[i_beta] = reverse_box_cox_transform(params_transformed[i_beta], box_cox_lambdas[0])
    params_transformed[i_rho] = reverse_box_cox_transform(params_transformed[i_rho], box_cox_lambdas[1])
    params_transformed[i_theta] = reverse_box_cox_transform(params_transformed[i_theta], box_cox_lambdas[2])

    return params_transformed


def get_equal_sequence_collate_fn(sequence_length):
    def equal_sequence_collate_fn(batch):
        sequences, params_list = zip(*batch)

        # Pad or randomly crop sequences
        processed_sequences = []
        for seq in sequences:
            if len(seq) < sequence_length:
                padding = [torch.zeros_like(seq[0]) for _ in range(sequence_length - len(seq))]
                seq = seq + padding
            elif len(seq) > sequence_length:
                start_idx = random.randint(0, len(seq) - sequence_length)
                seq = seq[start_idx:start_idx + sequence_length]
            processed_sequences.append(torch.stack(seq))

        return torch.stack(processed_sequences), torch.stack(params_list)

    return equal_sequence_collate_fn


def padding_collate_fn(batch):
    sequences, params_list, t = zip(*batch)

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=False, padding_value=0)

    padded_t = pad_sequence(t, batch_first=False, padding_value=-1)

    # Compute sequence lengths
    seq_lengths = torch.tensor([len(seq) for seq in sequences])

    # Create masks based on sequence lengths
    max_len = seq_lengths.max()
    masks = (torch.arange(max_len).expand(len(sequences), max_len) < seq_lengths.unsqueeze(1)).T

    return padded_sequences, masks, torch.stack(params_list), padded_t
