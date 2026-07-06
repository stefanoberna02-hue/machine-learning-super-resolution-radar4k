import os
from PIL import Image
import numpy as np
import torch
import glob
import re
from collections import defaultdict

# Important to notice that whenever we apply enhancement to the dataset, for how we are parallelizing, images frames number are contigous for each enhancement family,
# that is: if we apply enhancment s on n original frames we end up with n*s final frames but they are stored as:
# <day>_<expt>_<frame>.png      with frame going back to 0 inside each experiment and each day
# ORDER: the s-1 frames past frame(i) are his enhancements and frame(i+s) is the next original frame, followed by his enhancement family
# PROBLEM: in the history we don't want to mix different enhancements but ideally we would like to load 41 original frames, 41 versions of the same enhancements for this frames, and so on for different enhancements and different frames
class SLAMDataset(torch.utils.data.Dataset):
    """
    Dataloader for the NEW SLAM-RF processed dataset:

    dataset_SLAM/
        train/
            lidar/
                <day>_exp<k>_<idx>.png
            radar/
                <day>_exp<k>_<idx>.png
        test/
            lidar/
            radar/

    This loader pairs radar–lidar images by identical filenames.
    It supports history M exactly like the original RadarHD loader.
    """

    _PATTERN = re.compile(r"^(?P<day>\d+)_exp(?P<exp>\d+)_(?P<idx>\d+)\.png$", re.IGNORECASE)

    # Here M really represents the history and not the number of channels, so if you want 41 channels you should use history M of 40
    # num_augs should analogously include the original frame, so if we are enhancing all the images by flipping them LR, TB and LR+TP; it means we have the original image and 3 enhanced versions so num_aum = 4
    def __init__(self, basepath, split, M=0, num_augs=1, mixed_augmentations_in_same_channel=False):
        """
        basepath : path to e.g. dataset_SLAM_112005/
        split    : 'train' or 'test'
        M        : history length (M past frames + 1 current frame)
        """
        # --- sanity checks on parameters ---
        if M < 0:
            raise ValueError(f"M must be >= 0, got {M}")

        if num_augs < 1:
            raise ValueError(f"num_augs must be >= 1, got {num_augs}")

        if mixed_augmentations_in_same_channel and num_augs > 1:
            print(
                "[WARN] mixed_augmentations_in_same_channel=True with num_augs>1: "
                "history will mix augmentations by construction."
            )

        self.basepath = basepath
        self.split = split
        self.history = M
        self.num_augs = num_augs

        self.lidar_dir = os.path.join(basepath, split, "lidar")
        self.radar_dir = os.path.join(basepath, split, "radar")

        # List all .png files
        lidar_files_raw = glob.glob(os.path.join(self.lidar_dir, "*.png"))
        radar_files_raw = glob.glob(os.path.join(self.radar_dir, "*.png"))

        # Build basename -> fullpath maps
        lidar_map = {os.path.basename(p): p for p in lidar_files_raw}
        radar_map = {os.path.basename(p): p for p in radar_files_raw}

        lidar_keys = set(lidar_map.keys())
        radar_keys = set(radar_map.keys())

        # Must match exactly
        if lidar_keys != radar_keys:
            missing_in_radar = sorted(lidar_keys - radar_keys)
            missing_in_lidar = sorted(radar_keys - lidar_keys)
            raise RuntimeError(
                "Mismatch between LiDAR and Radar filenames.\n"
                f"present in lidar only (first 10): {missing_in_radar[:10]}\n"
                f"present in radar only (first 10): {missing_in_lidar[:10]}\n"
                "Check that train_test_split generated symmetric directories."
            )

        # Parse key and sort numerically by (day, exp, idx)
        def parse_key(basename):
            m = self._PATTERN.match(basename)
            if m is None:
                raise ValueError(
                    f"Filename '{basename}' does not match '<day>_exp<k>_<idx>.png'"
                )
            day = int(m.group("day"))
            exp = int(m.group("exp"))
            idx = int(m.group("idx"))
            return day, exp, idx

        records = []
        for k in lidar_keys:
            day, exp, idx = parse_key(k)
            records.append((day, exp, idx, k))
        records.sort(key=lambda t: (t[0], t[1], t[2]))

        # Store references in deterministic numeric order (for compatibility)
        self.lidar_files = [lidar_map[k] for (_, _, _, k) in records]
        self.radar_files = [radar_map[k] for (_, _, _, k) in records]

        # Group by (day, exp) so history never crosses experiments/days
        groups = defaultdict(list)  # (day, exp) -> list of (idx, basename)
        for day, exp, idx, k in records:
            groups[(day, exp)].append((idx, k))
        for g in groups.values():
            g.sort(key=lambda t: t[0])  # sort by idx within group

        # --- Build input sequences depending on history M ---
        if M == 0:
            # No history → simple pairs radar[i], lidar[i]
            self.input_sequences = self.radar_files
            self.label_sequences = self.lidar_files

        elif not mixed_augmentations_in_same_channel:
            # Build windows of length M+1 for radar on different preprocessed series so that every M+1*spatial_size contains the history of frames to which we applied the same preprocessing
            self.input_sequences = []
            self.label_sequences = []

            # Process each experiment separately
            for (day, exp), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
                keys_in_group = [k for (_, k) in items]
                radar_group = [radar_map[k] for k in keys_in_group]
                lidar_group = [lidar_map[k] for k in keys_in_group]

                num_files = len(radar_group)

                if num_files % self.num_augs != 0:
                    raise ValueError(
                        f"Inside dataloader_slam.py : Number of files ({num_files}) is not a multiple of num_augs ({self.num_augs}).\n"
                        "This violates the assumption that each original frame has exactly num_augs augmentations (considering the original) "
                        "stored contiguously."
                    )

                T = num_files // self.num_augs  # numero di frame reali (per questo esperimento)

                for aug_id in range(self.num_augs):
                    for frame_id in range(M, T):
                        indices = [
                            (frame_id - j) * self.num_augs + aug_id
                            for j in reversed(range(M + 1))
                        ]

                        radar_seq = [radar_group[k] for k in indices]
                        lidar_tgt = lidar_group[frame_id * self.num_augs + aug_id]

                        self.input_sequences.append(radar_seq)
                        self.label_sequences.append(lidar_tgt)

        else:
            # here we keep the original ordering of the dataset where enhanced frames are grouped toghether in the same 41 channels
            self.input_sequences = []
            self.label_sequences = []

            for (day, exp), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
                keys_in_group = [k for (_, k) in items]
                radar_group = [radar_map[k] for k in keys_in_group]
                lidar_group = [lidar_map[k] for k in keys_in_group]

                for i in range(M, len(radar_group)):
                    # Take M previous frames + current frame (within same experiment/day)
                    window = radar_group[i - M : i + 1]
                    self.input_sequences.append(window)
                    self.label_sequences.append(lidar_group[i])

    def __len__(self):
        return len(self.input_sequences)

    def load_image(self, fname, normalize=True):
        """Load a grayscale PNG as a tensor [1, H, W]."""
        img = Image.open(fname).convert("L")
        arr = np.asarray(img, dtype=np.float32)

        if normalize:
            arr /= 255.0

        return torch.tensor(arr).unsqueeze(0)  # shape: (1, H, W)

    def __getitem__(self, index):
        if self.history == 0:
            radar_img = self.load_image(self.input_sequences[index])
            lidar_img = self.load_image(self.label_sequences[index])
        else:
            # Gather M+1 radar frames
            radar_stack = []
            for path in self.input_sequences[index]:
                radar_stack.append(self.load_image(path))

            # Stack over channel dimension → shape (M+1, H, W)
            radar_img = torch.cat(radar_stack, dim=0)

            # Single lidar frame (1, H, W)
            lidar_img = self.load_image(self.label_sequences[index])

        return radar_img, lidar_img


# import os
# from PIL import Image
# import numpy as np
# import torch
# import glob

# # Important to notice that whenever we apply enhancement to the dataset, for how we are parallelizing, images frames number are contigous for each enhancement family,
# # that is: if we apply enhancment s on n original frames we end up with n*s final frames but they are stored as:
# # <day>_<expt>_<frame>.png      with frame going back to 0 inside each experiment and each day
# # ORDER: the s-1 frames past frame(i) are his enhancements and frame(i+s) is the next original frame, followed by his enhancement family
# # PROBLEM: in the history we don't want to mix different enhancements but ideally we would like to load 41 original frames, 41 versions of the same enhancements for this frames, and so on for different enhancements and different frames
# class SLAMDataset(torch.utils.data.Dataset):
#     """
#     Dataloader for the NEW SLAM-RF processed dataset:
    
#     dataset_SLAM/
#         train/
#             lidar/
#                 <day>_<exp>_<idx>.png
#             radar/
#                 <day>_<exp>_<idx>.png
#         test/
#             lidar/
#             radar/

#     This loader pairs radar–lidar images by identical filenames.
#     It supports history M exactly like the original RadarHD loader.
#     """
# # Here M really represents the history and not the number of channels, so if you want 41 channels you should use history M of 40
# # num_augs should analogously include the original frame, so if we are enhancing all the images by flipping them LR, TB and LR+TP; it means we have the original image and 3 enhanced versions so num_aum = 4
#     def __init__(self, basepath, split, M=0, num_augs=1, mixed_augmentations_in_same_channel=False):
#         """
#         basepath : path to e.g. dataset_SLAM_112005/
#         split    : 'train' or 'test'
#         M        : history length (M past frames + 1 current frame)
#         """
#         # --- sanity checks on parameters ---
#         if M < 0:
#             raise ValueError(f"M must be >= 0, got {M}")

#         if num_augs < 1:
#             raise ValueError(f"num_augs must be >= 1, got {num_augs}")

#         if mixed_augmentations_in_same_channel and num_augs > 1:
#             print(
#                 "[WARN] mixed_augmentations_in_same_channel=True with num_augs>1: "
#                 "history will mix augmentations by construction."
#             )
#         self.basepath = basepath
#         self.split = split
#         self.history = M
#         self.num_augs = num_augs

#         self.lidar_dir = os.path.join(basepath, split, "lidar")
#         self.radar_dir = os.path.join(basepath, split, "radar")

#         # List all .png files in deterministic temporal order
#         lidar_files = sorted(glob.glob(os.path.join(self.lidar_dir, "*.png")))
#         radar_files = sorted(glob.glob(os.path.join(self.radar_dir, "*.png")))

#         # Extract basenames to verify pairing
#         lidar_keys = [os.path.basename(f) for f in lidar_files]
#         radar_keys = [os.path.basename(f) for f in radar_files]

#         # Must match exactly
#         assert lidar_keys == radar_keys, (
#             "Mismatch between LiDAR and Radar filenames.\n"
#             "Check that train_test_split generated symmetric directories."
#         )

#         # Store references
#         self.lidar_files = lidar_files
#         self.radar_files = radar_files

#         # --- Build input sequences depending on history M ---
#         if M == 0:
#             # No history → simple pairs radar[i], lidar[i]
#             self.input_sequences = radar_files
#             self.label_sequences = lidar_files

#         elif not mixed_augmentations_in_same_channel:
#             # Build windows of length M+1 for radar on different preprocessed series so that every M+1*spatial_size contains the history of frames to which we applied the same preprocessing
#             self.input_sequences = []
#             self.label_sequences = []

#             num_files = len(radar_files)

#             if num_files % self.num_augs != 0:
#                 raise ValueError(
#                     f"Inside dataloader_slam.py : Number of files ({num_files}) is not a multiple of num_augs ({self.num_augs}).\n"
#                     "This violates the assumption that each original frame has exactly num_augs augmentations (considering the original) "
#                     "stored contiguously."
#                 )
#             T = len(radar_files) // self.num_augs  # numero di frame reali

#             for aug_id in range(self.num_augs):
#                 for frame_id in range(M, T):
#                     indices = [
#                         (frame_id - j) * self.num_augs + aug_id
#                         for j in reversed(range(M+1))
#                     ]

#                     radar_seq = [radar_files[k] for k in indices]
#                     lidar_tgt = lidar_files[frame_id * self.num_augs + aug_id]
#                     # --- DEBUG: visualize grouping ---
#                     if frame_id < M + 2 and aug_id < min(2, self.num_augs):
#                         pretty = [
#                             f"(global={k}, frame={k // self.num_augs}, aug={k % self.num_augs})"
#                             for k in indices
#                         ]
#                         print(
#                             f"[DEBUG] aug_id={aug_id}, frame_id={frame_id} → history:\n  "
#                             + "\n  ".join(pretty)
#                         )

#                     self.input_sequences.append(radar_seq)
#                     self.label_sequences.append(lidar_tgt)
#         else:
#             #here we keep the original ordering of the dataset where enhanced frames are grouped toghether in the same 41 channels
#             self.input_sequences = []
#             self.label_sequences = []
#             for i in range(M, len(radar_files)):
#                 # Take M previous frames + current frame
#                 window = radar_files[i-M:i+1]
#                 self.input_sequences.append(window)
#                 self.label_sequences.append(lidar_files[i])



#     def __len__(self):
#         return len(self.input_sequences)

#     def load_image(self, fname, normalize=True):
#         """Load a grayscale PNG as a tensor [1, H, W]."""
#         img = Image.open(fname).convert("L")
#         arr = np.asarray(img, dtype=np.float32)

#         if normalize:
#             arr /= 255.0

#         return torch.tensor(arr).unsqueeze(0)  # shape: (1, H, W)

#     def __getitem__(self, index):

#         if self.history == 0:
#             radar_img = self.load_image(self.input_sequences[index])
#             lidar_img = self.load_image(self.label_sequences[index])

#         else:
#             # Gather M+1 radar frames
#             radar_stack = []
#             for path in self.input_sequences[index]:
#                 radar_stack.append(self.load_image(path))
            
#             # Stack over channel dimension → shape (M+1, H, W)
#             radar_img = torch.cat(radar_stack, dim=0)

#             # Single lidar frame (1, H, W)
#             lidar_img = self.load_image(self.label_sequences[index])

#         return radar_img, lidar_img