# PoseMamba Colab demo

A single Colab notebook that runs [PoseMamba](https://github.com/nankingjing/PoseMamba) (AAAI 2025, monocular 3D human pose estimation) on a video and produces a side-by-side result: the 2D skeleton on the footage and a 3D skeleton.

## Usage
1. Open `PoseMamba_Demo_Clean.ipynb` in [Google Colab](https://colab.research.google.com) (File > Open notebook > GitHub, or upload it).
2. Runtime > Change runtime type > **T4 GPU**.
3. Run the cells top to bottom. Setup takes about 15 minutes, mostly compiling GPU kernels.
4. For each video, run Step 3. Set `USE_SAMPLE = False` to upload your own.

Colab sessions reset after a timeout, so setup has to be repeated in a new session.

## What the notebook does
- Clones the original PoseMamba repo and installs its requirements.
- Patches the CUDA kernel source for CUDA 13 (`cub::LaneId` and `cub::CTA_SYNC` were removed) and builds both the `core` and `oflex` kernels.
- Downloads the model weights from Hugging Face and the `demo/` folder (detector code, YOLOv3/HRNet weights, sample videos) from the Google Drive link in the PoseMamba README.
- Allows `torch.load(weights_only=False)` for the authors' checkpoint, which newer PyTorch rejects by default.

No weights or model code are stored in this repo. All credit for the model goes to the PoseMamba authors; see their repository for the license and citation.

## Source copy
`PoseMamba/` is a copy of the project source with the fixes above already applied (see `PoseMamba/LICENSE`). The notebook itself still clones the original repo and patches it at runtime.
