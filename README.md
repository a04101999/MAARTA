# MAARTA

## Table of Contents

- [Dataset](#dataset)
- [Usage](#usage)

# Simulated Error Dataset: <a name="dataset"></a>

Due to unavailability of Real world error dataset, this study was conducted on the simulated error dataset:

https://drive.google.com/drive/folders/1RzlGzvJ9Dl01dgrNlhNedY7JWhQ3pmJE?usp=sharing

It contains two files:

1. Error dataset with missing or masked fixations
2. Labels for the the cases

## Dataset Generation

This dataset was derived from the [EGD-CXR](https://physionet.org/content/egd-cxr/1.0.0/) dataset. The EGD-CXR dataset was created using an eye-tracking system to monitor a radiologist's gaze while interpreting and reading 1,083 publicly available chest X-ray (CXR) images. We provide a Python file within the `dataset_generation` directory with the code used to generate the simulated error dataset. To reproduce the synthesized error dataset, you will need to download the original EGD-CXR dataset from PhysioNet and extract the audio_segmentation_transcripts and fixation folders as displayed within the `dataset_generation` directory. Then run the `egd_cxr_processing.py` file with the following command:

```bash
python3 egd_cxr_processing.py
```

For convenience we have provided the respective folders in the `dataset_generation` directory and simply running the `egd_cxr_processing.py` file will generate the synthesized error dataset.

