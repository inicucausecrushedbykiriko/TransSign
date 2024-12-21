# Sign2Sign - A First Attempt

## Introduction

This project is part of a research initiative aimed at advancing communication for Deaf, Mute, and Hard-of-Hearing individuals by creating tools for **direct sign language translation**. Current translation tools often rely on intermediate spoken languages, limiting their accessibility. Sign2Sign seeks to address this by building a system capable of recognizing and translating hand gestures directly between sign languages.

The initial focus of the project was on translating gestures for numbers (1-10) between **American Sign Language (ASL)** and **Chinese Sign Language (CSL)**. While our Transformer-based attempts are still in progress, the groundwork laid through data collection, preprocessing, and model experimentation provides a strong foundation for future advancements.

This research was recognized at the **26th Symposium on Virtual and Augmented Reality**, earning the **2nd Best Paper Award** in the Undergraduate Workshop. Details of the earlier version using CNN-based models can be found in the published paper ([link](https://doi.org/10.5753/svr_estendido.2024.244070)) or in the project’s previous commits.

---

## Dataset Overview

The previous CNN version of Sign2Sign translator is in the previous commit named **purend**.

For the current transformer one's datasets, it consists of keypoint features extracted from gesture videos, annotated with essential metadata:
- **Time Sequence**: Frame order within the video.
- **Language ID**:
  - `1`: ASL (American Sign Language)
  - `2`: CSL (Chinese Sign Language)
- **Digit**: Numerical representation of the gesture (1-10).
- **59 Keypoints**: Each with 3D coordinates (X, Y, Z), covering:
  - 21 points per hand,
  - 15 points for the face,
  - 2 points for the shoulders.

Files are organized into directories (`asl_features` and `csl_features`) and saved in CSV format.

### Example CSV Structure
| Time | Digit | LanguageID | Point_1_X | Point_1_Y | Point_1_Z | ... | Point_59_Z |
|------|-------|------------|-----------|-----------|-----------|-----|------------|
| 0    | 1     | 1          | 0.123     | 0.456     | 0.789     | ... | 0.321      |
| 1    | 1     | 1          | 0.234     | 0.567     | 0.890     | ... | 0.432      |

This dataset is openly available for research and development purposes.

---

## Current Progress

While Transformer-based models have yet to achieve success, the project has made substantial strides in:
- **Data Preprocessing**: Extracting and organizing gesture features with time-sequential annotations.
- **Initial Attempts**: Early experiments with CNN-based models achieved moderate success, which is detailed in the [published paper](https://doi.org/10.5753/svr_estendido.2024.244070).
- **Model Design**: Developing Transformer architectures to model temporal dependencies across frames.

---

## Future Vision

The ultimate goal is to create a scalable system that can:
1. Recognize digits represented in sign language videos with high accuracy.
2. Generalize to unseen data by leveraging diverse training datasets.
3. Analyze new videos to predict the corresponding digit or report unrecognizable gestures.

Future iterations will integrate **large-scale language models (LLMs)** for contextual understanding, refine Transformer architectures, and explore real-world applications through immersive XR interfaces.

---

## Contact

For questions, feedback, or access to the dataset:
- **Researcher**: Titus Weng and Prof. SingChun Lee  
- **Email**: [tw013@bucknell.edu](mailto:tw013@bucknell.edu), [singchun.lee@bucknell.edu](mailto:singchun.lee@bucknell.edu)  
- **Institution**: Bucknell University  
