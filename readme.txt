Sign Language Translation Research - README


Introduction

This project is part of a research initiative focused on advancing sign language translation through machine learning techniques. 
The primary goal of this research is to build a robust system capable of recognizing and translating hand gestures from videos into another sign language.
While the current stage of the research has not successfully trained a Transformer model, significant progress has been made in data collection, preprocessing, and annotation. 
The dataset includes keypoint features extracted from videos, annotated with essential metadata such as time sequence, origin language (ASL or CSL), and the numerical digit represented by the hand gesture.
This dataset is open for public use. Researchers and developers are welcome to use it for training, experimentation, or to further refine the model.

Dataset Overview
The dataset consists of CSV files generated from processed videos. Each file contains the following:
Time Sequence: The frame order within the video.
Language Identifier:
1: ASL (American Sign Language)
2: CSL (Chinese Sign Language)
Digit: The numerical representation of the hand gesture (1-10).
59 Keypoints: Each with 3D coordinates (X, Y, Z), covering:
21 keypoints per hand
15 keypoints for the face
2 keypoints for the shoulders
CSV Format
Each CSV file follows this structure:
Time	Digit	LanguageID	Point_1_X	Point_1_Y	Point_1_Z	...	Point_59_Z
0	1	1	0.123	0.456	0.789	...	0.321
1	1	1	0.234	0.567	0.890	...	0.432

Future Vision
The ultimate aim is to develop a model inspired by large-scale language models that can:

Successfully classify the digit represented in a sign language video.
Generalize to unseen data by leveraging large-scale, diverse training datasets.
Be applied to real-world use cases by analyzing new videos placed in a separate folder to predict the corresponding sign language digit.

Contact
For questions, feedback, or access to the dataset:

Researcher: Titus Weng
Email: tw013@bucknell.edu
Institution: Bucknell University