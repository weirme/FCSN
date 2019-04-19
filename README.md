# Video_Summary_using_FCSN
A PyTorch implementation of FCSN in paper "Video Summarization Using Fully Convolutional Sequence Networks"

#### Paper Publication Info
Video Summarization Using Fully Convolutional Networks
Mrigank Rochan, Linwei Ye, and Yang Wang 
European Conference on Computer Vision (ECCV), 2018   

Link: http://www.cs.umanitoba.ca/~mrochan/projects/eccv18/fcsn.html

#### Dataset

A preprocessed TVSum dataset (downsampled to 320 frames per video) is available [here](https://drive.google.com/file/d/1cz78IaFjFcYmO7XdORpnp9wI5S4522oB/view?usp=sharing). There are 50 groups in the hdf5 file named `video_1` to `video_50` . Datasets in each group is as follows:

| name  | description |
| :-------: | :----------------------------: |
|    `length`       |scalar, number of video frames|
| `feature` |       shape (320, 1024)        |
|  `label`  |          shape (320, )          |
| `change_points` | shape (n_segments, 2) <br>stores begin and end of each segment |
| `n_frame_per_seg` | shape (n_segments) <br>number of frames in each segment |
| `user_summary` | shape (20, length) <br>summary from 20 users, each row is a binary vector |



