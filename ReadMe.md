# TrackMania Nations Forever AI

This projects aim to design and implement a gym environnement running in real time on the game TrackMania Nations Forever, using TMInterface. The objective will then be to implement a bunch on different classic RL methods on this environnement.

# Game screen image analysis

I use MSS to quickly take screenshots of the game window and then use a lot of post-processing using OpenCV to clean up the image to only keep the track borders.

<img src="assets/frame_1.png" width="512">

The screenshot is then resized and the middle of it is kept to only keep the track borders. After that, the image is converted to grayscale and then to binary using a threshold. We apply a canny filter, then a dilation and finaly a gaussian blur before thresholding again.

<img src="assets/processed_1.png" width="512">

Finally in order to get an input for our RL agents we simulate a LIDAR on the processed frame by tracing rays starting from the bottom middle of the processed image.

<img src="assets/raytrace_1.png" width="512">

The input given to our agents can be either of these images or the array of distances computed from the raycasting.

# Current Objectives
- [x] Extract and process frames from the game
- [x] Send controls to the game using windows api
- [x] Get car status from TMInterface
- [x] Implement arrow keys input simulation
- [ ] Implement gamepad input simulation
- [x] Run TMInterface in separate thread to get data from the game more efficiently
- [ ] Implement general training routines for most RL methods
- [ ] Allow for online or offline training
- [ ] High level agent factory for easy testing
- [ ] WandB benchmarking tool and dashboard
- [ ] (OPTIONAL) Look into implementing parallel envs by running multiple instances of the game
- [ ] (OPTIONAL) Automate setup of the game


# References

- TMInterface : https://donadigo.com/tminterface/
- TMRL : https://github.com/trackmania-rl/tmrl
- MSS : https://python-mss.readthedocs.io/
