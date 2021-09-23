# Pose

Pose is a Python script for live head pose estimation

## Setup (MacOS)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install -r requirements.txt
```

If installing dlib throws an error, ensure the command line tools are installed.
```bash
xcode-select --install
```


Then, using [Homebrew](https://brew.sh/), run:
```bash
brew install cmake
pip install cmake
brew install dlib
pip install dlib
```

Also ensure that **`shape_predictor_68.dat`** is located in the same folder as **`pose.py`**


## Usage

### Modifications due to variable webcam framerates and variable output framerates:

In pose.py, edit the following to match your webcam's framerate:

```python
# CHANGE FRAMERATE ACCORDING TOWEBCAM
vs = VideoStream(src=0, framerate=30).start()
```

You can now run the script as follows: (press "q" to quit)

```bash
python3 pose.py
```

If you would like to use the video output file feature, you will also need to edit the output framerate to approximately match the average framerate you received running the script on your hardware.
```python
out = cv2.VideoWriter(
    "poseEstimate.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    15, #REPLACE WITH AVERAGE FPS
    (frame.shape[1], frame.shape[0]),
)
``` 
