# AnimNCA
## Neural Cellular Automata for Animation

AnimNCA is inspired by "Growing Neural Cellular Automata" (Mordvintsev et al., 2020), which showed that an NCA can grow an image from a single cell and keep it stable. AnimNCA asks whether an NCA can learn a looping walk cycle instead of a static image. The model gets no frame input or timer. Each cell only sees its neighbors, so the timing of the animation has to be kept by the cells themselves.

### Results
Model is grown from a single cell and continues the 6 frame walk cycle. Original walk cycle on the left for comparison. the original and NCA are aligned once at start, but will drift. the NCA runs freely and is not synced to it. 

https://github.com/user-attachments/assets/167d53a9-1456-454d-94e0-df3085c0ec57


### How It Works

Each pixel is a cell with 24 channels: 4 visible (RGBA) and 20 hidden. Every step, each cell looks at its neighbors and a small network decides how to update it. Only a random half of cells update each step, so there's no global sync.

Starting from a single cell, the NCA grows into the first frame, then has 30 steps to reach each next frame. It's checked against the target at the end of each frame's slot. Each run can start on any frame, as long as it keeps the right order after that.

Training uses the sample pool from Mordvintsev et al. States are saved and picked up again in later training steps, so the NCA learns to keep the loop running indefinitely instead of just playing it once. Each batch, the worst sample is replaced with a fresh seed so it doesn't forget how to grow. Since the target changes over time here, each saved state also keeps track of its age and which frame it started on, so it's always checked against the right frame.

### Running
Download ```anim_nca.py``` & ```sonic.gif```. optionally download ```anim_nca_weights.pt``` if you don't want to train from scratch. 
```
   pip install torch pygame pillow numpy
   python anim_nca.py
```
It loads the saved weights if present, otherwise trains from scratch. 

### Controls
- Left Click: Erase at mouse position
- Right Click: Spawn new seed
- Scroll Wheel: Change eraser size
- Up/Down: Change speed
- R: Restart
- Space (While Training): Pause training to watch grow
- T: Resume Training
- Escape: Quit
