DigBodha 1
Text-Prompted Motion & Collision-Risk Estimation from a Single Camera
Open-vocabulary video segmentation · Ego-motion compensation · Calibration-free time to contact · Interpretable collision-risk assessment

Python Computer Vision Dataset

Computer Vision Course Project · M.Tech CSE (Data Science & AI) · IIITDM Kancheepuram

Overview · Architecture · Quick Start · Results · Limitations · References

[!IMPORTANT] Research prototype, not a deployable safety system. The evaluated implementation operates offline and its risk estimates can be wrong or unavailable. Do not rely on it for driving, navigation, or collision avoidance.

Overview
DigBodha 1 takes a monocular video and text prompts such as vehicle,person, then tracks the matching objects and estimates their motion and potential collision risk relative to the camera.

Ordinary object detection answers “What is in the scene?” DigBodha extends that to:

Where is it? Instance masks, bounding boxes, and left/centre/right zones.
Which object is it? Persistent track IDs while an object remains in view.
How is it moving? Raw image motion and camera-compensated motion.
Is it approaching? Apparent-size expansion over time.
When could it reach the camera’s plane? Time to contact (TTC) with a low/high uncertainty band.
Will it pass or be on course? Predicted lateral miss distance at contact.
How trustworthy is the estimate? Explicit reliability flags and reasons.
The pipeline deliberately combines a pretrained SAM 3.1 video model for text-prompted instance segmentation and tracking with classical computer vision and geometric estimation for motion, TTC, and risk. No task-specific model training or fine-tuning was performed for this project.

Key capabilities
Capability	Implementation
Text-prompted instance segmentation	SAM 3.1 video predictor
Object identity across video chunks	Mask IoU, Hungarian assignment, short velocity-based grace window
Image feature selection	Shi–Tomasi corners
Point tracking	Pyramidal Lucas–Kanade optical flow + forward–backward consistency
Camera-motion estimation	Background homography fitted with RANSAC
Independent image motion	Observed optical flow minus homography-predicted camera flow
Time to contact	Least-squares line fit to inverse object height, 1/s
Collision-course prediction	Extrapolated normalized lateral offset
Risk assessment	Rule-based safe, caution, danger, uncertain
Evaluation	KITTI Tracking 3D annotations, frame and hazard-event metrics
System architecture
Monocular video / JPG frame folder

SAM 3.1

Text prompts: vehicle, person, ...

Instance masks + object IDs

Chunk overlap and ID linking

Clean masks and bounding boxes

Background corner tracking

Shi–Tomasi + pyramidal Lucas–Kanade

RANSAC homography

Camera-compensated object motion

Object size and horizontal offset history

Inverse-size TTC fit

Collision-course / miss-distance fit

Motion labels and reliability signals

Risk rules + reliability checks

frames.csv

tracks.json

annotated.mp4

run.json

Stage 1 — Text-prompted segmentation and tracking
src/segment.py loads the SAM 3.1 multiplex video predictor and follows objects matching the requested concepts.

SAM tracking state grows with video length, so the implementation uses 60-frame chunks with a 2-frame overlap by default. Objects are linked between chunks through overlap-mask IoU and Hungarian matching (threshold 0.3). A 10-frame velocity-shifted grace window helps recover objects that briefly disappear around a chunk boundary. When a chunk fails due to GPU memory pressure, processing can retry at smaller chunk sizes, down to 15 frames.

The tracker supports up to 16 objects in the default pipeline configuration. It does not re-identify an object after that object completely leaves the camera view and re-enters.

Stage 2 — Ego-motion and independent object motion
src/egomotion.py estimates the camera-induced motion that would make stationary scene points move in the image:

Downscale frames to 640 px width and convert to grayscale.
Exclude detected objects (including a dilated boundary) from background feature selection.
Detect up to 800 Shi–Tomasi corners on the background.
Track corners using pyramidal Lucas–Kanade with a forward–backward consistency check.
Fit a 3×3 homography to the background correspondences using RANSAC.
Track up to 60 feature points per object and subtract the predicted camera motion.
For a tracked image point p, independent (camera-compensated) motion is:

$$ \mathbf{v}{\mathrm{own}}=\mathbf{p}{t}-H(\mathbf{p}_{t-1}), $$

where H(p) denotes the homography-warped point in Cartesian image coordinates. If too few valid points survive, object-motion estimation falls back to the mask centroid.

A homography is exact for pure camera rotation or a planar scene; with translating cameras and general 3D backgrounds it is an approximation because of parallax. The estimated camera motion supports motion labels and reliability checking, while TTC is computed from raw image-size measurements to preserve motion relative to the camera.

Stage 3 — Time to contact, collision course, and reliability
src/ttc.py measures the bounding box of the largest connected component of each object mask. Object height is the preferred apparent-size measurement; width can be used when the top or bottom is truncated but the sides remain usable.

Time to contact from looming
In a pinhole camera, an object of physical height H at depth Z has apparent image height s:

$$ s(t)=\frac{fH}{Z(t)}, \qquad \frac{1}{s(t)}=\frac{Z(t)}{fH}. $$

For approximately constant closing speed, 1/s is linear in time. DigBodha fits

$$ u(t)=\frac{1}{s(t)}=a+bt $$

using a recent 0.8-second history, then estimates remaining TTC as

$$ \boxed{\mathrm{TTC}=\frac{u_{\mathrm{now}}}{-b}}\quad\text{for }b<0. $$

The object’s real height and the camera focal length cancel: the TTC estimate does not require explicit metric distance, object size, or camera calibration under this model. The estimator needs at least five observations, limits estimates to a 10-second horizon, and derives a low/high band from the fitted slope’s standard error. These bounds are diagnostic uncertainty estimates, not calibrated probabilistic confidence intervals.

The final implementation fits 1/s rather than log(s): the latter produced delayed TTC estimates under constant-speed approach in the team’s experiments.

Collision course
TTC answers when an object reaches the camera’s depth plane, not whether it will be in the camera’s path. Let x be the object’s horizontal image offset from the image centre and s its image height:

$$ q=\frac{x}{s}, \qquad m=q_{\mathrm{now}}+\dot q,\mathrm{TTC}. $$

m is the predicted lateral miss distance in object-height units. In the implemented heuristic, an object is on course when |m| < 1. This assumes the image centre approximately represents the direction of travel and lateral motion can be extrapolated over the prediction horizon.

Schema note: The output column is named miss_widths for historical reasons, but the current implementation measures miss distance in object heights.

Risk rules
Output	Default condition
danger	On course and TTC < 2 s
caution	On course and TTC < 4 s, or approaching with TTC < 1.5 s (close pass)
safe	No caution/danger condition applies
uncertain	A caution/danger condition is indicated, but at least one reliability check failed
Reliability checks flag cases such as warming_up, too_small, noisy_expansion, size_jump, inconsistent_growth, camera_unreliable, cut_top_bottom, and cut_and_turning. The optional warning debounce is disabled by default (debounce=1) because a longer debounce reduced false alarms but also missed hazards in the team’s tests.

Getting started
Requirements
Python 3.12 (the development environment used for this project).
NVIDIA CUDA-capable GPU for SAM inference; the project was tested on an RTX 5060 Laptop GPU (8 GB VRAM) under WSL2 Ubuntu.
A compatible CUDA/PyTorch installation (the repository pins PyTorch 2.10.0 and a CUDA 12.8 wheel for the documented environment).
Access to the required Meta SAM 3 / 3.1 model checkpoints on Hugging Face. Checkpoint access may require approval and authentication.
Sufficient RAM, disk space, and available GPU memory for input frames and SAM tracking state.
[!NOTE] These are the documented development settings, not a verified cross-platform support matrix. Memory usage depends on frame count, number of objects, prompts, and resolution.

1. Clone the repository
git clone https://github.com/spi727/digbodha.git
cd digbodha
 
2. Create a Python environment
conda create -n cvproj python=3.12 -y
conda activate cvproj
python -m pip install --upgrade pip
 
3. Install PyTorch and project dependencies
The repository documents the following PyTorch build:

pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -c constraints.txt
 
The requirements.txt file installs SAM from Meta’s official repository and the project’s additional dependencies. Use an environment-compatible PyTorch/CUDA configuration if reproducing on different hardware; changes are not covered by the reported evaluation.

4. Obtain SAM checkpoints
Request checkpoint access following the official SAM 3 setup instructions, then authenticate your machine as instructed by Meta/Hugging Face; for example:

hf auth login
 
Model weights are not bundled with this repository. The code uses the SAM 3.1 video predictor provided by the installed sam3 package.

5. Analyze an MP4 video
python analyze.py \
  --video path/to/video.mp4 \
  --prompts "vehicle,person" \
  --out outputs/demo
 
analyze.py extracts numbered JPG frames from video inputs before running segmentation. To analyze an existing folder of JPG frames, supply its frame rate:

python analyze.py \
  --video path/to/jpg_frames \
  --fps 10 \
  --prompts "vehicle,person" \
  --out outputs/frame_demo
 
For faster runs without an annotated video, add --no_video.

Main command-line options
Argument	Description	Default
--video	MP4/video file or directory of JPG frames	Required
--prompts	Comma-separated text concepts to track	vehicle,person
--fps	Frame rate; required for a frame folder	Video metadata when available
--out	Output directory	outputs/<video-name>
--max_objects	Maximum objects configured for the predictor	16
--chunk	Video chunk size in frames	60
--overlap	Number of overlapping frames at chunk boundaries	2
--no_video	Skip rendering annotated.mp4	Off
Outputs
A typical run creates:

outputs/demo/
├── frames.csv       # One record per tracked object per frame
├── tracks.json      # Per-track summaries and worst observed risk
├── annotated.mp4    # Visualized masks, IDs, TTC and risk (unless --no_video)
├── run.json         # Run configuration, timing and chunk statistics
└── frames/          # Extracted JPGs when the input is a video file
 
Notable frames.csv columns include:

Field	Meaning
frame, time_s	Frame index and timestamp
id, label	Tracked identity and prompted concept
bx0, by0, bx1, by1	Object bounding box
zone	left, centre, or right
raw_dx, raw_dy	Observed image displacement
own_dx, own_dy	Camera-compensated displacement
motion	still, moving_left, moving_right, or moving
ttc_s, ttc_lo, ttc_hi	Estimated TTC and low/high band in seconds
miss_widths	Predicted miss distance in object heights (legacy column name)
on_course	Whether predicted miss distance is inside the on-course threshold
risk, reliable, reasons	Risk category and reliability diagnostics
Repository structure
digbodha/
├── analyze.py                  # Main inference and output pipeline
├── src/
│   ├── segment.py             # SAM 3.1, chunking, track-ID linking
│   ├── egomotion.py            # Shi–Tomasi, LK optical flow, RANSAC homography
│   └── ttc.py                  # TTC, collision course, risk and reliability
├── eval/
│   ├── evaluate_kitti.py      # Per-sequence KITTI evaluation
│   ├── aggregate.py           # Pooled evaluation metrics
│   ├── analyze_errors.py      # Error attribution and hazard-event metrics
│   ├── rescore.py             # Recompute stage 3 from stored CSV data
│   └── scan_kitti_labels.py   # Find hazard-rich KITTI sequences
├── scripts/
│   ├── run_kitti_eval.sh      # Full KITTI workflow
│   ├── rescore_all.sh         # Batch re-score and evaluate saved runs
│   ├── test_sam3.py           # SAM smoke tests and profiling
│   ├── test_chunked.py        # Chunked tracking checks
│   ├── test_egomotion.py      # Camera-motion checks
│   ├── build_notebook.py      # Notebook generation utility
│   └── make_results_pack.sh   # Result packaging helper
├── Digbodha1.ipynb           # Project notebook
├── requirements.txt
└── constraints.txt
 
Evaluation and results
Benchmark and protocol
We evaluated on the KITTI Object Tracking dataset (10 FPS), which supplies per-frame 3D object annotations. KITTI does not directly provide the TTC labels used here; the evaluator derives reference TTC from object depth Z and its estimated slope:

$$ \mathrm{TTC}_{\mathrm{GT}}=\frac{Z}{-dZ/dt}\quad\text{when the object is approaching}. $$

Predictions are matched to labelled objects by compatible class and 2D bounding-box IoU ≥ 0.5. Evaluation considers labelled objects within 40 m. TTC error is calculated on matched predictions with approaching ground truth and a predicted TTC; detection coverage and alert metrics must be read separately.

Split	KITTI sequences	Role
Development	0000	Debugging
Tuning	0013, 0001, 0020	Choose rules and reliability checks
Held-out	0019, 0007	Evaluation after finalizing the approach
Reported results
Final configuration: height-based TTC with width fallback, no added debounce.

Metric	Tuning	Held-out (pooled)
Median relative TTC error, reliable estimates	12.9%	11.5%
Approach-detection precision	0.97	0.93
Approach-detection recall	0.89	0.90
Alert precision, per matched object-frame	0.66	0.56
Alert recall, per matched object-frame	0.58	0.45
Recall when uncertain also counts as caught	0.81	0.70
Hazard events warned	129 / 149 (87%)	40 / 57 (70%)
False-alarm events	32	24
Median warning lead time	0.9 s	1.2 s
On held-out sequences, TTC estimates flagged unreliable had roughly 60–70% median relative error, versus 11.5% for reliable estimates. The reliability flag therefore identified many difficult measurements, but the final alerting performance remained substantially weaker than TTC estimation on reliable matched objects.

Important interpretation: The 11.5% TTC figure is conditional on estimates being available, matched, and labelled reliable; it is not an end-to-end collision-warning accuracy figure. Similarly, 70% refers to the specific held-out hazard-event protocol, not to deployment safety performance.

Run a KITTI evaluation
Obtain KITTI Tracking images and annotations from the official dataset website under its terms, and arrange them in the locations expected by the scripts. For an already prepared JPG sequence and corresponding labels:

python analyze.py \
  --video data/kitti_0013_340 \
  --fps 10 \
  --out outputs/kitti_0013 \
  --no_video
 
python eval/evaluate_kitti.py \
  --pred outputs/kitti_0013/frames.csv \
  --labels "$HOME/datasets/kitti_tracking/training/label_02/0013.txt" \
  --fps 10 \
  --out outputs/eval_0013
 
To pool evaluation outputs:

python eval/aggregate.py \
  outputs/eval_0013 outputs/eval_0001 outputs/eval_0020 \
  --out outputs/eval_pooled
 
python eval/analyze_errors.py \
  outputs/eval_0013 outputs/eval_0001 outputs/eval_0020
 
A batch runner is also provided:

# Point KITTI at the extracted tracking dataset root and ZIP at the image archive.
export KITTI="$HOME/datasets/kitti_tracking"
export ZIP="/path/to/data_tracking_image_2.zip"
 
bash scripts/run_kitti_eval.sh 0013:340 0001:426 0020:300
 
The batch script uses local dataset paths: edit or set KITTI and ZIP for your machine. KITTI files and SAM checkpoints are downloaded separately; neither dataset nor model weights are supplied with the project.

Re-score TTC and risk without re-running SAM
Saved frames.csv measurements can be reused to test changes to the risk stage:

python eval/rescore.py \
  --pred outputs/kitti_0013/frames.csv \
  --frames data/kitti_0013_340 \
  --fps 10 \
  --out outputs/kitti_0013_rescored/frames.csv \
  --debounce 2
 
This only re-runs the TTC/risk calculations; it does not re-run segmentation or change the underlying tracked objects. This made the team’s ablation experiments much faster.

Ablations and engineering decisions
Design decision	Observation
Fit 1/s instead of log(s)	On one test approach (true TTC 1.50 s), estimated TTC improved from 1.89 s to 1.55 s
Use object height over mask area	Reduced sensitivity to thin masks and changing visible width
Use width fallback for certain top/bottom cuts	Improved tuning-set warned hazards from 127 to 129 of 149 without extra false-alarm events
Enable 2-frame debounce	Reduced tuning false-alarm events from 32 to 21, but warned only 123 instead of 129 hazards
Use chunked SAM inference	Avoided persistent video-tracker memory growth on the tested 8 GB GPU
Runtime
On the reported RTX 5060 Laptop GPU (8 GB), SAM inference was the main bottleneck. The document reports around 2.1 FPS with one prompt, 0.6–1 FPS with two prompts, and approximately 25 ms per frame for the classical ego-motion and TTC stages. These figures are observations from that environment, not general performance guarantees.

Limitations
Offline speed: Full pipeline is not real-time on the tested 8 GB laptop GPU.
Crowded scenes: The default 16-object limit can reduce tracking coverage.
No long-term re-identification: Objects returning after leaving the view receive new IDs.
Approximate camera-motion model: A single homography cannot fully represent 3D parallax.
Simple future-motion assumption: Accelerations, braking, and curved pedestrian trajectories can invalidate extrapolations.
Threshold sensitivity: Small TTC errors near the risk cutoffs create false warnings.
Truncation and segmentation noise: Partial visibility can distort object size and position.
2D geometric output: TTC is in seconds, and lateral miss distance is in object-height units—not metric 3D trajectory reconstruction.
Heading assumption: Image centre is treated as the approximate path direction.
Evaluation domain: Reported experiments use KITTI driving scenes, not pedestrian-mounted cameras or Indian traffic environments.
Future directions
Potential extensions, not implemented or validated in the reported results, include:

Substitute a faster open-vocabulary segmenter/tracker, such as a suitable YOLOE variant, for SAM 3.1.
Improve tracking capacity and re-identification in crowded scenes.
Replace straight-line motion extrapolation with models that handle changes in velocity and direction.
Replace rigid warning thresholds with better-calibrated risk estimates.
Evaluate on pedestrian viewpoints, different camera motions, and more diverse road environments.
Explore structured object-motion annotations for uncertainty-aware vision-language assistance.
Authors
K. Praveen Sagar · CS26M1002
Arnav Abhijit Rajurkar · CS26M1007
Madhur Vijay Patil · CS26M1024

M.Tech CSE (Data Science & AI)
Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram

Computer Vision course project · November 2026

References
The project builds upon the following published methods and upstream resources:

N. Carion et al., “SAM 3: Segment Anything with Concepts”, 2025. arXiv:2511.16719; Meta SAM 3 repository.
A. Geiger, P. Lenz, and R. Urtasun, “Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite”, CVPR, 2012. KITTI.
D. N. Lee, “A Theory of Visual Control of Braking Based on Information about Time-to-Collision”, Perception, 1976.
B. D. Lucas and T. Kanade, “An Iterative Image Registration Technique with an Application to Stereo Vision”, IJCAI, 1981.
J. Shi and C. Tomasi, “Good Features to Track”, CVPR, 1994.
C. Harris and M. Stephens, “A Combined Corner and 
