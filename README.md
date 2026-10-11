# Detector shootout on an F1 photo

A beginner computer-vision project: run ORB and SIFT on one photo, then rotate, scale, blur and darken the photo and see how many feature matches each detector keeps. The result is a head-to-head scoreboard: who wins each condition.

AKAZE and BRISK were in the original plan but are left out: in OpenCV 5 they are only in the `opencv-contrib-python` build (under `cv2.xfeatures2d`). They can be added later without changing the structure.

**What you learn:** what keypoints and descriptors are, how matching works, and how invariant each detector is to rotation, scale, blur and brightness.

This file is a guide, not a solution. Each step has a concept, a task, hints, and questions to answer from your own output.

## Setup

- Python 3 with `opencv-python` (written against 5.0.0) and optionally `matplotlib`
- One high-resolution photo saved as `f1.jpg`
- Run with `python3 main.py`

## Roadmap

| Step | Goal | Status |
|------|------|--------|
| 1 | Detect and draw keypoints with ORB | done |
| 2 | Run both detectors and compare | done |
| 3 | Compute descriptors and match an image against itself | in progress |
| 4 | Build the transformations | |
| 5 | Match original against transformed and count survivors | |
| 6 | Build the scoreboard | |

---

## Step 1: keypoints with ORB

**Concept.** A *keypoint* is a spot that is distinctive enough to find again in another photo of the same scene: corners, blobs, sharp texture. A keypoint is more than a position; it also has a size and an orientation.

**Task.** Detect ORB keypoints on the image and save a copy with them drawn on.

**Hints.**
- `cv2.imread` returns `None` on a bad path instead of raising, so check with `if img is None:`.
- Detectors work on intensity, so load or convert to grayscale.
- Create the detector with `cv2.ORB_create()` and call `.detect()` on the gray image.
- Use `cv2.drawKeypoints` with `flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`.
- Save with `cv2.imwrite`; a high-res image is easier to inspect in the editor than in an `imshow` window.

**Questions.**
- Why is the keypoint count exactly 500?
- What do `.pt`, `.size`, `.angle`, `.response` and `.octave` mean?
- What do the circle radius and the line inside each circle show?
- Where do keypoints cluster, and where are there none? Why are long straight edges mostly empty?

---

## Step 2: ORB against SIFT

**Concept.** The two detectors look for different things. ORB looks for corners using fast intensity tests on a small image pyramid. SIFT looks for blob-like structures across a *scale space* (the image at many levels of blur and size). They share one interface in OpenCV, so swapping them is cheap.

**Task.** Run ORB and SIFT on the same image. Save one output image per detector and print each keypoint count and detection time.

**Hints.**
- The factories are `cv2.ORB_create()` and `cv2.SIFT_create()`.
- Pass the detector into your detection function instead of creating ORB inside it.
- A dictionary mapping name to detector, looped over, keeps `main` short. You will reuse it in every later step.
- Time the `.detect()` call with `time.perf_counter()`.

**Questions.**
- How do the keypoint counts compare, and why is ORB's so much lower?
- Which detector is faster, and by what factor?
- Where does SIFT place keypoints that ORB did not?
- How do the circle sizes differ? ORB had only 8 distinct sizes; how many does SIFT have, and what is its smallest?

---

## Step 3: descriptors and matching

**Concept.** A keypoint says *where* something is. A *descriptor* says *what it looks like*: a vector of numbers summarising the patch around the keypoint, computed relative to the keypoint's size and angle. Two keypoints match when their descriptors are close. There are two families:

- **Float descriptors** (SIFT): compared with Euclidean distance.
- **Binary descriptors** (ORB): strings of bits, compared with Hamming distance (the number of differing bits).

**Task.** Compute descriptors for each detector, then match the image against itself as a sanity check.

**Hints.**
- `.detectAndCompute(img, None)` returns keypoints and descriptors together.
- Print each descriptor array's `.shape` and `.dtype`. The row count should equal the keypoint count.
- Create a matcher with `cv2.BFMatcher(normType, crossCheck=True)`. The norm must suit the descriptor: `cv2.NORM_L2` for SIFT, `cv2.NORM_HAMMING` for ORB. Store the right norm next to each detector in your dictionary.
- `.match(des1, des2)` returns `DMatch` objects. Look at `.queryIdx`, `.trainIdx` and `.distance`.
- `cv2.drawMatches` draws two images side by side with lines between matched keypoints. Draw only the best 50 or the picture is unreadable.

**Questions.**
- How long is each detector's descriptor, and what type? How many bytes does one keypoint cost in each?
- When matching the image against itself, what should every match distance be? Is it?
- What does `crossCheck=True` do, and what happens to the match count if you turn it off?
- What happens if you match SIFT descriptors with `NORM_HAMMING`?

---

## Step 4: the transformations

**Concept.** A detector is *invariant* to a change if it finds the same keypoints, with similar descriptors, after the change. To test that, you need controlled changes where you know exactly what was done to the image.

**Task.** Write one function per transformation. Each takes an image and a strength parameter and returns the transformed image. Save an example of each to check it by eye.

| Condition | Function to look up | Parameter |
|-----------|--------------------|-----------|
| Rotation | `cv2.getRotationMatrix2D` + `cv2.warpAffine` | angle in degrees |
| Scaling | `cv2.resize` | scale factor |
| Blur ("rain spray") | `cv2.GaussianBlur` | kernel size |
| Brightness ("night race") | `cv2.convertScaleAbs` | `alpha` (contrast), `beta` (offset) |

**Hints.**
- Rotate around the image centre. The corners get cut off; that is fine for now, but note it.
- For rotation and scaling, return the 2x3 transformation matrix as well as the image. You need it in step 5. For scaling you can build that matrix by hand.
- Gaussian kernel sizes must be odd.
- For shrinking, compare `interpolation=cv2.INTER_AREA` against the default.
- Pick a few strengths per condition, for example angles of 15, 45, 90 and 180 degrees.

**Questions.**
- After a 45 degree rotation, what fraction of the original image is still visible?
- Why does `convertScaleAbs` exist, instead of just multiplying the array by a number?
- Which transformations move pixels, and which only change their values? Why does that distinction matter for step 5?

---

## Step 5: count the survivors

**Concept.** Raw match counts lie: a brute-force matcher always returns its nearest neighbour, even when that neighbour is wrong. You need a way to tell good matches from bad ones. There are two, and you should build both:

1. **Lowe's ratio test.** For each descriptor, get the two nearest neighbours. Keep the match only if the best is clearly closer than the second best. An ambiguous match is probably a wrong one.
2. **Geometric check.** You know the transformation, so you know where each original keypoint *should* land. A match is correct only if it lands within a few pixels of that spot.

**Task.** For every detector and every condition, match the original against the transformed image and count the matches that survive.

**Hints.**
- For the ratio test use `matcher.knnMatch(des1, des2, k=2)` with `crossCheck=False`, and keep matches where `m.distance < 0.75 * n.distance`.
- For the geometric check, apply the 2x3 matrix from step 4 to the original keypoint's `.pt` and measure the distance to the matched keypoint's `.pt`. For blur and brightness the matrix is the identity.
- Guard against `descriptors` being `None`, and against `knnMatch` returning fewer than two neighbours. Both happen on heavily degraded images.
- ORB is capped at 500 keypoints while SIFT finds far more, so raw counts are unfair. Report a ratio too, such as correct matches divided by keypoints in the original.

**Questions.**
- How many ratio-test matches fail the geometric check? Which detector produces the most false matches?
- Which detector loses more under scaling? Under rotation? Under blur? Can you explain it from how the detector works?
- Does rotation by 90 degrees behave differently from 45 degrees? Why might it?
- Does raising ORB's `nfeatures` change its ranking?

---

## Step 6: the scoreboard

**Task.** Print the winner of each condition with both scores and the gap between them, then an overall head-to-head tally.

**Hints.**
- Collect results into a nested dictionary: condition, then detector, then score.
- `max(scores, key=scores.get)` gives the winner of one condition. Using `sorted(..., reverse=True)` instead keeps the code working if you add more detectors later.
- Decide which score ranks the detectors, raw count or ratio, and state it in the output.
- Count a win per condition and strength (for example each rotation angle separately), and print the final tally like a qualifying head-to-head: "SIFT 9 - 3 ORB".

**Extensions.**
- Plot score against strength for one condition, such as matches against rotation angle from 0 to 360 degrees, one line per detector.
- Add speed as a column: matches per millisecond.
- Add AKAZE and BRISK via `opencv-contrib-python`.
- Add a condition of your own: JPEG compression, Gaussian noise, or a perspective warp with `cv2.warpPerspective`.
- Swap the brute-force matcher for `cv2.FlannBasedMatcher` and compare the time.

**Final questions.**
- If you had to pick one detector for a real-time robot with a weak CPU, which would it be, and what do you give up?
- Which would you pick for offline 3D reconstruction, where accuracy matters more than speed?
