# Formats guide

This guide describes everything that goes into the eye tracker and everything that comes out of it: the screen model, the log of a session recorded on a website, the gaze and fixation files, and the areas of interest. All files are CSV (comma separator, decimal point, one header row) or JSON (UTF-8), so they open in a spreadsheet, pandas, R or a web page without any extra tool.

## 1. From gaze direction to pixel

The calibrated model knows nothing about the screen. It predicts a gaze direction g in the webcam's frame, and the screen model turns it into pixels. You can therefore change screen, resolution or browser window without recalibrating, as long as the webcam stays the same and stays in the plane of the screen.

Notation. o is the midpoint of the two eyes in the webcam frame (mm; x towards the user's left, y up, z towards the user), g the gaze direction, (c_x, c_y) the position of the lens in the screen frame (mm from the top-left corner of the picture; X to the right, Y down), ρ the size of one CSS pixel in mm.

```
h = o + s·g                                    (gaze ray, s ≥ 0)
h_z = 0  ⇔  o_z + s·g_z = 0                    (the screen is the plane z = 0 of the webcam)
         ⇔  s = −o_z / g_z                      (g_z ≠ 0: the person looks towards the screen)
P(X, Y) = (−(X − c_x), −(Y − c_y), 0)           (change of frame screen → webcam, no mirroring)
h = P(X, Y)  ⇔  X = c_x − h_x  and  Y = c_y − h_y   (component-wise identification)
ρ = width_mm / width_px                         (square pixels)
u = X / ρ,  v = Y / ρ                           (CSS pixel on the screen)
```

Going to the browser window and then to the page subtracts the position of the viewport on the screen (w_x, w_y) and adds the scroll offset (d_x, d_y):

```
u_window = u − w_x,          v_window = v − w_y          (translation)
u_page   = u_window + d_x,   v_page   = v_window + d_y   (translation)
```

Finally, inside an element at (x_e, y_e) of width l_e and height h_e, displaying an image of n_w × n_h pixels:

```
u_element = (u_window − x_e) · n_w / l_e,   v_element = (v_window − y_e) · n_h / h_e   (scaling)
```

A useful order of magnitude for sizing areas: an angular error δ at a distance D is D·tan δ on the screen, i.e. D·tan δ / ρ pixels. At 50 cm, 0.9° gives 0.79 cm, i.e. 39 CSS pixels on a 14-inch MacBook Pro (ρ = 0.2 mm). A reliable area of interest is at least two to three times that error in each direction, and the `margin_px` field of an areas file enlarges every area by that amount.

## 2. Screen model: `screen.json`

```json
{
 "name": "MacBook Pro 14-inch",
 "width_px": 1512, "height_px": 982,
 "width_mm": 302.4, "height_mm": 196.4,
 "camera_x_mm": 151.2, "camera_y_mm": 3.4
}
```

`width_px` and `height_px` are the resolution in CSS pixels, i.e. the browser's `screen.width` and `screen.height` (1512 × 982 for a 14-inch MacBook Pro at its default scaling, not 3024 × 1964). `width_mm` and `height_mm` are the physical size of the displayed picture. `camera_x_mm` and `camera_y_mm` locate the centre of the lens from the top-left corner of the picture: 3.4 mm below the edge for a webcam in the notch, a negative value for a webcam above the picture, about −12 mm for a webcam clipped on top of an external monitor.

There are three ways to get one. The calibration page offers the models listed in `calibration/screens.json` (recognised from the resolution) or any screen described by its diagonal, and `calibrate.py` writes the result next to the model, in `models/<name>_screen.json`. It can also be written from the command line:

```bash
python code/screen.py --list
python code/screen.py my_screen.json --preset "Pro 16" --resolution 1728 1117
python code/screen.py my_screen.json --diagonal 27 --resolution 2560 1440 --camera-mm 298.9 -15
```

or edited by hand. `track.py` uses the one stored with the model, or another one given with `--screen-model my_screen.json`.

## 3. Session log: `session_*.json` (format `gaze-session-1`)

Written by `web/gaze-recorder.js` next to the video. Every time `t` is in milliseconds since the start of the video recording.

| Field | Content |
|---|---|
| `url`, `created`, `user_agent` | page where the recording started, date, browser |
| `screen` | `width`, `height`, `device_pixel_ratio` |
| `video` | MIME type, width, height and frame rate of the webcam |
| `flashes` | `[{label, t_start, t_end}]`, white synchronisation flashes at the start and the end |
| `check` | `[{t_start, t_end, x, y}]`, the 9 check points, in window pixels |
| `snapshots` | page state at every change: `t`, `url`, `scroll_x`, `scroll_y`, `viewport` (size and position of the window, full screen or not), `aoi` (visible elements `{name, x, y, w, h}` in window pixels, plus `natural_w`, `natural_h` for an image) |
| `events` | `click` (`t`, `x`, `y` in the window, `element`) and `mark` (`t`, `label`, set by `GazeRecorder.mark`) |

No keystroke and no form content is recorded.

Element names come from the `data-gaze-aoi` attribute, or, in automatic mode (`data-auto="on"`), from the element's id or its tag and text (for example `button: Add to cart`). Elements sharing a name are numbered in document order (`add (2)`).

## 4. Outputs of `track.py`

A folder `results/tracking/<video name>/` contains the following files.

`gaze.csv`, one row per webcam frame:

| Column | Content |
|---|---|
| `t_ms` | time in ms, same clock as the session log |
| `frame` | frame number |
| `status` | `ok`, `blink` or `no_face` |
| `phase` | `check` during the check points, `session` afterwards |
| `x_px`, `y_px` | gaze point, CSS pixels from the top-left corner of the screen |
| `x_mm`, `y_mm` | the same point in millimetres |
| `on_screen` | 1 if the point is on the screen |
| `fixation_id` | id of the current fixation (or smooth pursuit), −1 otherwise |
| `yaw_deg`, `pitch_deg` | gaze direction (Fick angles, webcam frame, filtered) |
| `head_distance_cm` | distance from the head to the webcam |
| `client_x`, `client_y` | with `--session`: point in the browser window |
| `page_x`, `page_y` | with `--session`: point in the page, scrolling included |
| `element`, `element_x`, `element_y` | with `--session`: smallest recorded element under the gaze, and position inside it (in pixels of the original image for an image) |
| `url` | with `--session`: page being displayed |

`fixations.csv`, one row per fixation or smooth pursuit: `id`, `kind` (`fixation` or `pursuit`), `start_ms`, `end_ms`, `duration_ms`, position `x_px`, `y_px`, `x_mm`, `y_mm`, direction `yaw_deg`, `pitch_deg`, amplitude of the saccade that led to it `saccade_deg`, and with `--session` the same `client_*`, `page_*`, `element*` and `url` columns as above.

`aoi_elements.csv` (for a session whose page marks elements) and `aoi_<name>.csv` (for each `--aoi <name>.json`), one row per area plus a `(no area)` row:

| Column | Content |
|---|---|
| `fixations` | number of fixations in the area |
| `dwell_ms`, `dwell_share` | total fixation time and share of all fixation time (%) |
| `time_to_first_fixation_ms` | delay before the first fixation, counted from the end of the check points |
| `first_fixation_ms`, `mean_fixation_ms` | duration of the first fixation, mean duration |
| `visits`, `revisits` | entries into the area (runs of consecutive fixations), and returns |
| `gaze_ms` | time spent, counted frame by frame rather than by fixation |
| `clicks`, `time_to_first_click_ms` | clicks in the area and delay before the first one |

`gaze.json` gathers everything: `meta` (screen, synchronisation offset, recalibration and measured error, clicks), `fixations`, `aoi` (the reports above) and `frames` (the columns of `gaze.csv` plus the head–eye decomposition and the complex-plane coordinates of the gaze). Missing values are `null` in JSON and empty in CSV.

`heatmap_<space>.png`: heatmap of fixations weighted by their duration, with the numbered scanpath, on the screen or, with `--heatmap-space page --background screenshot.png`, on a screenshot of the whole page.

## 5. Areas of interest: `areas.json`

```json
{
 "space": "screen",
 "margin_px": 20,
 "areas": [
  {"name": "buy button", "x": 1200, "y": 640, "w": 220, "h": 70},
  {"name": "logo", "cx": 90, "cy": 40, "r": 50},
  {"name": "face", "polygon": [[400, 120], [620, 110], [650, 420], [380, 430]]}
 ]
}
```

An area is a rectangle (`x`, `y`, `w`, `h`), a circle (`cx`, `cy`, `r`) or a polygon. When areas overlap, a point counts for the smallest one. `space` sets the coordinate system: `screen`, `client` (browser window), `page` (whole page, scrolling included) or the name of a recorded element, for example `image` for the objects of a photo shown by `web/image_test.html`, in pixels of the original photo. An areas file can be applied during tracking (`track.py ... --aoi areas.json`) or afterwards on a folder that has already been produced (`python code/aoi.py results/tracking/<name> areas.json`).
