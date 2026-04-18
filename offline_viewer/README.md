# UniT Offline Point Cloud Viewer

A standalone MeshLab-style viewer for tuning the parameters used on the
UniT project page. Everything runs in one browser tab — no server required
beyond a static file host.

## Running it

The viewer needs to be served over `http(s)://`, because browsers block
`DecompressionStream`, file reads, and WebGL shader compilation on
`file://` URLs in some configurations.

Any of these works:

```sh
# From webpage/offline_viewer
python3 -m http.server 8080
# Then visit http://localhost:8080/
```

```sh
# Or serve the whole webpage/ so you can drop .pnt.gz files from assets/demos/
cd webpage
python3 -m http.server 8080
# Then visit http://localhost:8080/offline_viewer/
```

## What it does

- **Open PLY…** or drag-drop: loads a binary_little_endian `.ply` or a
  `.pnt.gz` (the format produced by `sparsify_ply.py`).
- **Display** controls: point size, sampling rate (randomly-shuffled
  prefix — fast), brightness multiplier, ambient-boost floor, opacity,
  and background colour.
- **Orientation**: pitch / yaw / roll sliders and a flip-Y toggle, so
  crooked captures can be straightened interactively.
- **Camera**: numeric X/Y/Z unit offsets from the bounding-sphere centre
  plus an FOV slider. _Use current_ reads back the live orbit camera so
  you can spin the scene to the best angle and bake it.
- **Copy Config**: dumps a JSON-ish block that drops straight into the
  `DEMO_CONFIGS` map in `webpage/script.js`.

## Workflow

1. Serve the webpage directory (see above).
2. Open the offline viewer, drag a new capture in (or pick a `.pnt.gz`
   from `webpage/assets/demos/<scene>/`).
3. Tweak sliders until the scene reads well. Spin to your preferred
   entry angle and click _Use current_ to bake the camera.
4. Click _Copy Config_ and paste the snippet into `DEMO_CONFIGS` in
   `script.js` (replacing the matching scene's block).
5. Reload the main page; the tweaks are live.
