# Image-to-Timeseries Extraction Test

This directory contains test images and results for the image-to-timeseries extraction utility.

## Test Files

- `hand-drawn-graphs.png` - Input image containing hand-drawn time-series graphs bounded by dots
- `extracted-graphs.png` - Output visualization showing the extracted time-series data

## Running the Test

### Option 1: Using the test namespace

```clojure
lein repl
```

Then in the REPL:

```clojure
(require '[clajent.image-timeseries-test :as test])
(test/run-test)
```

### Option 2: Using the standalone script

From the project root:

```bash
lein repl < test_extract.clj
```

### Option 3: Custom extraction

```clojure
(require '[clajent.image-timeseries :as img-ts])
(require '[clajent.timeseries-plot :as plot])

;; Extract series
(def series
  (img-ts/extract-timeseries-from-image
    "resources/test-images/hand-drawn-graphs.png"
    "2025-01-01T00:00:00Z"
    "2025-01-01T23:59:59Z"
    100
    {:dot-threshold 120
     :line-threshold 160
     :min-dot-size 15}))

;; Visualize
(let [timestamps (:timestamps (first series))
      series-names (map-indexed (fn [i _] (str "Series " (inc i))) series)
      values-list (map (fn [s] (map #(if % % Double/NaN) (:values s))) series)]
  (plot/plot-timeseries timestamps series-names values-list
    {:output {:file "resources/test-images/extracted-graphs.png"}
     :width 1600
     :height 1000
     :combined false}))
```

## Expected Output

The extraction should:
1. Detect dark clusters (dots and lines) in the image
2. Group dots into rectangular regions (4 corners each)
3. Sample each line at 100 evenly-spaced points
4. Normalize Y values from 0 (bottom) to 1 (top)
5. Generate timestamps from start to end time
6. Create a visualization with each series in its own subplot

## Tuning Parameters

If extraction doesn't work well, try adjusting:

- `:dot-threshold` (default 120) - Lower values detect lighter dots
- `:line-threshold` (default 160) - Lower values detect lighter lines
- `:min-dot-size` (default 15) - Minimum pixels for a dot cluster

## Troubleshooting

**No regions detected:**
- Check that dots are visible and dark enough
- Lower the `:dot-threshold` value
- Ensure dots are large enough (increase `:min-dot-size` if too sensitive)

**Lines not extracted correctly:**
- Lower the `:line-threshold` value
- Ensure lines are continuous and dark

**Wrong number of series:**
- Verify each graph has exactly 4 corner dots
- Check that dots form reasonable rectangles
