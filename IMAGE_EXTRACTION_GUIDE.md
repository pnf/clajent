# Image-to-Timeseries Extraction Guide

This utility extracts time-series data from images containing hand-drawn graphs bounded by dots.

## Overview

The `clajent.image-timeseries` namespace provides functionality to:
1. Load images containing multiple time-series graphs
2. Detect bounding dots that mark each graph region
3. Extract line values at evenly-spaced intervals
4. Normalize Y-axis values from 0 to 1
5. Generate timestamps and return properly formatted time-series data

## Usage

### Basic Example

```clojure
(require '[clajent.image-timeseries :as img-ts])

;; Extract time series from an image
(def series
  (img-ts/extract-timeseries-from-image
    "/path/to/graph-image.jpg"
    "2025-01-01T00:00:00Z"   ; start time
    "2025-01-01T23:59:59Z"   ; end time
    100))                     ; number of points

;; The result is a list of maps, each with :timestamps and :values
;; [{:timestamps ["2025-01-01T00:00:00Z" ...] :values [0.5 0.6 ...]}
;;  {:timestamps ["2025-01-01T00:00:00Z" ...] :values [0.3 0.4 ...]}]
```

### Using with Timeseries Analysis

The extracted series can be directly used with the timeseries analysis functions:

```clojure
(require '[clajent.timeseries :as ts])

;; Get statistics for each extracted series
(doseq [s series]
  (println "Mean value:"
    (ts/get-descriptive-statistics
      {:series s
       :statistic-name "mean"
       :col-name "target_col"})))

;; Check for outliers
(doseq [s series]
  (println "Number of outliers:"
    (ts/get-number-of-outliers {:series s})))

;; Get moving average
(doseq [s series]
  (println "Moving average:"
    (ts/get-moving-average {:series s :window-size 5})))
```

### Plotting the Extracted Series

You can visualize the extracted series using the timeseries plotting functions:

```clojure
(require '[clajent.timeseries-plot :as plot])

;; Extract timestamps and values from all series
(let [timestamps (:timestamps (first series))
      series-names (map-indexed (fn [i _] (str "Series " (inc i))) series)
      values-list (map :values series)]

  ;; Plot all series together
  (plot/plot-timeseries timestamps series-names values-list
    {:output :swing
     :width 1200
     :height 800
     :combined true}))
```

## Image Requirements

For best results, your images should:

1. **Have clear bounding dots**: Each graph should be bounded by 4 dots marking the corners
   - Top-left, top-right, bottom-left, bottom-right

2. **Use dark lines on light background**: The utility detects dark pixels (lines and dots)

3. **Provide sufficient contrast**: Lines should be clearly visible against the background

4. **Keep graphs separated**: Dots should clearly define distinct rectangular regions

## Advanced Options

You can customize the detection parameters:

```clojure
(img-ts/extract-timeseries-from-image
  "/path/to/image.jpg"
  "2025-01-01T00:00:00Z"
  "2025-01-02T00:00:00Z"
  200
  {:dot-threshold 100      ; brightness threshold for dots (lower = darker)
   :line-threshold 150     ; brightness threshold for lines
   :min-dot-size 10})      ; minimum pixels for a dot cluster
```

## Time Format Support

The start and end times support multiple formats (via `timeseries-plot/parse-time`):
- Java `Date` objects
- Java `Instant` objects
- Java `LocalDateTime` objects
- Long milliseconds since epoch
- ISO-8601 strings (e.g., "2025-01-01T00:00:00Z")

## Return Format

Each extracted series is a map with:
```clojure
{:timestamps [t1 t2 t3 ...]   ; ISO-8601 strings
 :values [v1 v2 v3 ...]}      ; Normalized 0.0-1.0 values
```

This format is compatible with all functions in `clajent.timeseries`.

## Troubleshooting

**No regions detected:**
- Check that your image has visible dots at the corners of each graph
- Try adjusting `:dot-threshold` (lower values detect lighter dots)
- Ensure dots are large enough (adjust `:min-dot-size`)

**Lines not extracted correctly:**
- Adjust `:line-threshold` (lower values detect lighter lines)
- Ensure lines are continuous and clear in the image
- Check that the Y-axis normalization makes sense for your data

**Wrong number of series:**
- Verify that each graph has exactly 4 corner dots
- Check that dots are aligned in reasonable rectangles
- Try manually inspecting which dots are detected
