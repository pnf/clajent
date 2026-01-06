#!/usr/bin/env lein exec

(ns extract-test
  (:require [clajent.image-timeseries :as img-ts]
            [clajent.timeseries-plot :as plot])
  (:import [java.io File]))

(println "=== Image-to-Timeseries Extraction Test ===\n")

(def input-path "resources/test-images/hand-drawn-graphs.png")
(def output-path "resources/test-images/extracted-graphs.png")

(println "Input image:" input-path)
(println "Output will be saved to:" output-path)
(println)

;; Extract series
(println "Extracting time series from image...")
(def series
  (img-ts/extract-timeseries-from-image
    input-path
    "2025-01-01T00:00:00Z"
    "2025-01-01T23:59:59Z"
    100
    {:dot-threshold 120
     :line-threshold 160
     :min-dot-size 15}))

(println "\n=== Extraction Results ===")
(println "Number of series extracted:" (count series))
(println)

;; Print statistics for each series
(doseq [[idx s] (map-indexed vector series)]
  (let [values (:values s)
        non-nil-values (filter some? values)]
    (println (format "Series %d:" (inc idx)))
    (println (format "  Total points: %d" (count values)))
    (println (format "  Valid points: %d" (count non-nil-values)))
    (when (seq non-nil-values)
      (println (format "  Min value: %.3f" (apply min non-nil-values)))
      (println (format "  Max value: %.3f" (apply max non-nil-values)))
      (println (format "  Mean value: %.3f"
                      (/ (reduce + non-nil-values)
                         (count non-nil-values)))))
    (println)))

;; Create visualization if we have series
(when (seq series)
  (println "Creating visualization...")
  (let [timestamps (:timestamps (first series))
        series-names (map-indexed (fn [i _] (str "Series " (inc i))) series)
        values-list (map (fn [s]
                          (map #(if % % Double/NaN) (:values s)))
                        series)]

    (plot/plot-timeseries timestamps series-names values-list
      {:output {:file output-path}
       :width 1600
       :height 1000
       :combined false})

    (println "✓ Visualization saved to:" output-path)))

(println "\n=== Test Complete ===")
(System/exit 0)
