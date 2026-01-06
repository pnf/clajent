(ns clajent.image-timeseries-test
  "Test and demonstration of image-to-timeseries extraction"
  (:require [clajent.image-timeseries :as img-ts]
            [clajent.timeseries-plot :as plot]))

(defn extract-and-plot
  "Extract time series from an image and create a visualization.

  Arguments:
  - image-path: Path to the input image
  - output-path: Path for the output PNG
  - start-time: Start time for the series
  - end-time: End time for the series
  - n-points: Number of points to sample
  - options: Optional extraction parameters"
  ([image-path output-path start-time end-time n-points]
   (extract-and-plot image-path output-path start-time end-time n-points {}))

  ([image-path output-path start-time end-time n-points options]
   (println "=== Extracting Time Series from Image ===")
   (println "Input image:" image-path)
   (println "Output file:" output-path)
   (println "Time range:" start-time "to" end-time)
   (println "Sample points:" n-points)
   (println)

   ;; Extract series from image
   (println "Extracting series...")
   (let [series (img-ts/extract-timeseries-from-image
                  image-path start-time end-time n-points options)]

     (println "Extracted" (count series) "series")
     (println)

     ;; Print summary statistics for each series
     (doseq [[idx s] (map-indexed vector series)]
       (let [values (:values s)
             non-nil-values (filter some? values)
             nil-count (- (count values) (count non-nil-values))]
         (println (format "Series %d:" (inc idx)))
         (println (format "  Total points: %d" (count values)))
         (println (format "  Valid points: %d" (count non-nil-values)))
         (println (format "  Missing points: %d" nil-count))
         (when (seq non-nil-values)
           (println (format "  Min value: %.3f" (apply min non-nil-values)))
           (println (format "  Max value: %.3f" (apply max non-nil-values)))
           (println (format "  Mean value: %.3f"
                           (/ (reduce + non-nil-values)
                              (count non-nil-values)))))
         (println)))

     ;; Create visualization if we have any series
     (when (seq series)
       (println "Creating visualization...")
       (let [timestamps (:timestamps (first series))
             series-names (map-indexed (fn [i _] (str "Series " (inc i))) series)
             ;; Replace nil values with NaN for plotting
             values-list (map (fn [s]
                               (map #(if % % Double/NaN) (:values s)))
                             series)]

         ;; Plot all series together
         (plot/plot-timeseries timestamps series-names values-list
           {:output {:file output-path}
            :width 1600
            :height 1000
            :combined false})  ; Each series gets its own subplot

         (println "Visualization saved to:" output-path)
         (println)
         (println "=== Extraction Complete ===")

         ;; Return the series for further analysis
         series)))))

(defn run-test
  "Run the extraction test on the uploaded graph image.
  This function demonstrates the full workflow from image to visualization."
  []
  (let [input-image "resources/test-images/hand-drawn-graphs.jpg"
        output-image "resources/test-images/extracted-graphs.png"
        start-time "2025-01-01T00:00:00Z"
        end-time "2025-01-01T23:59:59Z"
        n-points 100]

    (extract-and-plot input-image output-image start-time end-time n-points
      {:dot-threshold 120        ; Adjust based on image brightness
       :line-threshold 160       ; Adjust for line detection
       :min-dot-size 15})))      ; Minimum size for corner dots

(comment
  ;; Run the test extraction
  (run-test)

  ;; Or extract with custom parameters
  (extract-and-plot
    "resources/test-images/hand-drawn-graphs.jpg"
    "resources/test-images/extracted-graphs-custom.png"
    "2025-01-01T00:00:00Z"
    "2025-01-02T00:00:00Z"
    200
    {:dot-threshold 100
     :line-threshold 150
     :min-dot-size 10}))
