(ns clajent.image-timeseries
  "Extract time-series data from images containing hand-drawn graphs.
  Each graph should be bounded by four dots marking the corners."
  (:import [java.awt.image BufferedImage]
           [javax.imageio ImageIO]
           [java.io File]
           [java.time Instant]
           [java.time.temporal ChronoUnit])
  (:require [clajent.timeseries-plot :as plot]))

;; ============================================================================
;; Image Processing Utilities
;; ============================================================================

(defn- load-image
  "Load an image from a file path."
  [^String file-path]
  (ImageIO/read (File. file-path)))

(defn- pixel-brightness
  "Calculate brightness of a pixel (0 = black, 255 = white)."
  [^BufferedImage img x y]
  (let [rgb (.getRGB img x y)
        r (bit-and (bit-shift-right rgb 16) 0xFF)
        g (bit-and (bit-shift-right rgb 8) 0xFF)
        b (bit-and rgb 0xFF)]
    (/ (+ r g b) 3.0)))

(defn- is-dark-pixel?
  "Check if a pixel is dark (below threshold)."
  [^BufferedImage img x y threshold]
  (< (pixel-brightness img x y) threshold))

(defn- flood-fill
  "Flood fill starting from a point, finding connected dark pixels."
  [^BufferedImage img start-x start-y threshold visited width height]
  (loop [queue [[start-x start-y]]
         cluster #{}]
    (if (empty? queue)
      cluster
      (let [[x y] (first queue)
            rest-queue (rest queue)]
        (if (or (contains? @visited [x y])
                (< x 0) (>= x width)
                (< y 0) (>= y height)
                (not (is-dark-pixel? img x y threshold)))
          (recur rest-queue cluster)
          (do
            (swap! visited conj [x y])
            (recur (concat rest-queue
                          [[(dec x) y] [(inc x) y]
                           [x (dec y)] [x (inc y)]])
                   (conj cluster [x y]))))))))

(defn- find-dark-clusters
  "Find clusters of dark pixels in the image. Returns list of {:x :y :size} maps."
  [^BufferedImage img threshold min-size]
  (let [width (.getWidth img)
        height (.getHeight img)
        visited (atom #{})
        clusters (atom [])]

    ;; Scan image for dark pixels and cluster them
    (doseq [y (range height)
            x (range width)]
      (when (and (is-dark-pixel? img x y threshold)
                 (not (contains? @visited [x y])))
        (let [cluster (flood-fill img x y threshold visited width height)]
          (when (>= (count cluster) min-size)
            (let [xs (map first cluster)
                  ys (map second cluster)
                  center-x (/ (+ (apply min xs) (apply max xs)) 2)
                  center-y (/ (+ (apply min ys) (apply max ys)) 2)]
              (swap! clusters conj {:x center-x
                                   :y center-y
                                   :size (count cluster)}))))))
    @clusters))

(defn- distance
  "Calculate Euclidean distance between two points."
  [p1 p2]
  (Math/sqrt (+ (Math/pow (- (:x p1) (:x p2)) 2)
                (Math/pow (- (:y p1) (:y p2)) 2))))

(defn- dots-form-rectangle?
  "Check if 4 dots form a reasonable rectangle."
  [d1 d2 d3 d4 tolerance]
  (let [;; Sort by Y first, then X to get: top-left, top-right, bottom-left, bottom-right
        sorted (sort-by (juxt :y :x) [d1 d2 d3 d4])
        [tl tr bl br] sorted

        ;; Check if top two dots are at similar Y
        top-y-diff (Math/abs (double (- (:y tl) (:y tr))))
        ;; Check if bottom two dots are at similar Y
        bottom-y-diff (Math/abs (double (- (:y bl) (:y br))))
        ;; Check if left two dots are at similar X
        left-x-diff (Math/abs (double (- (:x tl) (:x bl))))
        ;; Check if right two dots are at similar X
        right-x-diff (Math/abs (double (- (:x tr) (:x br))))

        ;; Check if it forms a reasonable sized rectangle
        width (- (max (:x tr) (:x br)) (min (:x tl) (:x bl)))
        height (- (max (:y bl) (:y br)) (min (:y tl) (:y tr)))]

    (and (< top-y-diff tolerance)
         (< bottom-y-diff tolerance)
         (< left-x-diff tolerance)
         (< right-x-diff tolerance)
         (> width 50)
         (> height 30))))

(defn- find-rectangular-regions
  "Find rectangular regions bounded by 4 dots in the image.
  Uses optimized spatial approach to find rectangles efficiently."
  [dots]
  (let [;; Filter to keep only larger clusters (likely corner dots)
        ;; Take top 20% by size to reduce search space more aggressively
        sorted-dots (reverse (sort-by :size dots))
        threshold-size (if (> (count sorted-dots) 30)
                        (:size (nth sorted-dots (quot (count sorted-dots) 5)))
                        (if (> (count sorted-dots) 0)
                          (:size (nth sorted-dots (min 20 (dec (count sorted-dots)))))
                          0))
        filtered-dots (take 60 (filter #(>= (:size %) threshold-size) sorted-dots))
        _ (println "Filtered to" (count filtered-dots) "larger clusters")
        _ (when (seq filtered-dots)
            (println "Cluster size range:" (:size (first filtered-dots)) "to" (:size (last filtered-dots))))

        regions (atom [])
        n (count filtered-dots)
        tolerance 50
        max-distance 1000] ;; Max pixels between dots in a rectangle

    ;; Only try combinations if we have a reasonable number of dots
    (when (<= n 200)
      ;; Try all combinations of 4 dots
      (doseq [i (range n)
              j (range (inc i) n)
              :let [d1 (nth filtered-dots i)
                    d2 (nth filtered-dots j)]
              :when (< (distance d1 d2) max-distance)
              k (range (inc j) n)
              :let [d3 (nth filtered-dots k)]
              :when (and (< (distance d1 d3) max-distance)
                        (< (distance d2 d3) max-distance))
              l (range (inc k) n)
              :let [d4 (nth filtered-dots l)]
              :when (and (< (distance d1 d4) max-distance)
                        (< (distance d2 d4) max-distance)
                        (< (distance d3 d4) max-distance))]
        (when (dots-form-rectangle? d1 d2 d3 d4 tolerance)
          (let [sorted (sort-by (juxt :y :x) [d1 d2 d3 d4])
                [tl tr bl br] sorted]
            (swap! regions conj {:top-left tl
                                :top-right tr
                                :bottom-left bl
                                :bottom-right br})))))

    ;; Remove duplicate regions (same corners)
    (distinct @regions)))

;; ============================================================================
;; Line Extraction
;; ============================================================================

(defn- extract-line-value
  "Extract the Y position of the line at a given X coordinate within a region.
  Scans vertically from top to find the first dark pixel.
  Returns normalized value (0 = bottom, 1 = top), or nil if no line found."
  [^BufferedImage img x y-min y-max threshold]
  (let [;; Scan from top to bottom
        y-range (range (int y-min) (int (inc y-max)))]
    (when-let [line-y (first (filter #(is-dark-pixel? img x % threshold) y-range))]
      ;; Normalize: 0 at bottom, 1 at top
      (let [normalized (/ (- y-max line-y) (- y-max y-min))]
        (max 0.0 (min 1.0 normalized))))))

(defn- sample-line-in-region
  "Sample a line within a bounded region at N evenly-spaced points.
  Returns vector of normalized Y values (0 to 1)."
  [^BufferedImage img region n-points threshold]
  (let [{:keys [top-left top-right bottom-left bottom-right]} region
        x-min (:x top-left)
        x-max (:x top-right)
        y-min (min (:y top-left) (:y top-right))
        y-max (max (:y bottom-left) (:y bottom-right))

        ;; Generate evenly-spaced X coordinates
        x-step (/ (- x-max x-min) (dec n-points))
        x-coords (map #(+ x-min (* % x-step)) (range n-points))]

    ;; Extract Y value at each X coordinate
    (vec (map #(extract-line-value img (int %) y-min y-max threshold) x-coords))))

;; ============================================================================
;; Timestamp Generation
;; ============================================================================

(defn- generate-timestamps
  "Generate N evenly-spaced timestamps between start-time and end-time.
  Times are parsed using timeseries-plot/parse-time and returned as ISO-8601 strings."
  [start-time end-time n-points]
  (let [;; Parse times to Instant for calculation
        start-instant (-> start-time
                         (clajent.timeseries-plot/parse-time)
                         (.toInstant))
        end-instant (-> end-time
                       (clajent.timeseries-plot/parse-time)
                       (.toInstant))

        ;; Calculate total milliseconds between start and end
        total-millis (.until start-instant end-instant ChronoUnit/MILLIS)
        step-millis (/ total-millis (dec n-points))]

    ;; Generate timestamps
    (vec (for [i (range n-points)]
           (-> start-instant
               (.plusMillis (* i step-millis))
               (.toString))))))

;; ============================================================================
;; Public API
;; ============================================================================

(defn extract-timeseries-from-image
  "Extract multiple time-series from an image containing hand-drawn graphs.

  Each graph should be bounded by four dots marking the corners. The function will:
  1. Detect the bounding dots
  2. Extract the line within each bounded region
  3. Sample the line at n-points evenly-spaced intervals
  4. Normalize Y-axis values from 0 (bottom) to 1 (top)
  5. Generate timestamps from start-time to end-time

  Arguments:
  - image-path: Path to the image file
  - start-time: Start time (any format parseable by timeseries-plot/parse-time)
  - end-time: End time (same format as start-time)
  - n-points: Number of points to sample for each series
  - options: Optional map with keys:
    :dot-threshold - Brightness threshold for dot detection (default 100)
    :line-threshold - Brightness threshold for line detection (default 150)
    :min-dot-size - Minimum pixel count for a dot (default 10)

  Returns:
  List of maps, each with {:timestamps [...] :values [...]}, suitable for
  passing as :series argument to functions in timeseries.clj"
  ([image-path start-time end-time n-points]
   (extract-timeseries-from-image image-path start-time end-time n-points {}))

  ([image-path start-time end-time n-points options]
   (let [{:keys [dot-threshold line-threshold min-dot-size]
          :or {dot-threshold 100
               line-threshold 150
               min-dot-size 10}} options

         img (load-image image-path)

         ;; Find dots
         dots (find-dark-clusters img dot-threshold min-dot-size)
         _ (println "Found" (count dots) "dark clusters")

         ;; Group dots into rectangular regions
         regions (find-rectangular-regions dots)
         _ (println "Found" (count regions) "rectangular regions")

         ;; Generate timestamps once (shared across all series)
         timestamps (generate-timestamps start-time end-time n-points)]

     ;; Extract values for each region
     (for [region regions]
       (let [values (sample-line-in-region img region n-points line-threshold)]
         {:timestamps timestamps
          :values values})))))

;; ============================================================================
;; Example Usage
;; ============================================================================

(comment
  ;; Extract time series from an image
  (def series
    (extract-timeseries-from-image
      "/path/to/image.jpg"
      "2025-01-01T00:00:00Z"
      "2025-01-01T23:59:59Z"
      100))

  ;; The result can be used with timeseries functions
  (require '[clajent.timeseries :as ts])

  (doseq [s series]
    (println "Series statistics:"
             (ts/get-descriptive-statistics
               {:series s
                :statistic-name "mean"
                :col-name "target_col"}))))
