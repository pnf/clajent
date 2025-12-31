(ns clajent.ts-sample-data
  "Port of timesense/basic_usage.py generate_sample_data and EXAMPLE_REQUESTS
   Pure JVM implementation only. For Python interop, see clajent.ts-sample-data-python"
  (:import [java.time LocalDateTime ZoneOffset]
           [java.time.format DateTimeFormatter]
           [java.util Random]))

;; ============================================================================
;; Pure JVM Implementation
;; ============================================================================

(defn- linspace
  "Generate N evenly spaced values between start and end (inclusive)"
  [start end n]
  (let [step (/ (- end start) (dec n))]
    (mapv #(+ start (* % step)) (range n))))

(defn- random-normal
  "Generate a sequence of n random numbers from normal distribution N(mean, stddev)"
  [mean stddev n]
  (let [rng (Random.)]
    (mapv (fn [_] (+ mean (* stddev (.nextGaussian rng)))) (range n))))

(defn- date-range
  "Generate n ISO datetime strings starting from base-date, incrementing by hours"
  [base-date n]
  (let [formatter (DateTimeFormatter/ISO_LOCAL_DATE_TIME)]
    (mapv #(.format (.plusHours base-date %) formatter)
          (range n))))

(defn generate-sample-data-jvm
  "Pure JVM implementation of generate_sample_data() from basic_usage.py
   Returns [series-1 series-2 series-3] as Clojure maps."
  []
  (let [n 100
        dates (date-range (LocalDateTime/of 2025 1 1 0 0) n)

        ;; Series 1: Trending series with spike
        base-values (linspace 10.0 30.0 n)
        noise (random-normal 0.0 2.0 n)
        values-1 (mapv + base-values noise)
        values-1-with-spike (assoc values-1 50 50.0)

        series-1 {:name "metric_a"
                  :timestamps dates
                  :values values-1-with-spike}

        ;; Series 2: Series with change point
        part1 (random-normal 10.0 1.0 40)
        part2 (random-normal 20.0 1.0 60)
        values-2 (vec (concat part1 part2))

        series-2 {:name "metric_b"
                  :timestamps dates
                  :values values-2}

        ;; Series 3: Cyclical pattern
        t (range n)
        sin-component (mapv #(* 5.0 (Math/sin (/ (* 2.0 Math/PI %) 20.0))) t)
        noise-3 (random-normal 0.0 0.5 n)
        values-3 (mapv #(+ 15.0 %1 %2) sin-component noise-3)

        series-3 {:name "metric_c"
                  :timestamps dates
                  :values values-3}]

    [series-1 series-2 series-3]))

(defn example-requests-jvm
  "Pure JVM implementation of EXAMPLE_REQUESTS from basic_usage.py
   Returns a map of example requests using JVM-generated sample data."
  []
  (let [[s1 s2 s3] (generate-sample-data-jvm)]
    {:analyze-extreme
     {:series [s1]
      :question "What is the maximum value in this time series and where does it occur?"
      :task-type "extreme"
      :verify true}

     :analyze-trend
     {:series [s1]
      :question "What is the overall trend of this time series?"
      :task-type "trend"
      :verify true}

     :detect-spike
     {:series [s1]
      :question "Are there any spikes or anomalies in this time series?"
      :task-type "spike"
      :verify true}

     :detect-change-point
     {:series [s2]
      :question "Identify any change points where the series behavior changes significantly."
      :task-type "change_point"
      :verify true}

     :segment-series
     {:series [s3]
      :question "Segment this time series into distinct phases."
      :task-type "segment"
      :verify true
      :window-size 20}

     :compare-series
     {:series [s1 s2]
      :question "Compare the behavior of these two time series."
      :aspect "trends and volatility"}

     :detect-anomalies
     {:series [s1 s2]
      :interval [40 60]
      :anomaly-rules ["Series 1: values above 45 are anomalous"
                      "Series 2: sudden jumps > 5 are anomalous"]}

     :comprehensive-analysis
     {:series [s1]
      :question "Provide a comprehensive analysis of this time series including trends, anomalies, and key patterns."
      :task-type "describe"
      :verify false}}))

;; ============================================================================
;; Python Interop Implementation - See clajent.ts-sample-data-python
;; ============================================================================

;; ============================================================================
;; Verification Functions
;; ============================================================================

(defn- series-similar?
  "Check if two time series are structurally similar (same keys, same lengths, similar values).
   Values are considered similar if they're within tolerance (for floating point comparison)."
  [s1 s2 tolerance]
  (and (= (:name s1) (:name s2))
       (= (count (:timestamps s1)) (count (:timestamps s2)))
       (= (count (:values s1)) (count (:values s2)))
       ;; Timestamps should be identical strings
       (= (:timestamps s1) (:timestamps s2))
       ;; Values should be close (within tolerance)
       (every? (fn [[v1 v2]]
                (< (Math/abs (- v1 v2)) tolerance))
              (map vector (:values s1) (:values s2)))))

(defn verify-implementations
  "Verify that JVM and Python implementations produce structurally equivalent data.
   Returns {:match? boolean :details string}"
  []
  (try
    (require '[clajent.ts-sample-data-python :as py-impl])
    (let [jvm-data (generate-sample-data-jvm)
          py-data ((resolve 'clajent.ts-sample-data-python/generate-sample-data-python))
          tolerance 0.01  ; Allow small floating point differences

          ;; Check each series
          series-checks (mapv (fn [idx]
                               (let [jvm-series (nth jvm-data idx)
                                     py-series (nth py-data idx)]
                                 {:index idx
                                  :name (:name jvm-series)
                                  :match? (series-similar? jvm-series py-series tolerance)
                                  :jvm-length (count (:values jvm-series))
                                  :py-length (count (:values py-series))}))
                             (range 3))

          all-match? (every? :match? series-checks)]

      {:match? all-match?
       :series-checks series-checks
       :details (if all-match?
                 "✓ All series match structurally"
                 (str "✗ Mismatch detected: "
                      (pr-str (filter (complement :match?) series-checks))))})

    (catch Exception e
      {:match? false
       :error (.getMessage e)
       :details (str "Error during verification: " (.getMessage e))})))

(comment
  ;; Test JVM implementation
  (def jvm-result (generate-sample-data-jvm))
  (first jvm-result)

  ;; Test Python implementation
  (def py-result (generate-sample-data-python))
  (first py-result)

  ;; Verify they match
  (verify-implementations)

  ;; Get example requests
  (keys (example-requests-jvm))
  (keys (example-requests-python))

  ;; Compare a specific request
  (get (example-requests-jvm) :analyze-extreme)
  (get (example-requests-python) :analyze-extreme))
