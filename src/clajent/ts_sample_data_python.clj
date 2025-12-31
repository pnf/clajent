(ns clajent.ts-sample-data-python
  "Python interop invocation of timesense/basic_usage.py"
  (:require [libpython-clj2.python :as py]
            [libpython-clj2.require :refer [require-python]]
            [babashka.fs :as fs])
  (:import [java.io File]))

(require-python  "./timeseries/timesense" '[basic_usage :reload :as bu] )

(defn generate-sample-data-python
  "Python interop implementation that calls the original Python generate_sample_data()
   Converts all results to pure Clojure data structures.
   Returns [series-1 series-2 series-3] as Clojure maps."
  []
  (let [        ;; Call Python function and get tuple result
        py-result (bu/generate_sample_data)
        ;; Convert Python tuple to Clojure vector of maps
        clj-result (mapv (fn [py-series]
                          {:name (py/get-item py-series "name")
                           :timestamps (vec (py/get-item py-series "timestamps"))
                           :values (vec (py/get-item py-series "values"))})
                        py-result)]
    clj-result))

(defn example-requests-python
  "Python interop implementation that uses Python-generated sample data
   Returns a map of example requests with data from Python."
  []
  (let [
        ;; Get EXAMPLE_REQUESTS dict from Python
        py-requests bu/EXAMPLE_REQUESTS
        ;; Convert to Clojure structure
        convert-series (fn [py-series]
                        {:name (py/get-item py-series "name")
                         :timestamps (vec (py/get-item py-series "timestamps"))
                         :values (vec (py/get-item py-series "values"))})

        ;; Helper to safely get optional Python dict items
        py-get-opt (fn [py-obj key]
                    (try
                      (py/get-item py-obj key)
                      (catch Exception _ nil)))

        convert-request (fn [py-req]
                         (let [base {:question (py-get-opt py-req "question")
                                    :task-type (py-get-opt py-req "task_type")
                                    :verify (py-get-opt py-req "verify")
                                    :aspect (py-get-opt py-req "aspect")
                                    :window-size (py-get-opt py-req "window_size")
                                    :interval (when-let [iv (py-get-opt py-req "interval")]
                                              (vec iv))
                                    :anomaly-rules (when-let [rules (py-get-opt py-req "anomaly_rules")]
                                                    (vec rules))
                                    :series (when-let [series (py-get-opt py-req "series")]
                                             (mapv convert-series series))}]
                           ;; Remove nil values
                           (into {} (filter (comp some? val) base))))]

    ;; Convert each request
    {:analyze-extreme (convert-request (py/get-item py-requests "analyze_extreme"))
     :analyze-trend (convert-request (py/get-item py-requests "analyze_trend"))
     :detect-spike (convert-request (py/get-item py-requests "detect_spike"))
     :detect-change-point (convert-request (py/get-item py-requests "detect_change_point"))
     :segment-series (convert-request (py/get-item py-requests "segment_series"))
     :compare-series (convert-request (py/get-item py-requests "compare_series"))
     :detect-anomalies (convert-request (py/get-item py-requests "detect_anomalies"))
     :comprehensive-analysis (convert-request (py/get-item py-requests "comprehensive_analysis"))}))
