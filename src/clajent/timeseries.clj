(ns clajent.timeseries
  "Clojure wrapper for Python timeseries analysis functions from tsuting/timeseries.py"
  (:require [libpython-clj2.python :as py]
            [libpython-clj2.require :refer [require-python]]))

;; Import the Python timeseries module
(require-python "./timeseries/tsuting" '[timeseries :reload :as ts])

;; Also need pandas for DataFrame conversion
(require-python '[pandas :as pd])

;; ============================================================================
;; Conversion Functions: Clojure → Python DataFrame
;; ============================================================================

(defn clj-series->pandas-df
  "Convert a Clojure time series map to a pandas DataFrame.

   Input format:
   {:name \"metric_a\"
    :timestamps [\"2025-01-01T00:00:00\" \"2025-01-01T01:00:00\" ...]
    :values [10.2 11.5 12.3 ...]}

   Output: pandas DataFrame with two columns (time_col and target_col)"
  [{:keys [timestamps values name] :as series}]
  (when-not (and timestamps values)
    (throw (ex-info "Series must contain :timestamps and :values"
                    {:series series})))

  ;; Create a Python dict with the data
  (let [data (py/->py-dict {"time_col" timestamps
                            "target_col" values})]
    ;; Create DataFrame from dict
    (pd/DataFrame data)))

(defn save-df-to-temp-csv
  "Save a pandas DataFrame to a temporary CSV file and return the path.
   This is needed because many functions expect a file_path argument."
  [df]
  (let [temp-file (java.io.File/createTempFile "timeseries-" ".csv")]
    (.deleteOnExit temp-file)
    (let [path (.getAbsolutePath temp-file)]
      ;; Use py/call-attr-kw for keyword arguments
      (py/call-attr-kw df "to_csv" [path] {:index false})
      path)))

;; ============================================================================
;; Wrapped Python Functions
;; ============================================================================

(defn get-time-col-and-target-col
  "Get the time column and target column names from a CSV file or DataFrame.

   Args:
     file-path: Path to CSV file (optional if df provided)
     df: pandas DataFrame (optional if file-path provided)

   Returns:
     {:time-col \"time_col_name\" :target-col \"target_col_name\"}"
  [{:keys [file-path df]}]
  (let [result (if df
                 (ts/get_time_col_and_target_col :df df)
                 (ts/get_time_col_and_target_col :file_path file-path))]
    {:time-col (py/get-item result "time_col")
     :target-col (py/get-item result "target_col")}))

(defn get-descriptive-statistics
  "Get descriptive statistics for a column.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)
     :statistic-name - One of: \"count\", \"mean\", \"std\", \"min\", \"25%\", \"50%\", \"75%\", \"max\", \"sum\"
     :col-name - \"time_col\" or \"target_col\"
     :start-index - Start index (optional, can be negative)
     :end-index - End index (optional, can be negative)
     :selected-day-in-a-week - 0-6 for Mon-Sun (optional)

   Returns: float"
  [{:keys [file-path series statistic-name col-name start-index end-index selected-day-in-a-week]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_descriptive_statistics
     :file_path file-path
     :statistic_name statistic-name
     :col_name col-name
     :start_index start-index
     :end_index end-index
     :selected_day_in_a_week selected-day-in-a-week)))

(defn get-number-of-outliers
  "Get the number of outliers in the target column.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)

   Returns: int"
  [{:keys [file-path series]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_number_of_outliers :file_path file-path)))

(defn get-frequency
  "Get the inferred frequency of the time series.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)

   Returns: string (e.g., \"H\" for hourly, \"D\" for daily)"
  [{:keys [file-path series]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_frequency :file_path file-path)))

(defn get-number-of-missing-datetime
  "Get the number of missing datetime values.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)

   Returns: int"
  [{:keys [file-path series]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_number_of_missing_datetime :file_path file-path)))

(defn get-number-of-null-values
  "Get the number of null values in a column.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)
     :col-name - \"time_col\" or \"target_col\"

   Returns: int"
  [{:keys [file-path series col-name]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_number_of_null_values :file_path file-path :col_name col-name)))

(defn get-seasonality
  "Check if the time series has seasonality or return the seasonality period.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)
     :type - \"has_seasonality\" (returns bool) or \"seasonality_period\" (returns int)

   Returns: bool or int depending on :type"
  [{:keys [file-path series type]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_seasonality :file_path file-path :type type)))

(defn get-trend-by-pearson-correlation
  "Get the Pearson correlation between the trend and the index.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)
     :seasonality-period - The seasonality period (int)

   Returns: float (correlation coefficient)"
  [{:keys [file-path series seasonality-period]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_trend_by_pearson_correlation
     :file_path file-path
     :seasonality_period seasonality-period)))

(defn get-moving-average
  "Calculate the moving average.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)
     :window-size - Window size for moving average (int)

   Returns: vector of floats"
  [{:keys [file-path series window-size]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))
        result (ts/get_moving_average
                :file_path file-path
                :window_size window-size)]
    ;; Convert pandas Series to Clojure vector
    (vec (py/->jvm result))))

(defn get-forecasting
  "Forecast the next data point using naive or average method.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)
     :model-name - \"naive\" or \"average\"
     :forecast-time - Datetime string (e.g., \"2025-01-01 12:00:00\")

   Returns: float (forecasted value)"
  [{:keys [file-path series model-name forecast-time]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_forecasting
     :file_path file-path
     :model_name model-name
     :forecast_time forecast-time)))

(defn get-extreme-acf-pacf-lag
  "Find the lag with extreme (max or min) autocorrelation.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)
     :start-lag - Start lag (int)
     :end-lag - End lag (int)
     :type-of-correlation - \"acf\" or \"pacf\"
     :return-max-or-min - \"max\" or \"min\"
     :return-absolute - true or false

   Returns: int (lag index)"
  [{:keys [file-path series start-lag end-lag type-of-correlation
           return-max-or-min return-absolute]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/get_extreme_acf_pacf_lag
     :file_path file-path
     :start_lag start-lag
     :end_lag end-lag
     :type_of_correlation type-of-correlation
     :return_max_or_min return-max-or-min
     :return_absolute return-absolute)))

(defn calculate-weekend-pearson-correlation
  "Calculate Pearson correlation between target and weekends.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)

   Returns: float (correlation coefficient)"
  [{:keys [file-path series]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))]
    (ts/calculate_weekend_pearson_correlation :file_path file-path)))

(defn retrieve-a-row-by-index
  "Retrieve a row from the dataset by index.

   Args (map):
     :file-path - Path to CSV file (required unless :series provided)
     :series - Clojure time series map (alternative to :file-path)
     :index - Row index (int, can be negative)

   Returns: map with row data"
  [{:keys [file-path series index]}]
  (let [file-path (or file-path
                      (when series
                        (save-df-to-temp-csv (clj-series->pandas-df series))))
        result (ts/retrieve_a_row_by_index :file_path file-path :index index)]
    ;; Convert Python dict to Clojure map
    (into {} (py/->jvm result))))

;; ============================================================================
;; Example Usage
;; ============================================================================

(comment
  ;; Create a sample time series
  (def sample-series
    {:name "test-metric"
     :timestamps (mapv #(format "2025-01-01 %02d:00:00" %) (range 24))
     :values (mapv #(+ 10 (* 2 %) (rand)) (range 24))})

  ;; Get statistics
  (get-descriptive-statistics
   {:series sample-series
    :statistic-name "mean"
    :col-name "target_col"})

  ;; Check for outliers
  (get-number-of-outliers {:series sample-series})

  ;; Get frequency
  (get-frequency {:series sample-series})

  ;; Get moving average
  (get-moving-average {:series sample-series
                       :window-size 3})

  ;; Simple forecast
  (get-forecasting {:series sample-series
                    :model-name "average"
                    :forecast-time "2025-01-01 20:00:00"}))
