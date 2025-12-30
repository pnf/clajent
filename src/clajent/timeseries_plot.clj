(ns clajent.timeseries-plot
  (:import [clajent NiceScale]
           [org.jfree.chart ChartFactory ChartPanel ChartUtils]
           [org.jfree.chart.plot PlotOrientation XYPlot CombinedDomainXYPlot]
           [org.jfree.data.time TimeSeries TimeSeriesCollection Millisecond]
           [org.jfree.chart.axis NumberAxis DateAxis]
           [org.jfree.chart.renderer.xy XYLineAndShapeRenderer]
           [java.awt Color]
           [java.io ByteArrayOutputStream File]
           [java.time Instant LocalDateTime ZoneId]
           [java.util Date Locale TimeZone]
           [javax.swing JFrame]))

(defn- parse-time
  "Parse a time value into a java.util.Date.
  Accepts:
  - java.util.Date (returned as-is)
  - java.time.Instant
  - java.time.LocalDateTime
  - Long (milliseconds since epoch)
  - String (ISO-8601 format)"
  [time-val]
  ; TimeZone.setDefault(TimeZone.getTimeZone("UTC")
  (TimeZone/setDefault (TimeZone/getTimeZone "UTC"))
  (cond
    (instance? Date time-val)
    time-val

    (instance? Instant time-val)
    (Date/from time-val)

    (instance? LocalDateTime time-val)
    (Date/from (.toInstant (.atZone time-val (ZoneId/systemDefault))))

    (number? time-val)
    (Date. (long time-val))

    (string? time-val)
    (try
      (Date/from (Instant/parse time-val))
      (catch Exception e
        (throw (IllegalArgumentException.
                 (str "Unable to parse time string: " time-val
                      ". Expected ISO-8601 format.")))))

    :else
    (throw (IllegalArgumentException.
             (str "Unsupported time type: " (type time-val))))))

(defn- create-time-series
  "Create a JFreeChart TimeSeries from a name, times, and values."
  [series-name times values]
  (let [ts (TimeSeries. series-name)]
    (doseq [[time-val value] (map vector times values)]
      (when (and time-val value (not (Double/isNaN value)))
        (let [date (parse-time time-val)
              ms (Millisecond. date)]
          (.add ts ms (double value)))))
    ts))

(defn- create-single-plot-dataset
  "Create a dataset with multiple time series for a single plot."
  [series-names times values-list]
  (let [dataset (TimeSeriesCollection.)]
    (doseq [[name values] (map vector series-names values-list)]
      (.addSeries dataset (create-time-series name times values)))
    dataset))

(defn- get-min-max
  "Get the min and max values from a list of numbers, ignoring NaN values."
  [values]
  (let [valid-values (filter #(and % (not (Double/isNaN %))) values)]
    (if (seq valid-values)
      [(apply min valid-values) (apply max valid-values)]
      [0.0 1.0])))

(defn- apply-nice-scale
  "Apply NiceScale to a NumberAxis."
  [axis values]
  (let [[min-val max-val] (get-min-max values)
        nice-scale (NiceScale. min-val max-val)
        nice-min (.getNiceMin nice-scale)
        nice-max (.getNiceMax nice-scale)]
    (.setRange axis nice-min nice-max)))

(defn- create-iso-date-axis
  "Create a DateAxis formatted with ISO-8601 format in UTC."
  [label]
  (let [utc-tz (TimeZone/getTimeZone "UTC")
        axis (DateAxis. label utc-tz (Locale/getDefault))
        date-format (java.text.SimpleDateFormat. "yyyy-MM-dd'T'HH:mm:ss'Z'")]
    (.setTimeZone date-format utc-tz)
    (.setDateFormatOverride axis date-format)
    axis))

(defn- create-trellis-plot
  "Create a combined plot with individual subplots for each series."
  [series-names times values-list]
  (let [combined-plot (CombinedDomainXYPlot. (create-iso-date-axis "Time"))]
    (doseq [[name values] (map vector series-names values-list)]
      (let [dataset (TimeSeriesCollection.)
            _ (.addSeries dataset (create-time-series name times values))
            range-axis (NumberAxis. name)
            _ (apply-nice-scale range-axis values)
            subplot (XYPlot. dataset
                            nil
                            range-axis
                            (XYLineAndShapeRenderer. true false))]
        (.setRangeAxisLocation subplot org.jfree.chart.axis.AxisLocation/BOTTOM_OR_LEFT)
        (.add combined-plot subplot 1)))
    (.setOrientation combined-plot PlotOrientation/VERTICAL)
    (doto (org.jfree.chart.JFreeChart.
            "Time Series"
            org.jfree.chart.JFreeChart/DEFAULT_TITLE_FONT
            combined-plot
            true)
      (.setBackgroundPaint Color/WHITE))))

(defn- create-single-chart
  "Create a chart with all series on the same plot."
  [series-names times values-list]
  (let [dataset (create-single-plot-dataset series-names times values-list)
        chart (ChartFactory/createTimeSeriesChart
                "Time Series"
                "Time"
                "Value"
                dataset
                true  ; legend
                true  ; tooltips
                false) ; urls
        plot (.getPlot chart)
        date-axis (create-iso-date-axis "Time")
        range-axis (.getRangeAxis plot)
        all-values (apply concat values-list)]
    (.setDomainAxis plot date-axis)
    (apply-nice-scale range-axis all-values)
    (.setBackgroundPaint chart Color/WHITE)
    chart))

(defn- render-to-swing
  "Display the chart in a Swing window."
  [chart width height]
  (let [frame (JFrame. "Time Series Chart")
        panel (ChartPanel. chart)]
    (.setPreferredSize panel (java.awt.Dimension. width height))
    (.setContentPane frame panel)
    (.pack frame)
    (.setVisible frame true)
    (.setDefaultCloseOperation frame JFrame/DISPOSE_ON_CLOSE)
    frame))

(defn- render-to-file
  "Save the chart to a PNG file."
  [chart width height file-path]
  (ChartUtils/saveChartAsPNG (File. file-path) chart width height))

(defn- render-to-bytes
  "Render the chart to a byte array containing PNG data."
  [chart width height]
  (let [baos (ByteArrayOutputStream.)]
    (ChartUtils/writeChartAsPNG baos chart width height)
    (.toByteArray baos)))

(defn plot-timeseries
  "Create a time-series chart with flexible options.

  Arguments:
  - times: A list of N time values (Date, Instant, LocalDateTime, Long ms, or ISO-8601 String)
  - series-names: A list of M series names (Strings)
  - values-list: A list of M lists, each containing N numeric values
  - options: Optional map with keys:
    :output - One of:
      :swing (default) - Display in a Swing window
      {:file \"path/to/file.png\"} - Save to PNG file
      :bytes - Return byte array containing PNG
    :width - Width in pixels (default 800)
    :height - Height in pixels (default 600)
    :combined - Boolean, if true all series on one plot, if false separate subplots (default true)

  Returns:
  - For :swing output: the JFrame
  - For :file output: the file path
  - For :bytes output: byte array containing PNG data

  Example:
  (plot-timeseries
    [1640000000000 1640000060000 1640000120000]
    [\"CPU\" \"Memory\"]
    [[10 20 30] [50 60 70]]
    {:output :swing :width 1024 :height 768 :combined true})"
  ([times series-names values-list]
   (plot-timeseries times series-names values-list {}))

  ([times series-names values-list options]
   (let [{:keys [output width height combined]
          :or {output :swing
               width 800
               height 600
               combined true}} options

         _ (when (not= (count times) (count (first values-list)))
             (throw (IllegalArgumentException.
                      (str "Dimension mismatch: times has " (count times)
                           " elements but first series has "
                           (count (first values-list)) " elements"))))

         _ (when (not= (count series-names) (count values-list))
             (throw (IllegalArgumentException.
                      (str "Dimension mismatch: " (count series-names)
                           " series names but " (count values-list) " value lists"))))

         chart (if combined
                 (create-single-chart series-names times values-list)
                 (create-trellis-plot series-names times values-list))]

     (cond
       (= output :swing)
       (render-to-swing chart width height)

       (and (map? output) (:file output))
       (do
         (render-to-file chart width height (:file output))
         (:file output))

       (= output :bytes)
       (render-to-bytes chart width height)

       :else
       (throw (IllegalArgumentException.
                (str "Unknown output type: " output
                     ". Expected :swing, {:file \"path\"}, or :bytes")))))))

(comment
  ;; Example usage with milliseconds - all series on one plot
  (plot-timeseries
    [1640000000000 1640000060000 1640000120000 1640000180000]
    ["Temperature" "Humidity"]
    [[20.5 21.0 21.5 22.0]
     [65.0 66.0 64.5 63.0]]
    {:output :swing
     :width 1024
     :height 768
     :combined true})  ; All series on one plot

  ;; Example with ISO-8601 strings - separate subplots
  (plot-timeseries
    ["2024-01-01T00:00:00Z" "2024-01-01T01:00:00Z" "2024-01-01T02:00:00Z"]
    ["CPU" "Memory" "Disk"]
    [[10 20 30]
     [50 60 70]
     [30 35 40]]
    {:output :swing
     :combined false})  ; Each series gets its own subplot

  ;; Example saving to file
  (plot-timeseries
    [1640000000000 1640000060000 1640000120000]
    ["Series A" "Series B"]
    [[1 2 3] [4 5 6]]
    {:output {:file "/tmp/timeseries.png"}
     :width 800
     :height 600})

  ;; Example getting bytes
  (let [png-bytes (plot-timeseries
                    [1640000000000 1640000060000]
                    ["Data"]
                    [[1 2]]
                    {:output :bytes
                     :width 400
                     :height 300})]
    (println "Generated" (count png-bytes) "bytes")))
