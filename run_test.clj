(ns run-test
  "Runner script for image extraction test"
  (:require [clajent.image-timeseries-test :as test]))

(println "Starting image extraction test...")
(try
  (test/run-test)
  (println "Test completed successfully!")
  (System/exit 0)
  (catch Exception e
    (println "Error during test:")
    (println (.getMessage e))
    (.printStackTrace e)
    (System/exit 1)))
