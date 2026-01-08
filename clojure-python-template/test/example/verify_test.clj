(ns example.verify-test
  "Compact test suite to verify Clojure + Python/pandas environment setup.
   Run with: lein test
   Or in REPL: (require '[example.verify-test :as t]) (t/run-all-tests)"
  (:require [clojure.test :refer [deftest testing is run-tests]]
            [libpython-clj2.python :refer [py. py.. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]))

;; Initialize Python - this must succeed for any Python interop to work
(deftest test-python-initialization
  (testing "Python runtime initializes"
    (is (py/initialize!) "Python should initialize successfully")))

;; Test numpy import and basic operations
(deftest test-numpy-import
  (testing "numpy imports and works"
    (require-python '[numpy :as np])
    (let [arr (np/array [1 2 3 4 5])]
      (is (some? arr) "numpy array should be created")
      (is (= 15.0 (py. (np/sum arr) __float__)) "numpy sum should work"))))

;; Test pandas import and DataFrame creation
(deftest test-pandas-dataframe
  (testing "pandas DataFrame creation and operations"
    (require-python '[pandas :as pd])
    (let [df (pd/DataFrame {"a" [1 2 3] "b" [4 5 6]})]
      (is (some? df) "DataFrame should be created")
      (is (= [3 2] (vec (py.- df shape))) "DataFrame shape should be [3, 2]")
      (is (= ["a" "b"] (vec (py.- df columns))) "DataFrame columns should be [a, b]"))))

;; Test pandas Series operations
(deftest test-pandas-series
  (testing "pandas Series operations"
    (require-python '[pandas :as pd])
    (let [s (pd/Series [10 20 30 40 50])]
      (is (= 150.0 (py. (py. s sum) __float__)) "Series sum should be 150")
      (is (= 30.0 (py. (py. s mean) __float__)) "Series mean should be 30"))))

;; Test matplotlib import (doesn't render, just verifies import)
(deftest test-matplotlib-import
  (testing "matplotlib imports"
    (require-python '[matplotlib :as mpl])
    (require-python '[matplotlib.pyplot :as plt])
    (is (some? plt) "matplotlib.pyplot should import")))

;; Convenience function to run all tests from REPL
(defn run-all-tests
  "Run all verification tests and return results summary."
  []
  (println "\n=== Running Environment Verification Tests ===\n")
  (run-tests 'example.verify-test))
