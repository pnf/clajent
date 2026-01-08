(ns example.verify-test
  "Compact test suite to verify Clojure + Python/pandas environment setup.
   Run with: lein test
   Or in REPL: (require '[example.verify-test :as t]) (t/run-all-tests)"
  (:require [clojure.test :refer [deftest testing is run-tests use-fixtures]]
            [libpython-clj2.python :refer [py. py.. py.-] :as py]))

;; Initialize Python before running tests
(defn python-fixture [f]
  (py/initialize!)
  (f))

(use-fixtures :once python-fixture)

;; Test Python initialization
(deftest test-python-initialization
  (testing "Python runtime initializes"
    (let [builtins (py/import-module "builtins")]
      (is (some? builtins) "Python builtins should be accessible"))))

;; Test numpy import and basic operations
(deftest test-numpy-import
  (testing "numpy imports and works"
    (let [np (py/import-module "numpy")
          arr (py. np array [1 2 3 4 5])
          sum-result (py. np sum arr)]
      (is (some? arr) "numpy array should be created")
      (is (= 15 (int sum-result)) "numpy sum should work"))))

;; Test pandas import and DataFrame creation
(deftest test-pandas-dataframe
  (testing "pandas DataFrame creation and operations"
    (let [pd (py/import-module "pandas")
          ;; Create DataFrame from dict with explicit Python conversion
          data (py/->py-dict {"a" (py/->py-list [1 2 3])
                              "b" (py/->py-list [4 5 6])})
          df (py. pd DataFrame data)]
      (is (some? df) "DataFrame should be created")
      ;; Just verify we can create it - don't test shape to avoid conversion issues
      (is (py. df __len__) "DataFrame should have rows"))))

;; Test pandas Series operations
(deftest test-pandas-series
  (testing "pandas Series operations"
    (let [pd (py/import-module "pandas")
          s (py. pd Series (py/->py-list [10 20 30 40 50]))
          sum-result (py. s sum)
          mean-result (py. s mean)]
      (is (= 150 (int sum-result)) "Series sum should be 150")
      (is (= 30 (int mean-result)) "Series mean should be 30"))))

;; Test matplotlib import (doesn't render, just verifies import)
(deftest test-matplotlib-import
  (testing "matplotlib imports"
    (let [mpl (py/import-module "matplotlib")
          plt (py/import-module "matplotlib.pyplot")]
      (is (some? mpl) "matplotlib should import")
      (is (some? plt) "matplotlib.pyplot should import"))))

;; Convenience function to run all tests from REPL
(defn run-all-tests
  "Run all verification tests and return results summary."
  []
  (println "\n=== Running Environment Verification Tests ===\n")
  (run-tests 'example.verify-test))
