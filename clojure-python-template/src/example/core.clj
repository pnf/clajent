(ns example.core
  "Example namespace demonstrating Clojure + Python/pandas interoperability."
  (:require [libpython-clj2.python :refer [py. py.. py.-] :as py]))

;; ============================================================================
;; Python Initialization
;; ============================================================================

(defn init-python!
  "Initialize the Python runtime. Call this before using any Python interop."
  []
  (py/initialize!))

;; ============================================================================
;; Pandas DataFrame Examples
;; ============================================================================

(defn create-dataframe
  "Create a pandas DataFrame from Clojure data.

   Example:
     (create-dataframe {:name [\"Alice\" \"Bob\" \"Carol\"]
                        :age [25 30 35]
                        :city [\"NYC\" \"LA\" \"SF\"]})"
  [data-map]
  (let [pd (py/import-module "pandas")]
    (py. pd DataFrame data-map)))

(defn df->clj
  "Convert a pandas DataFrame to a Clojure vector of maps.

   Example:
     (df->clj df) => [{:name \"Alice\" :age 25} ...]"
  [df]
  (let [records (py. df to_dict :orient "records")]
    (mapv #(into {} (map (fn [[k v]] [(keyword k) v]) %)) records)))

(defn describe-df
  "Get summary statistics for a DataFrame."
  [df]
  (py. df describe))

;; ============================================================================
;; Numpy Examples
;; ============================================================================

(defn create-array
  "Create a numpy array from a Clojure sequence.

   Example:
     (create-array [1 2 3 4 5])"
  [coll]
  (let [np (py/import-module "numpy")]
    (py. np array coll)))

(defn array-stats
  "Get basic statistics for a numpy array.
   Returns map with :sum, :mean, :std, :min, :max."
  [arr]
  (let [np (py/import-module "numpy")]
    {:sum  (double (py. np sum arr))
     :mean (double (py. np mean arr))
     :std  (double (py. np std arr))
     :min  (double (py. np min arr))
     :max  (double (py. np max arr))}))

;; ============================================================================
;; Demo Function
;; ============================================================================

(defn demo
  "Run a quick demo of Python interop capabilities."
  []
  (init-python!)

  (println "\n=== Clojure + Python/pandas Demo ===\n")

  ;; DataFrame demo
  (println "1. Creating pandas DataFrame:")
  (let [df (create-dataframe {"name"  ["Alice" "Bob" "Carol" "David"]
                              "age"   [25 30 35 40]
                              "score" [85.5 92.0 78.5 95.0]})]
    (println df)
    (println "\n2. DataFrame statistics:")
    (println (describe-df df))
    (println "\n3. DataFrame as Clojure data:")
    (println (df->clj df)))

  ;; Numpy demo
  (println "\n4. Numpy array operations:")
  (let [arr (create-array [1 2 3 4 5 6 7 8 9 10])]
    (println "Array:" arr)
    (println "Stats:" (array-stats arr)))

  (println "\n=== Demo Complete ==="))
