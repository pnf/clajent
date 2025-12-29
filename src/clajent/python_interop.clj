(ns clajent.python-interop
  (:require [libpython-clj2.python :refer [py. py.. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [gigasquid.plot :as gplot]
            ))

(require-python '[numpy :as np]
                '[numpy.random :as np-random]
                '[datetime :as dt]
                'darts
                '[matplotlib.pyplot :as pyplot]
                )
;(require-python '[operators :refer :*])

(dt/datetime 2025 1 1)

;(gplot/with-show (pyplot/plot [[1 2 3 4 5] [1 2 3 4 10]] :label "linear"))

;; We need to create dataframes for fake time-series with interesting features

(require-python  "/Users/pnf/dev/clajent/timeseries/tsuting" '[timeseries :reload :as ts] )

(require-python  "/Users/pnf/dev/clajent/timeseries/timesense" '[basic_usage :reload :as bu] )

(require-python  "/Users/pnf/dev/clajent/timeseries/timesense" '[server :reload :as sv] )

(def er (bu/get_example_requests))


(def x (py/->py-list (map #(py/call-attr (dt/datetime 2025 01 01) :__add__ (dt/timedelta :hours %)) (range 0 100))))

(def y (np/add (np/linspace 10 30 100) (np-random/normal 0 2 100)))


