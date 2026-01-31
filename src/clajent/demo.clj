(ns clajent.demo
  (:require
    [clajent.core :as agent]))

(defn mystery [x]
  (Math/sin x))

(def call-function-tool (agent/function-tool "call-function" mystery "calls a function"
                                       "x" :number "mystery parameter"))

(def tools [call-function-tool])

(defn do-it [] (agent/process "Figure out what the mystery tool does." tools false))

(do-it)

