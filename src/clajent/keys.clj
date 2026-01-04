(ns clajent.keys
  (:require [clojure.java.shell :as sh]
            [clojure.string :as s])
  )

; security add-generic-password -s OPENAI_API_KEY -a "$(whoami)" -w "$(pbpaste)"
; security add-generic-password -s OPENAI_API_KEY -a "$(whoami)" -w "$(pbpaste)" -U
; security find-generic-password -s OPENAI_API_KEY -a "$(whoami)" -w | pbcopy

(defn get [^String k]
  (or (:out (System/getenv k))
      (-> (sh/sh "security" "find-generic-password" "-s" k "-a" (System/getProperty "user.name") "-w")
          :out
          s/trim)
      (throw (IllegalArgumentException. (str "Cannot find " k " in either environment or keychain")))
      ))
