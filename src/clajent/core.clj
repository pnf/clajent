(ns clajent.core
  (:require    [clojure.data.json :as json]
               [clajent.secrets :as secrets]                ;; should define token
               )
  (:import [com.openai.client.okhttp OpenAIOkHttpClient]
           [com.openai.core JsonValue ObjectMappers]
           [com.openai.models.responses Response ResponseCreateParams ResponseCreateParams$Builder ResponseCreateParams$Input ResponseFunctionToolCall
                                        ResponseInputItem ResponseInputItem$FunctionCallOutput ResponseInputItem$Message ResponseInputItem$Message$Role Tool FunctionTool FunctionTool$Parameters]
           [com.openai.core ObjectMappers JsonValue]
           [com.openai.models Reasoning Reasoning$Summary Reasoning$Summary$Companion ReasoningEffort]
           (java.util Optional))
  )


(def client (.. OpenAIOkHttpClient (builder)
                (apiKey secrets/open-router-token)
                (baseUrl "https://openrouter.ai/api/v1")
                (build)))

(defn ^ResponseCreateParams$Builder
  newParamsBuilder []  (.. ResponseCreateParams
      (builder)
      (model "openai/gpt-4o-mini")
      (temperature 0.0)
      (reasoning (.. Reasoning (builder) (effort ReasoningEffort/MEDIUM) (summary Reasoning$Summary/CONCISE) (build)))
      ))

(defn ^ResponseInputItem user-prompt [^String input]
  (ResponseInputItem/ofMessage
    (.. ResponseInputItem$Message (builder)
        (addInputTextContent input)
        (role ResponseInputItem$Message$Role/USER)
        (build))))

(defn ^JsonValue jv
  "Pithier creation of JsonValue from arbitrary object."
  [x] (JsonValue/from x))

(defn fn-parameters
  [& params]
  "Create the function parameter object from param1 val1 param2 val2 ..."
  (-> (FunctionTool$Parameters/builder)
                               (#(reduce (fn [bld [name value]]
                                           (.putAdditionalProperty bld name (jv value))) % (partition 2 params)))
                               (#(.putAdditionalProperty % "additionalProperties" (jv false)))
                               (.build)))

(defn tool [name desc & args]
  "Create a tool. Each argument is a triplet vector [arg-name type description]."
  (Tool/ofFunction
    (.. (FunctionTool/builder)
        (name name)
        (description desc)
        (parameters  (fn-parameters
                      "type" (jv "object")
                      ; Build the function argument map
                      "properties" (jv (reduce (fn [props [nme tpe desc]]
                                                 (assoc props nme {"type" tpe "description" desc})) {} args))
                      "required" (jv (map first args))
                      ))
        (strict true)
        (build))))

(def call-function-tool (tool "call-function" "calls a function" ["x" "number" "blah"]))

(defn dispatch [^ResponseFunctionToolCall fn]
  (case (.name fn)
    "call-function" (let [args (jv (.readTree (ObjectMappers/jsonMapper) (.arguments fn)))
                          x (double (-> args (.values) (.get "x") (.value)))]
                     (println "Evaluating for " x)
                     (json/write-str { :x (Math/sin x)})
                     )))

(defn oget [^Optional opt]
  "Optional -> truthy."
  (if (.isPresent opt) (.get opt) nil))
(defn from-java-coll [java-collection] (into [] java-collection))
(defn flat-mop [opt-coll] (filter some? (map oget opt-coll)))
(defn not-empty? [coll] (not (empty? coll)))


(defn process [initial-prommpt tools]
  (loop [context [(user-prompt initial-prommpt)]]
    (let [builder (.input (reduce #(.addTool %1 %2) (newParamsBuilder) tools)
                          (ResponseCreateParams$Input/ofResponse context))
          _ (println "Thinking ...")
          ^Response response (-> client (.responses) (.create (.build builder)))
          output-items (.output response)
          reasoning-ctx (->> output-items (map #(.reasoning %)) (flat-mop) (map ResponseInputItem/ofReasoning) )
          function-calls (->> output-items (map #(.functionCall %)) (flat-mop))
          function-ctx (flatten (map (fn [fc]
                                       [(ResponseInputItem/ofFunctionCall fc) ; interleave function calls and output
                                        (ResponseInputItem/ofFunctionCallOutput
                                          (.. (ResponseInputItem$FunctionCallOutput/builder)
                                              (callId (.callId fc))
                                              (output (dispatch fc))
                                              (build)))]
                                       ) function-calls))
          output-messages  (->> output-items (map #(.message %)) (flat-mop))
          msg-context (map ResponseInputItem/ofResponseOutputMessage output-messages)
          to-print (->> output-messages (map #(from-java-coll (.content %))) (flatten)
                        (map #(oget (.outputText %))) (filter some?) (map #(.text %))
                        )
          _ (run! println to-print)
          response (if (and (empty? function-ctx) (not-empty? to-print))
                     (do (print "--> ('stop' to stop) ") (flush) (read-line))  "")
          input-ctx (if (or (empty? response) (= "stop" response)) [] [(user-prompt response)])
          ]
      (if (or (not-empty? input-ctx) (not-empty? function-ctx))
        (recur (concat context reasoning-ctx function-ctx msg-context input-ctx)))
      )))

(defn go []
       (process "Figure out what function is implemented by the call-function tool, by testing it for multiple values", [call-function-tool]))
