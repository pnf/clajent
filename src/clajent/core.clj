(ns clajent.core
  (:require    [clojure.data.json :as json]
               [clajent.keys])
  (:import (com.openai.client OpenAIClient)
           [com.openai.client.okhttp OpenAIOkHttpClient]
           [com.openai.core JsonValue ObjectMappers]
           (com.openai.errors BadRequestException)
           [com.openai.models.responses Response ResponseCreateParams ResponseCreateParams$Builder ResponseCreateParams$Input ResponseFunctionToolCall
                                        ResponseInputItem ResponseInputItem$FunctionCallOutput ResponseInputItem$Message ResponseInputItem$Message$Role Tool FunctionTool FunctionTool$Parameters]
           [com.openai.core ObjectMappers JsonValue]
           [com.openai.models Reasoning Reasoning$Summary Reasoning$Summary$Companion ReasoningEffort]
           (java.util Optional)
           )
  )

(defn- oget [^Optional opt]
  "Optional -> truthy."
  (if (.isPresent opt) (.get opt) nil))
(defn- from-java-coll [java-collection] (into [] java-collection))
(defn- flat-mop [opt-coll] (filter some? (map oget opt-coll)))
(defn- not-empty? [coll] (not (empty? coll)))

;; Record holding an openapi tool object, along with its implementation
(defrecord FT [name params impl tool-object])
;; Defines a parameter for a function-tool
(defrecord Param [name tpe description])

(def ^OpenAIClient client (.. OpenAIOkHttpClient (builder)
                              (apiKey (clajent.keys/get "OPEN_ROUTER_KEY"))
                              (baseUrl "https://openrouter.ai/api/v1")
                              (build)))

(defn- ^ResponseCreateParams$Builder
  newParamsBuilder [mod]  (.. ResponseCreateParams
      (builder)
      (model mod)
      (temperature 0.0)
      (reasoning (.. Reasoning (builder) (effort ReasoningEffort/MEDIUM) (summary Reasoning$Summary/CONCISE) (build)))
      ))

(defn ^ResponseInputItem user-prompt [^String input]
  (ResponseInputItem/ofMessage
    (.. ResponseInputItem$Message (builder)
        (addInputTextContent input)
        (role ResponseInputItem$Message$Role/USER)
        (build))))

(defn- ^JsonValue jv
  "Pithier creation of JsonValue from arbitrary object."
  [x] (JsonValue/from x))

(defn- fn-parameters
  [& params]
  "Create the function parameter object from param1 val1 param2 val2 ..."
  (-> (FunctionTool$Parameters/builder)
                               (#(reduce (fn [bld [name value]]
                                           (.putAdditionalProperty bld name (jv value))) % (partition 2 params)))
                               (#(.putAdditionalProperty % "additionalProperties" (jv false)))
                               (.build)))

(defn function-tool
  "Define a function tool record with zero or more arguments.
  fname - Unique name by which model will request a function call
  f - actual implementation, taking zero or more arguments and returning something that can be turned into
      valid json via write-str
  desc - full description, with enough information for the model to determine when to use the tool
  Arguments are specified as name1 type1 desc1 name2 type2 desc2 ...
  Type is a valid json type expressed as a symbol, i.e. :string, :number, :integer, :object, :array, :boolean"
  ([fname f desc] (function-tool fname f desc []))
  ([fname f desc arg-name arg-tpe arg-desc & more]
   (function-tool fname f desc (map #(apply ->Param %)(partition 3 (concat [arg-name arg-tpe arg-desc] more)))))
  ([fname f desc params]
   (let [
         ft (FT. fname  params f
                 (Tool/ofFunction
                   (.. (FunctionTool/builder)
                       (name fname)
                       (description desc)
                       (parameters (fn-parameters
                                     "type" (jv "object")
                                     ; Build the function argument map
                                     "properties" (jv (reduce (fn [props {nme :name tpe :tpe desc :description }]
                                                                (assoc props nme {"type" (name tpe) "description" desc}))
                                                              {} params))
                                     "required" (jv (map :name params))
                                     ))
                       (strict true)
                       (build))) )
         ]
     ft
     ) )
  )

; Useful for capturing a complex response object to decode in the repl
(def resp (atom nil))

(defn- dispatch [function-tools  ^ResponseFunctionToolCall ftc]
  ; Dispatch a function tool call from the model to the implementation function
  ;(reset! resp ftc)
  (let [arg-map (-> ftc .arguments json/read-str)           ; extract map of parm name to arg value
        nme (.name ftc)                                     ;
        ft (some #(if (= (:name %) nme) %)  function-tools) ; lookup the function tool by name
        {params :params fn :impl} ft                        ; extract implementation and param meta data
        args (map #(get arg-map (:name %)) params)          ; get arguments in correct order for the implementation
        _ (println "Evaluating:" nme args)
        res (apply fn args)                                 ; actually call the function
        _ (println "   -->" res)
        ]
    (json/write-str res)                                    ; send it back as json
    )
  )

; Agentic loop
(defn process [initial-prompt function-tools interactive & {:keys [:max-iterations max-iteration
                                                                   :model model]}]
  (loop [context [(user-prompt (str initial-prompt "\nWhen you are done, end your output with the string \"__DONE__\""))]
         iter (or max-iteration 10)
         ]
    (let [^ResponseCreateParams$Builder builder
          (.input (reduce #(.addTool %1 %2)         ;Create new params builder and iteratively add tools to it
                          (newParamsBuilder (or model "openai/gpt-4o-mini"))
                          (map :tool-object function-tools))
                  (ResponseCreateParams$Input/ofResponse context))
          _ (println "Thinking ...")
          ^Response response (try (-> client
                                      (.responses)          ; get response service from the client
                                      (.create (.build builder)) ; build params and send to service
                                      )
                                  (catch BadRequestException e
                                    (println "Error" (.body e))
                                    (throw e)
                                    ))

          output-items (.output response)
          ;; Construct context items for the next round, based on this response
          reasoning-ctx (->> output-items (map #(.reasoning %)) (flat-mop) (map ResponseInputItem/ofReasoning) )
          function-calls (->> output-items (map #(.functionCall %)) (flat-mop))
          ;; make all requested function calls, and interleave the call objects with corresponding responses
          function-ctx (flatten (map (fn [fc]
                                       [ ; function call request as valid input for the next round
                                        (ResponseInputItem/ofFunctionCall fc)
                                        ; function call output as valid input for the next round
                                        (ResponseInputItem/ofFunctionCallOutput
                                          (.. (ResponseInputItem$FunctionCallOutput/builder)
                                              (callId (.callId fc))
                                              (output (dispatch function-tools fc)) ;; actually dispatch call
                                              (build)))]
                                       ) function-calls))
          _ (reset! resp function-ctx)
          output-message-ctx  (->> output-items (map #(.message %)) (flat-mop))
          msg-context (map ResponseInputItem/ofResponseOutputMessage output-message-ctx)
          to-print (concat
                     (->> output-message-ctx (map #(from-java-coll (.content %))) (flatten)
                         (map #(oget (.outputText %))) (filter some?) (map #(.text %))
                         ))
          _ (run! println to-print)
          response (if (and interactive (empty? function-ctx) (not-empty? to-print))
                     (do (print "--> ('stop' to stop) ") (flush) (read-line))  "")
          input-ctx (if (or (empty? response) (= "stop" response)) [] [(user-prompt response)])
          ]
      (if (and
            (not (some #(.contains % "__END__") to-print))
            (pos? iter)
            (or (not-empty? input-ctx) (not-empty? function-ctx) (not-empty? reasoning-ctx)))
        (recur (concat context reasoning-ctx function-ctx msg-context input-ctx) (dec iter)))
      ))
  (println "Done")
  )

