(ns clajent.core
  (:require    [clojure.data.json :as json]
               [clajent.keys])
  (:import [com.openai.client.okhttp OpenAIOkHttpClient]
           [com.openai.core JsonValue ObjectMappers]
           (com.openai.errors BadRequestException)
           [com.openai.models.responses Response ResponseCreateParams ResponseCreateParams$Builder ResponseCreateParams$Input ResponseFunctionToolCall
                                        ResponseInputItem ResponseInputItem$FunctionCallOutput ResponseInputItem$Message ResponseInputItem$Message$Role Tool FunctionTool FunctionTool$Parameters]
           [com.openai.core ObjectMappers JsonValue]
           [com.openai.models Reasoning Reasoning$Summary Reasoning$Summary$Companion ReasoningEffort]
           (java.util Optional)
           )
  )

(defn oget [^Optional opt]
  "Optional -> truthy."
  (if (.isPresent opt) (.get opt) nil))
(defn from-java-coll [java-collection] (into [] java-collection))
(defn flat-mop [opt-coll] (filter some? (map oget opt-coll)))
(defn not-empty? [coll] (not (empty? coll)))
(defrecord FT [name params impl tool])
(defrecord Param [name tpe description])



(def client (.. OpenAIOkHttpClient (builder)
                (apiKey (clajent.keys/get "OPEN_ROUTER_KEY"))
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

(defn function-tool [fname f desc &
            [ arg-name arg-tpe arg-desc & more-args]]
  "Create a tool. Each argument is a triplet vector [arg-name type description]."
  (let [
        params (map #(apply ->Param %)(partition 3 (concat (and arg-name [arg-name arg-tpe arg-desc]) more-args)))
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
    ))

(defn to-tool-box [& fts]
  (reduce (fn [h ft] (assoc h (:name ft) (:tool ft)) ) {} fts)
  )

(defn mystery [x]
  (Math/sin x))

(def call-function-tool (function-tool "call-function" mystery "calls a function"
                                       "x" :number "mystery parameter"))
(def resp (atom nil))


(defn dispatch [tools  ^ResponseFunctionToolCall ftc]
  (reset! resp ftc)
  (let [arg-map (-> ftc .arguments json/read-str)
        nme (.name ftc)
        ft (some #(if (= (:name %) nme) %)  tools)
        {params :params fn :impl} ft
        args (map #(get arg-map (:name %)) params)
        _ (println "Evaluating:" nme args)
        res (apply fn args)
        _ (println "   -->" res)
        ]
    (json/write-str res)
    )
  )


(defn process [initial-prompt tools]
  (loop [context [(user-prompt initial-prompt)]]
    (let [builder (.input (reduce #(.addTool %1 %2)
                                  (newParamsBuilder)
                                  (map :tool tools))
                          (ResponseCreateParams$Input/ofResponse context))
          _ (println "Thinking ...")
          ^Response response (try (-> client (.responses) (.create (.build builder))) (catch BadRequestException e
                                                                                        (println "Error" (.body e))
                                                                                        (throw e)
                                                                                        ))
          output-items (.output response)
          reasoning-ctx (->> output-items (map #(.reasoning %)) (flat-mop) (map ResponseInputItem/ofReasoning) )
          function-calls (->> output-items (map #(.functionCall %)) (flat-mop))
          function-ctx (flatten (map (fn [fc]
                                       [(ResponseInputItem/ofFunctionCall fc) ; interleave function calls and output
                                        (ResponseInputItem/ofFunctionCallOutput
                                          (.. (ResponseInputItem$FunctionCallOutput/builder)
                                              (callId (.callId fc))
                                              (output (dispatch tools fc))
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
      ))
  (println "Done")
  )

(def tools [call-function-tool])

(defn do-it [] (process "Figure out what the mystery tool does" tools))

(do-it)