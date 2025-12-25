(require '[clajent.homeassistant :as ha])
(require '[clojure.pprint :as pp])

;; Check what services are available
(println "=== Checking media_player services ===")
(def services (ha/get-services))
(def media-services (->> services
                        (filter #(= (:domain %) "media_player"))
                        first
                        :services))
(println "\nAvailable media_player services:")
(pp/pprint (keys media-services))