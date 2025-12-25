(require '[clajent.homeassistant :as ha])

;; Test getting the entity state
(println "=== Testing entity state ===")
(def entity-state (ha/get-entity-state "media_player.chambre_principale"))
(println "Entity ID:" (:entity_id entity-state))
(println "State:" (:state entity-state))
(println "Friendly name:" (get-in entity-state [:attributes :friendly_name]))

;; Test browsing media at root level
(println "\n=== Testing browse media (root) ===")
(def media-result (ha/browse-media "media_player.chambre_principale"))
(println "Title:" (:title media-result))

;; If successful, print the children (available media)
(when-not (:error media-result)
  (println "\n=== Available media items ===")
  (doseq [item (:children media-result)]
    (println "  -" (:title item) "[" (:media_content_type item) "]")))

;; Browse into Music Library
(println "\n=== Testing browse Music Library ===")
(def library-result (ha/browse-media "media_player.chambre_principale"
                                      :media-content-type "library"
                                      :media-content-id ""))
(when-not (:error library-result)
  (println "Music Library categories:")
  (doseq [category (:children library-result)]
    (println "  -" (:title category))))