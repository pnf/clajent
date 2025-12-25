(ns clajent.homeassistant
  (:require [clj-http.client :as http]
            [clojure.data.json :as json]
            [clojure.string :as str]))

(declare home-assistant-token)

(try
  (load "secrets")
  (catch Exception _
    (def home-assistant-token (System/getenv "HA_TOKEN"))))

(def ha-url "http://homeassistant.local:8123")

(defn ha-headers
  "Generate headers for Home Assistant API requests"
  [token]
  {"Authorization" (str "Bearer " token)
   "Content-Type" "application/json"})

(defn ha-get
  "Make a GET request to Home Assistant API"
  [endpoint & {:keys [token url]
               :or {token home-assistant-token
                    url ha-url}}]
  (try
    (let [response (http/get (str url "/api/" endpoint)
                            {:headers (ha-headers token)
                             :as :json})]
      (:body response))
    (catch Exception e
      {:error (.getMessage e)})))

(defn ha-post
  "Make a POST request to Home Assistant API"
  [endpoint body & {:keys [token url]
                    :or {token home-assistant-token
                         url ha-url}}]
  (try
    (let [response (http/post (str url "/api/" endpoint)
                             {:headers (ha-headers token)
                              :body (json/write-str body)
                              :as :json})]
      (:body response))
    (catch Exception e
      {:error (.getMessage e)})))

(defn get-states
  "Get all entity states from Home Assistant"
  []
  (ha-get "states"))

(defn get-entity-state
  "Get state of a specific entity"
  [entity-id]
  (ha-get (str "states/" entity-id)))

(defn get-services
  "Get all available services"
  []
  (ha-get "services"))

(defn call-service
  "Call a Home Assistant service"
  [domain service entity-id & [service-data]]
  (ha-post (str "services/" domain "/" service)
           (merge {:entity_id entity-id}
                  service-data)))

(defn get-sonos-entities
  "Get all Sonos media player entities"
  []
  (let [states (get-states)]
    (filter #(and (str/starts-with? (:entity_id %) "media_player.")
                  (str/includes? (:entity_id %) "sonos"))
            states)))

(defn get-sonos-sources
  "Get available sources for a Sonos entity"
  [entity-id]
  (let [state (get-entity-state entity-id)]
    (get-in state [:attributes :source_list])))

(defn browse-media
  "Browse media library for a media player entity.
   Returns the media browser structure with available media content."
  [entity-id & {:keys [media-content-type media-content-id token url]
                :or {token home-assistant-token
                     url ha-url}}]
  (try
    (let [body (cond-> {:entity_id entity-id}
                 media-content-type (assoc :media_content_type media-content-type)
                 media-content-id (assoc :media_content_id media-content-id))
          response (http/post (str url "/api/services/media_player/browse_media?return_response=true")
                             {:headers (ha-headers token)
                              :body (json/write-str body)
                              :content-type :json
                              :as :json})
          entity-keyword (keyword entity-id)]
      (get-in response [:body :service_response entity-keyword]))
    (catch Exception e
      {:error (.getMessage e)
       :data (ex-data e)})))

(defn browse-sonos-library
  "Browse the Sonos music library for a given Sonos entity"
  [entity-id]
  (browse-media entity-id))

(defn browse-sonos-favorites
  "Browse Sonos favorites"
  [entity-id]
  (browse-media entity-id :media-content-type "favorites" :media-content-id ""))

(defn browse-sonos-albums
  "Browse Sonos albums"
  [entity-id]
  (browse-media entity-id :media-content-type "album" :media-content-id ""))

(defn browse-sonos-artists
  "Browse Sonos artists"
  [entity-id]
  (browse-media entity-id :media-content-type "artist" :media-content-id ""))

(defn browse-sonos-playlists
  "Browse Sonos playlists"
  [entity-id]
  (browse-media entity-id :media-content-type "playlist" :media-content-id ""))

(defn play-media
  "Play media on a Sonos device"
  [entity-id media-content-id media-content-type]
  (call-service "media_player" "play_media" entity-id
                {:media_content_id media-content-id
                 :media_content_type media-content-type}))