(defproject clajent "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "https://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.12.2"]
                 [net.clojars.wkok/openai-clojure "0.23.0"]
                 [com.openai/openai-java "4.8.0"]
                 [org.clojure/data.json "2.5.1"]
                 [clj-http "3.13.0"]
                 ]
  :repl-options {:init-ns clajent.core})
