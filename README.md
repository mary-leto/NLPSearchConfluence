# О проекте
Этот проект представляет собой чат-бота для взаимодействия с пользователями на основе информации из базы знаний Confluence с помощью инструментов NLP.
Библиотека состоит из ~1000 статей. Бот обрабатывает запросы на естественном языке и предоставляет пользователю ссылку на источник найденного ответа.
Мы используем библиотеку BeautifulSoup для парсинга HTML-страниц, а затем обрабатываем наш датасет через интеграцию с HuggingFace - Langchain фреймворка llamaindex, библиотеку Tensorflow и PyTorch.

Проект на стадии разработки.

## Описание файлов
parserHtmlToJson - парсинг HTML-страниц Confluence в датасет в формате JSON  
ElasticSearch - поиск по датасету с помощью ElasticSearch и RAG-модели saiga_mistral_7b_lora


## Ссылки и источники
+ _Подробнее о базе знаний [Confluence](https://www.atlassian.com/ru/software/confluence)_

+ _Подробнее об [ElasticSearch](http://stmarysguntur.com/wp-content/uploads/2019/04/1021302647.pdf) (ENG)_ 

+ _О [RAG](https://habr.com/ru/articles/779526/)_
  
+ _Наглядный [кейс](https://habr.com/ru/articles/811127/) в помощь начинающим_

## Запуск кода

**1. Устанавливаем необходимые эмбеддинги:**
```
#!pip install llama-index-embeddings-huggingface
#!pip install torch sentence-transformers
#!pip install llama_index
#!pip install tensorflow-io
#!pip install llama-index-vector-stores-chroma
```

**2. Устанавливаем актуальную версию ElasticSearch:**
```
#!pip install 'elasticsearch<7.14.0'
```

**3. Импорт необходимых библиотек:**
```
import os
import time
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
```
> /usr/local/lib/python3.10/dist-packages/sentence_transformers/cross_encoder/CrossEncoder.py:11: TqdmExperimentalWarning: Using `tqdm.autonotebook.tqdm` in notebook mode. Use `tqdm.tqdm` instead to force console mode (e.g. in jupyter console)
>  from tqdm.autonotebook import tqdm, trange

**4. Импорт библиотек и доп. модулей:**

```
# Импорт TensorFlow и TensorFlow I/O
import tensorflow as tf
import tensorflow_io as tfio

# Импорт дополнительных модулей из TensorFlow
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
```

**5. Скачивание и распаковка Elasticsearch**
```
%%bash

wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.10.0-linux-x86_64.tar.gz
wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.10.0-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-oss-7.10.0-linux-x86_64.tar.gz
sudo chown -R daemon:daemon elasticsearch-7.10.0/
shasum -a 512 -c elasticsearch-oss-7.10.0-linux-x86_64.tar.gz.sha512
```
> elasticsearch-oss-7.10.0-linux-x86_64.tar.gz: OK

**6. Запуск Elasticsearch:**

```
#Запуск Elasticsearch в фоновом режиме
%%bash --bg

sudo -H -u daemon elasticsearch-7.10.0/bin/elasticsearch
```

```
import time
time.sleep(30)
```
```
# Проверка запущенных процессов Elasticsearch
%%bash

ps -ef | grep elasticsearch
```
> root       33719   33717  0 06:28 ?        00:00:00 sudo -H -u daemon elasticsearch-7.10.0/bin/elast
daemon     33720   33719 78 06:28 ?        00:00:23 /content/elasticsearch-7.10.0/jdk/bin/java -Xsha
root       34038   34036  0 06:29 ?        00:00:00 grep elasticsearch

```
# Проверка соединения с Elasticsearch
%%bash

curl -sX GET "localhost:9200/"
```
>{
  "name" : "9dda0200f192",
  "cluster_name" : "elasticsearch",
  "cluster_uuid" : "RGWmq7oITz2ZwIdygJuRLw",
  "version" : {
    "number" : "7.10.0",
    "build_flavor" : "oss",
    "build_type" : "tar",
    "build_hash" : "51e9d6f22758d0374a0f3f5c6e8f3a7997850f96",
    "build_date" : "2020-11-09T21:30:33.964949Z",
    "build_snapshot" : false,
    "lucene_version" : "8.7.0",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  },
  "tagline" : "You Know, for Search"
}

```
# Создание объекта Elasticsearch, подключенный к локальному экземпляру Elasticsearch,
# который запущен на порту 9200 по протоколу HTTP
es = Elasticsearch(hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}])
es.ping()
```
>True


**7. Загружаем генеративную модель:**
```
# Загрузка модели для векторных представлений
embed_model = SentenceTransformer('distiluse-base-multilingual-cased')
# Загрузка модели для генерации ответов
model = 'IlyaGusev/saiga_mistral_7b_lora'
```

**8. Для корректного поиска по датасету:**
```
# Функция для генерации ответов
def generate_response(question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
```

```
# Загрузка и индексация статей
def index_articles(articles_path):
    with open(articles_path, 'r', encoding='utf-8') as file:
        articles = json.load(file)['pages']

    # Создание индекса в Elasticsearch, если он не существует
    if not es.indices.exists(index="articles"):
        es_index = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                    "link": {"type": "keyword"},
                    "text_vector": {"type": "dense_vector", "dims": 512}
                }
            }
        }
        es.indices.create(index="articles", body=es_index)

    # Индексация статей в Elasticsearch
    for article in articles:
        text_vector = embed_model.encode(article['text'])
        doc = {
            "title": article['title'],
            "text": article['text'],
            "link": article['link'],
            "text_vector": text_vector
        }
        es.index(index="articles", body=doc)
```

**9. Генерация и вывод результата:**
```
# Функция для интерактивного поиска и генерации ответов
def interactive_search():
    while True:
        inp_question = input("Пожалуйста, введите вопрос: ")

        # Вычисление векторного представления вопроса
        encode_start_time = time.time()
        question_embedding = embed_model.encode(inp_question)
        encode_end_time = time.time()

        # Лексический поиск с использованием TF-IDF
        tfidf_search = es.search(index="articles", body={
            "query": {
                "multi_match": {
                    "query": inp_question,
                    "fields": ["title", "text"],
                    "type": "best_fields"
                }
            }
        })

        # Семантический поиск с использованием векторных представлений
        sem_search = es.search(
            index="articles",
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                            "params": {"query_vector": question_embedding.tolist()}
                        }
                    }
                }
            }
        )

        # Вывод результатов поиска и времени выполнения
        print("Введенный вопрос:", inp_question)
        print(
            "Вычисление эмбеддинга заняло {:.3f} секунд, поиск по TF-IDF занял {:.3f} секунд, семантический поиск с ES занял {:.3f} секунд".format(
                encode_end_time - encode_start_time, tfidf_search["took"] / 1000, sem_search["took"] / 1000
            )
        )

        print("Результаты поиска по TF-IDF:")
        for hit in tfidf_search["hits"]["hits"][0:5]:
            print("\t{}".format(hit["_source"]["text"]))
            print("\tСсылка: {}".format(hit["_source"]["link"]))

        print("\nРезультаты семантического поиска:")
        for hit in sem_search["hits"]["hits"][0:5]:
            print("\t{}".format(hit["_source"]["text"]))
            print("\tСсылка: {}".format(hit["_source"]["link"]))

        # Генерация ответа на вопрос
        response = generate_response(inp_question)
        print("\nОтвет от модели SaigaMistral:")
        print(response)

        print("\n\n========\n")

# Основная функция
if __name__ == "__main__":
    articles_path = "/content/ConfluencePages.json"
    index_articles(articles_path)
    interactive_search()
```
