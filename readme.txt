start germanet mongod server:
- mongod --port 27027 --dbpath ./mongodb

start coreNLP server:
german:
java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-german.properties -port 9000 -timeout 15000 -quiet

english:
java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9500 -timeout 15000 -quiet
