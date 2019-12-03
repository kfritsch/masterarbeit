1) Embeddings: download fastext embeddings and put them here (https://fasttext.cc/docs/en/crawl-vectors.html):
  put it in /embeddings/<lang>/fasttext_w2w.txt
  use saveEmbeddinigsAsLMDB in SemSim for faster access using LMDB

2) GermaNet:
  get germanet data from http://www.sfs.uni-tuebingen.de/GermaNet/
  run the setup from pygermanet https://github.com/wroberts/pygermanet
  get sdewac-gn-words.tsv from pygermanet https://github.com/wroberts/pygermanet

3) Wordnet:
  comes with nltk

3) Tiger Lemma Dict:
  get data from https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger/
  use Germalemmatizer.save_to_pickle to create a lemma mapping

4) IWNLP Lemma Dict:
  download the dictionary from https://www.iwnlp.com/

5) CoreNlp:
  install java 8
  follow setup instructions from https://stanfordnlp.github.io/CoreNLP/download.html
  also get german model

6) RFTagger:
  install java 8
  get rftagger from here https://www.cis.uni-muenchen.de/~schmid/tools/RFTagger/

7) SemSim Tests: download semantic similartiy datasets from https://www.informatik.tu-darmstadt.de/ukp/research_6/data/index.en.jsp
