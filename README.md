# NNTranSegmentor
Undergraduate graduation project ---- Chinese Word Segmentation for Weibo text  
2020本科毕业设计 ---- 面向微博文本的中文分词

### Usage
- Preprocess  
  Build vocab and insts from corpus, and save them to file`o`. Details see `preprocess.py`.
  ```bash
  python ./preprocess.py -o ./data/pku/pku.vocab.data.bin --train ./data/pku/train.pku.hwc.seg --dev ./data/pku/dev.pku.hwc.seg --test ./data/pku/test.pku.hwc.seg
  ```
  - `o` (str) ------ path of vocab and insts built by `preprocess.py`.
  - `train` (str) ------ path of train text.
  - `dev` (str) ------ path of dev text.
  - `test` (str) ------ path of test text.
 
### Note And Part to be improved
- pretrained embeddings data should includes `oovKey` and `padKey`.