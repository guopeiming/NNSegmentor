# Config

- Function
  - config file and constants file.
  - Details see `config.py` and `config.cfg`, please.
  
- Following is `config.cfg` and `Constants.py` Parameter Details.

## config.cfg
- [Preprocess]
  - `char_min_fre` (integer) ------ The smallest character frequency when build vocab. A char will be assigned to OOV if its
  appearing times smaller than `char_min_fre`.
  - `word_min_fre` (integer) ------ The smallest word frequency when build vocab. A word will be assigned to OOV if its
  appearing times smaller than `word_min_fre`.
  - `bichar_min_char` (integer) ------ The smallest bichar frequency when build vocab. A bichar will be assigned to OOV if its
  appearing times smaller than `bichar_min_fre`.
  
- [Data]
  - `data_path` (path) ------ path of data that `preprocess.py` outputs.
  
- [Embed]
  - `pretrained_embed_char` (False or True) ------ whether to use pretrained embeddings.
  - `pretrained_embed_bichar` (False or True) ------ whether to use pretrained embeddings.
  - `pretrained_char_embed_file` (path) ------ pretrained char embeddings path.
  - `pretrained_bichar_embed_file` (path) ------ pretrained bichar embeddings path.
  - `char_gen_oov_mode` (zeros/avg/nnembed/uniform/randn) ------ mode for generating OOV embedding vector of char.
  - `bichar_gen_oov_mode` (zeros/avg/nnembed/uniform/randn) ------ mode for generating OOV embedding vector of bichar.
  - `char_gen_oov_uniform` (float) ------ parameter of uniform distribution to generate char oov embedding vector, which
  is valid when `char_gen_oov_uniform` is `uniform`.
  - `bichar_gen_oov_uniform` (float) ------ parameter of uniform distribution to generate bichar oov embedding vector, which
  is valid when `bichar_gen_oov_mode` is `uniform`.

- [Train]
  - `use_cuda` (True or False) ------ use `cuda` speed up.
  - `batch_size` (integer) ------ how many insts per batch to load.
  - `shuffle` (True or False) ------ set to True to have the data reshuffled at every epoch.
  - `num_worders` (integer) ------ how many subprocesses to use for data loading. 0 means that the data will be loaded 
  in the main process.
  - `drop_last` (True or False) ------ set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
  If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. 
  - `epoch` (integer) ------ train epoch.
  - `fine_tune` (False or True) ------ whether to fine tune embedding.
  - `savepath` (path) ------ path for saving model.
  - `visual_logger_path` (path) ------ path for visual logger.

- [Model]
  - `char_embed_dim` (integer) ------ char embeddings vector dim, remember to alter it when you use pretrained char embeddings.
  - `bichar_embed_dim` (integer) ------ bichar embeddings vector dim, remember to alter it when you use pretrained word embeddings.
  - `dropout_embed` (float) ------ dropout rate on char and bichar embeddings.
  - `char_embed_max_norm` (float) ------ max norm used in char embeddings layer(nn.Embedding), which is None if set to 0.0.
  - `bichar_embed_max_norm` (float) ------ max norm used in bichar embeddings layer(nn.Embedding), which is None if set to 0.0.
  - `encoder_embed_dim` (integer) ------ char_encoder embeddings dim of composing static, no_static, char and bichar embeddings.
  - `dropout_encoder_embed` (float) ------ dropout rate on encoder embeddings.
  - `encoder_lstm_hid_size` (integer) ------ hidden state dimension of encoder_LSTM.
  - `dropout_encoder_hid` (float) ------ dropout rate on output of encoder.
  - `subword_lstm_hid_size` (integer) ------ hidden state dimension of subword_LSTM.
  - `word_lstm_hid_size` (integer) ------ hidden state dimension of word_LSTM.

- [Optimizer]
  - `name` (string) ------ which optimizer to use.
  - `learning_rate` (float) ------ learning rate of optimizer.
  - `weight_decay` (float) ------ lamda

  
## Constants.py
- `oovKey` (string) ------ key of oov in vocab. You can not change it.
- `oovId` (integer) ------ id of oov in vocab. You can not change it.
- `padKey` (string) ------ key of pad in vocab. You can not change it.
- `padId` (integer) ------ id of pad in vocab. You can not change it.
- `APP` (integer) ------ digital representation of action append. You can not change it.
- `SEP` (integer) ------ digital representation of action separate. You can not change it.
- `actionPadId` (integer) ------ id of action pad.
- `EPSILON` (float) ------ threshold for subtraction comparision.
- `TEST_OOV_NUM` (integer) ------ the number for testing the number of oov in pretrained embeddings.