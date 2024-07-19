import tensorflow as tf
from bert import modeling
import pyhocon
from bert import tokenization
import coref_ops
import math

import optimization

import util

class CorefModel(object):

  def __init__(self, config):
    self.config = config
    self.max_segment_len = config['max_segment_len']
    self.max_span_width = config["max_span_width"]
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
    self.subtoken_maps = {}
    self.gold = {}
    self.eval_data = None # Load eval data lazily.
    self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
    self.tokenizer = tokenization.FullTokenizer(
                vocab_file=config['vocab_file'], do_lower_case=False)

    input_props = [
        (tf.int32, [None, None]),
        (tf.int32, [None, None]),
        (tf.int32, [None]),
        (tf.int32, [None, None]),
        (tf.int32, []),
        (tf.bool, []),
        (tf.int32, [None]),
        (tf.int32, [None]),
        (tf.int32, [None]),
        (tf.int32, [None])
    ]

  
    # Function to generate initial input tensors filled with zeros
    def generate_initial_input_tensors():
        for dtype, shape in input_props:
            yield tf.zeros([1 if dim is None else dim for dim in shape], dtype=dtype)

    # Create an initial dummy dataset
    initial_tensors = list(generate_initial_input_tensors())
    dataset = tf.data.Dataset.from_tensors(tuple(initial_tensors)).repeat()
    self.predictions, self.loss = self.get_predictions_and_loss(*initial_tensors)
    # bert stuff
    tvars = tf.compat.v1.trainable_variables()

    # If you're using TF weights only, tf_checkpoint and init_checkpoint can be the same
    # Get the assignment map from the tensorflow checkpoint. Depending on the extension, use TF/Pytorch to load weights.
    assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config['tf_checkpoint'])
    init_from_checkpoint = tf.compat.v1.train.init_from_checkpoint if config['init_checkpoint'].endswith('ckpt') else load_from_pytorch_checkpoint
    init_from_checkpoint(config['init_checkpoint'], assignment_map)
    print("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      # init_string)
      print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

    num_train_steps = int(
                    self.config['num_docs'] * self.config['num_epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    self.global_step = tf.compat.v1.train.get_or_create_global_step()
    self.train_op = optimization.create_custom_optimizer(tvars,
                      self.loss, self.config['bert_learning_rate'], self.config['task_learning_rate'],
                      num_train_steps, num_warmup_steps, False, self.global_step, freeze=-1,
                      task_opt=self.config['task_optimizer'], eps=config['adam_eps'])
    
    

    #print(initial_tensors)

  def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map):
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False,
            scope='bert')
        all_encoder_layers = model.get_all_encoder_layers()
        mention_doc = model.get_sequence_output()

        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

        num_sentences = tf.shape(mention_doc)[0]
        max_sentence_length = tf.shape(mention_doc)[1]
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)
        num_words = util.shape(mention_doc, 0)
        antecedent_doc = mention_doc

        flattened_sentence_indices = sentence_map
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]
        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]), flattened_candidate_mask) # [num_candidates]

        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids) # [num_candidates]

        candidate_span_emb = self.get_span_emb(mention_doc, mention_doc, candidate_starts, candidate_ends) # [num_candidates, emb]
        candidate_mention_scores =  self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

        # beam size
        k = tf.minimum(3900, tf.cast(tf.floor(tf.cast(num_words, tf.float32) * self.config["top_span_ratio"]), tf.int32))
        c = tf.minimum(self.config["max_top_antecedents"], k)
        # pull from beam
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                               tf.expand_dims(candidate_starts, 0),
                                               tf.expand_dims(candidate_ends, 0),
                                               tf.expand_dims(k, 0),
                                               num_words,
                                               True) # [1, k]

        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
        genre_emb = tf.gather(tf.Variable(initial_value=tf.keras.initializers.TruncatedNormal(stddev=0.02)(shape=[len(self.genres), self.config["feature_size"]]),
                              name="genre_embeddings"), genre) # [emb]

        if self.config['use_metadata']:
            speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)
            top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]i
        else:
            top_span_speaker_ids = None

        dummy_scores = tf.zeros([k, 1]) # [k, 1]
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
        num_segs, seg_len = util.shape(input_ids, 0), util.shape(input_ids, 1)
        word_segments = tf.tile(tf.expand_dims(tf.range(0, num_segs), 1), [1, seg_len])
        flat_word_segments = tf.boolean_mask(tf.reshape(word_segments, [-1]), tf.reshape(input_mask, [-1]))
        mention_segments = tf.expand_dims(tf.gather(flat_word_segments, top_span_starts), 1) # [k, 1]
        antecedent_segments = tf.gather(flat_word_segments, tf.gather(top_span_starts, top_antecedents)) #[k, c]
        segment_distance = tf.clip_by_value(mention_segments - antecedent_segments, 0, self.config['max_training_sentences'] - 1) if self.config['use_segment_distance'] else None #[k, c]


        if self.config['fine_grained']:
          for i in range(self.config["coref_depth"]):
            with tf.compat.v1.variable_scope("coref_layer", reuse=(i > 0)):
              top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]
              top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb, segment_distance) # [k, c]
              top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
              top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
              attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1) # [k, emb]
              with tf.name_scope("f"):
                f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1), util.shape(top_span_emb, -1))) # [k, emb]
                top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb # [k, emb]
        else:
           top_antecedent_scores = top_fast_antecedent_scores

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]

        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]
        top_antecedent_cluster_ids += tf.cast(tf.math.log(tf.cast(top_antecedents_mask, tf.float32)), tf.int32) # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]
        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]
        loss = tf.reduce_sum(loss) # []

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores], loss

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.math.log(tf.cast(antecedent_labels, tf.float32)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb, segment_distance=None):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
      speaker_pair_emb = tf.gather(tf.compat.v1.get_variable("same_speaker_emb", [2, self.config["feature_size"]], initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)), tf.cast(same_speaker, tf.int32)) # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.compat.v1.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]], initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)
    if segment_distance is not None:
      with tf.compat.v1.variable_scope('segment_distance', reuse=tf.compat.v1.AUTO_REUSE):
        segment_distance_emb = tf.gather(tf.compat.v1.get_variable("segment_distance_embeddings", [self.config['max_training_sentences'], self.config["feature_size"]], initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)), segment_distance) # [k, emb]
      feature_emb_list.append(segment_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

    with tf.compat.v1.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]
    return slow_antecedent_scores # [k, c]


  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
    fast_antecedent_scores += tf.math.log(tf.cast(antecedents_mask, tf.float32)) # [k, k]
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [k, k]
    if self.config['use_prior']:
      antecedent_distance_buckets = self.bucket_distance(antecedent_offsets) # [k, c]
      initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
      distance_scores = util.projection(
        tf.nn.dropout(tf.Variable(initializer([10, self.config["feature_size"]]), "antecedent_distance_emb"), self.dropout),1, initializer=initializer) #[10, 1]
      antecedent_distance_scores = tf.gather(tf.squeeze(distance_scores, 1), antecedent_distance_buckets) # [k, c]
      fast_antecedent_scores += antecedent_distance_scores

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.cast(tf.math.floor(tf.math.log(tf.cast(distances, tf.float32))/math.log(2)), tf.int32) + 3
    use_identity = tf.cast(distances <= 4, tf.int32)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_fast_antecedent_scores(self, top_span_emb):
    with tf.name_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts # [k]

        if self.config["use_features"]:
            span_width_index = span_width - 1 # [k]

            max_span_width = self.config["max_span_width"]
            feature_size = self.config["feature_size"]

            # Define the span width embeddings variable
            initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
            span_width_embeddings = tf.Variable(
                                      initializer(shape=[max_span_width, feature_size], dtype=tf.float32),
                                      name="span_width_embeddings")
            # Use tf.gather to gather the embeddings based on span_width_index
            span_width_emb = tf.gather(span_width_embeddings, span_width_index)  # [k, emb]

            #span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index) # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_reps = tf.matmul(mention_word_scores, context_outputs) # [K, T]
            span_emb_list.append(head_attn_reps)

        span_emb = tf.concat(span_emb_list, 1) # [k, emb]
        return span_emb # [k, emb]  

  def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
      num_words = util.shape(encoded_doc, 0) # T
      num_c = util.shape(span_starts, 0) # NC
      doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1]) # [K, T]
      mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1), doc_range <= tf.expand_dims(span_ends, 1)) #[K, T]

      initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
      with tf.name_scope("mention_word_attn"):
        word_attn = tf.squeeze(util.projection(encoded_doc, 1, initializer=initializer), 1)

      mention_word_attn = tf.nn.softmax(tf.math.log(tf.cast(mention_mask,tf.float32)) + tf.expand_dims(word_attn, 0))
      return mention_word_attn

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):

    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.cast(same_span, tf.int32)) # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels


  def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank  == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
           raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.cast(is_training, tf.float32) * dropout_rate)

  def get_mention_scores(self, span_emb, span_starts, span_ends):

      with tf.name_scope("mention_scores"):
        span_scores = util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]
      if self.config['use_prior']:
        span_width_emb = tf.Variable(
            initial_value=tf.keras.initializers.TruncatedNormal(stddev=0.02)(shape=[self.config["max_span_width"], self.config["feature_size"]]),
            name="span_width_prior_embeddings"
        )
        span_width_index = span_ends - span_starts # [NC]
        with tf.name_scope("width_scores"):
          width_scores =  util.ffnn(span_width_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [W, 1]
        width_scores = tf.gather(width_scores, span_width_index)
        span_scores += width_scores
      return span_scores

    

#[<KerasTensor: shape=(None, None, None) dtype=int32 (created by layer 'input_1')>, 
#<KerasTensor: shape=(None, None, None) dtype=int32 (created by layer 'input_2')>, 
#<KerasTensor: shape=(None, None) dtype=int32 (created by layer 'input_3')>, 
#<KerasTensor: shape=(None, None, None) dtype=int32 (created by layer 'input_4')>, 
#<KerasTensor: shape=(None,) dtype=int32 (created by layer 'input_5')>, 
#<KerasTensor: shape=(None,) dtype=bool (created by layer 'input_6')>, 
#<KerasTensor: shape=(None, None) dtype=int32 (created by layer 'input_7')>, 
#<KerasTensor: shape=(None, None) dtype=int32 (created by layer 'input_8')>, 
#<KerasTensor: shape=(None, None) dtype=int32 (created by layer 'input_9')>, 
#<KerasTensor: shape=(None, None) dtype=int32 (created by layer 'input_10')>]
if __name__ == "__main__":
    name = "bert_base"
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
    model = CorefModel(config)