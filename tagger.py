import tensorflow as tf
import datetime

# Tageris ar modeli utml
class Tagger(object):
    def __init__(self, use_wordform_embeddings, use_wordshape, use_ngrams, output_attributes, featurefactory=None):
        self.session = None
        self.use_wordform_embeddings = use_wordform_embeddings
        self.use_wordshape = use_wordshape
        self.use_ngrams = use_ngrams
        self.output_pos = False
        self.output_tag = False
        self.output_attributes = output_attributes
        self._featurefactory = featurefactory  # vajag tādēļ, ka modelim ir svarīgi vārdnīcu izmēri utml
        self._prepare_graph()

    def __del__(self):
        if self.session:
            self.session.close()

    def _feed_dict(self, sentence, train=False, feed_output=True):
        feed_dict = dict()
        feed_dict[self.sentence_length] = [len(sentence.wordform_ids)]
        if train:
            feed_dict[self.dropout_keep_prob] = 0.5  # TODO - should be configurable
        else:
            feed_dict[self.dropout_keep_prob] = 1.0
        feed_dict[self.wordform_ids] = sentence.wordform_ids
        if self.use_wordform_embeddings:
            feed_dict[self.wordform_embeddings] = sentence.wordform_embeddings
        if self.use_ngrams:
            feed_dict[self.ngrams] = sentence.ngrams
        if self.use_wordshape:
            feed_dict[self.wordshape] = sentence.wordshape
        if feed_output:
            if self.output_attributes:
                feed_dict[self.attribute_ids] = sentence.attribute_ids
        return feed_dict

    def _prepare_graph(self):
        tf.reset_default_graph()

        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.sentence_length = tf.placeholder(tf.int64, name='sentence_length')

        with tf.name_scope('input'):
            input_vectors = []
            compressed_input_vectors = []
            if True:  # input_features.get('wordform_onehot'):
                wordform_vector_size = self._featurefactory.wordform_vector_size()
                self.wordform_ids = tf.placeholder(tf.int64, [None], name='wordform_ids')
                wordform_onehot = tf.one_hot(self.wordform_ids, wordform_vector_size, 1.0, 0.0, name='wordform_onehot')
                compressed_wordform = fully_connected_layer(wordform_onehot, self.dropout_keep_prob, 400,
                                                            'compressed_wordform')
                input_vectors.append(wordform_onehot)
                compressed_input_vectors.append(compressed_wordform)
            if self.use_wordshape:
                wordshape_vector_size = self._featurefactory.wordshape_vector_size()
                self.wordshape = tf.placeholder(tf.float32, [None, wordshape_vector_size], name='wordshape')
                input_vectors.append(self.wordshape)
                compressed_input_vectors.append(self.wordshape)
            if self.use_wordform_embeddings:
                embedding_vector_size = self._featurefactory.embedding_vector_size()
                self.wordform_embeddings = tf.placeholder(tf.float32, [None, embedding_vector_size],
                                                          name='wordform_embeddings')
                input_vectors.append(self.wordform_embeddings)
                compressed_input_vectors.append(self.wordform_embeddings)
            if self.use_ngrams:
                ngram_vector_size = self._featurefactory.ngram_vector_size()
                self.ngrams = tf.placeholder(tf.float32, [None, ngram_vector_size], name='ngram_nhot')
                input_vectors.append(self.ngrams)
                compressed_ngrams = fully_connected_layer(self.ngrams, self.dropout_keep_prob, 400, 'compressed_ngrams')
                compressed_input_vectors.append(compressed_ngrams)
            input_vector = tf.concat(1, input_vectors)
            compressed_input_vector = tf.concat(1, compressed_input_vectors)
            del input_vectors
            del compressed_input_vectors

        with tf.name_scope('gold_output'):
            gold_output_vectors = []
            if self.output_attributes:
                attribute_vector_size = self._featurefactory.attribute_vector_size()
                self.attribute_ids = tf.placeholder(tf.float32, [None, attribute_vector_size], name='attribute_nhot')
                gold_output_vectors.append(self.attribute_ids)
            gold_output_vector = tf.concat(1, gold_output_vectors)
            del gold_output_vectors
            output_vector_size = gold_output_vector.get_shape().as_list()[1]

        # layer = input_vector
        layer = compressed_input_vector
        #         layer = convolution_layer(layer, window=3, hidden_units=500, name_scope = 'convolution1')
        #         layer = fully_connected_layer(layer, self.dropout_keep_prob, 400)
        # layer = recurrent_layer(layer, self.sentence_length, self.dropout_keep_prob, 'recurrent1', rnn_hidden=200)
        #         layer = recurrent_layer(layer, self.sentence_length, self.dropout_keep_prob, 'recurrent2', rnn_hidden=300)
        #         layer = recurrent_layer(layer, self.sentence_length, self.dropout_keep_prob, 'recurrent3', rnn_hidden=300)
        #         layer = tf.concat(1, [input_vector, layer]) # wide and deep
        final_layer = layer
        final_layer_size = final_layer.get_shape().as_list()[1]

        # FIXME - softmax is not appropriate for anything other than a one-hot output
        with tf.name_scope('softmax'):
            weights = tf.Variable(tf.truncated_normal([final_layer_size, output_vector_size], stddev=0.1),
                                  name='weights')  # simple mapping from all words to all tags
            bias = tf.Variable(tf.zeros([output_vector_size]), name='bias')
            output_vector = tf.nn.softmax(tf.matmul(final_layer, weights) + bias, name='output_vector')

        regularization_coeff = 1e-7
        #         regularization_coeff = 0
        with tf.name_scope('train'):
            cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(gold_output_vector * tf.log(output_vector), reduction_indices=[1]), name='cross_entropy')
            if regularization_coeff > 0:
                loss = cross_entropy + regularization_coeff * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias))
            else:
                loss = cross_entropy
            cross_entropy_summary = tf.scalar_summary('cross entropy', cross_entropy)
            loss_summary = tf.scalar_summary('loss', loss)
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, name='train_step')

        with tf.name_scope('evaluate'):
            start_index = 0
            if self.output_attributes:
                attributes_nhot = tf.slice(output_vector, [0, start_index], [-1, attribute_vector_size])
                self.attribute_answers = attributes_nhot
                start_index = start_index + attribute_vector_size
            assert start_index == output_vector_size  # ka viss outputs ir noprocesēts

        init = tf.initialize_all_variables()
        self.merged = tf.merge_all_summaries()
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        self.session.run(init)

    def train(self, train_data, epochs, output_dir, test_data=None):
        sentence_count = 0
        self.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.train_writer = tf.train.SummaryWriter(output_dir + '/tb/train' + self.run_id, self.session.graph)
        self.test_writer = tf.train.SummaryWriter(output_dir + '/tb/test' + self.run_id)

        for epoch in range(1, epochs + 1):
            for sentence in train_data:
                # Bez tensorboard tikai šis
                # self.session.run(self.train_step, feed_dict = {self.token_ids : sentence_wordforms, self.tag_ids : sentence_tags})
                sentence_count = sentence_count + 1
                if sentence_count % 1000 == 0:
                    #                     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    #                     run_metadata = tf.RunMetadata()
                    #                     summary, _ = self.session.run([self.merged, self.train_step], feed_dict = self._feed_dict(sentence, train=True),
                    #                                                  options = run_options, run_metadata = run_metadata)
                    #                     self.train_writer.add_run_metadata(run_metadata, 'sentences_{}k'.format(sentence_count // 1000), sentence_count)
                    summary, _ = self.session.run([self.merged, self.train_step],
                                                  feed_dict=self._feed_dict(sentence, train=True))
                    self.train_writer.add_summary(summary, sentence_count)
                    if test_data:
                        # TODO - lai būtu ar vienu parsējienu visi 3 rezultāti
                        if self.output_pos:
                            acc = self.evaluate_accuracy(test_data, self.pos_accuracy_measure)
                            test_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="pos_accuracy", simple_value=acc)])
                            self.test_writer.add_summary(test_summary, sentence_count)
                        if self.output_tag:
                            acc = self.evaluate_accuracy(test_data, self.tag_accuracy_measure)
                            test_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="tag_accuracy", simple_value=acc)])
                            self.test_writer.add_summary(test_summary, sentence_count)
                        if self.output_attributes:
                            # TODO - metrika atribūtu precizitātei
                            pass
                elif sentence_count % 10 == 9:
                    summary, _ = self.session.run([self.merged, self.train_step],
                                                  feed_dict=self._feed_dict(sentence, train=True))
                    self.train_writer.add_summary(summary, sentence_count)
                else:
                    self.session.run(self.train_step, feed_dict=self._feed_dict(sentence, train=True))
            print('Epoch {} done'.format(epoch))

    def evaluate_accuracy(self, document_vectors, measure):
        acc = 0.0
        tokens = 0.0
        for sentence in document_vectors:
            tokens = tokens + len(sentence.tag_ids)
            acc = acc + len(sentence.tag_ids) * self.session.run(measure, feed_dict=self._feed_dict(sentence))
        acc = acc / tokens
        return acc

    # TODO - lai būtu ar vienu parsējienu visi rezultāti
    def _parse_sentences_tags(self, document_vectors, vocabularies):
        for sentence in document_vectors:
            silver_tag_ids = self.session.run(self.tag_answers,
                                              feed_dict=self._feed_dict(sentence, feed_output=False))
            yield [vocabularies.voc_tags.reverse(tag_id) for tag_id in silver_tag_ids]

    def _parse_sentences_pos(self, document_vectors, vocabularies):
        for sentence in document_vectors:
            silver_pos_ids = self.session.run(self.pos_answers,
                                              feed_dict=self._feed_dict(sentence, feed_output=False))
            yield [vocabularies.voc_pos.reverse(pos_id) for pos_id in silver_pos_ids]

    def _parse_sentences_attributes(self, document_vectors, vocabularies):
        for sentence in document_vectors:
            silver_attribute_data = self.session.run(self.attribute_answers,
                                                     feed_dict=self._feed_dict(sentence, feed_output=False))
            result = []
            for token_silver_attributes in silver_attribute_data:
                result.append({vocabularies.voc_attributes.reverse(attribute_id): value for attribute_id, value in
                               enumerate(token_silver_attributes)})
            yield result

    def tag(self, test_doc, test_data, vocabularies, filename):
        # silver_tags = None
        # silver_pos = None
        silver_attributes = None
        # if self.output_pos:
        #     silver_pos = self._parse_sentences_pos(test_data, vocabularies)
        # if self.output_tag:
        #     silver_tags = self._parse_sentences_tags(test_data, vocabularies)
        if self.output_attributes:
            silver_attributes = self._parse_sentences_attributes(test_data, vocabularies)
        test_doc.output_tagged(silver_attributes, filename, vocabularies=vocabularies)

    def dump(self, folder, filename='model.tf'):
        self._featurefactory.dump(folder)
        self.saver.save(self.session, folder + '/' + filename)

    def load_model(self, folder):
        self.saver.restore(self.session, folder + '/model.tf')


# NN layer helper functions

# Layer for packing a bidirectional RNN
def recurrent_layer(input_layer, sequence_lengths, dropout_keep_prob, name_scope='recurrent', rnn_hidden=100,
                    num_layers=1):
    with tf.name_scope(name_scope):
        # Pārveidojam no "vārdu skaits x data width" uz "Batch size x time steps x data width"?
        batched_input_layer = tf.expand_dims(input_layer, 0)
        with tf.name_scope('fw'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(rnn_hidden, state_is_tuple=True)
            cell_with_dropout_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_prob)
            if num_layers > 1:
                multicell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_with_dropout_fw] * num_layers, state_is_tuple=True)
            else:
                multicell_fw = cell_fw

        with tf.name_scope('bw'):
            cell_bw = tf.nn.rnn_cell.LSTMCell(rnn_hidden, state_is_tuple=True)
            cell_with_dropout_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=dropout_keep_prob)
            if num_layers > 1:
                multicell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_with_dropout_bw] * num_layers, state_is_tuple=True)
            else:
                multicell_bw = cell_bw
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(multicell_fw, multicell_bw, batched_input_layer,
                                                         sequence_lengths, scope=name_scope + "BiRNN", dtype=tf.float32)
        rnn_output = tf.concat(2, rnn_outputs)
        return tf.reshape(rnn_output, [-1, rnn_hidden * 2])  # Flatten so that it's without batches again


# A simple fully connected layer with relu activation
def fully_connected_layer(input_layer, dropout_keep_prob, hidden_units, name_scope='fully_connected'):
    input_vector_size = input_layer.get_shape().as_list()[1]
    with tf.name_scope(name_scope):
        weights = tf.Variable(tf.truncated_normal([input_vector_size, hidden_units], stddev=0.1), name='weights')
        bias = tf.Variable(tf.zeros([hidden_units]), name='bias')  # FIXME - te vajag fiksētu 0.1+
        fc = tf.nn.relu(tf.matmul(input_layer, weights) + bias, name=name_scope)
        return tf.nn.dropout(fc, dropout_keep_prob)


# viendimensiju konvolūcija pa vārdiem; paņemam vektoru formā [vārdi x fīčas], atgriežam formā [vārdi x output]
# veidojot output ne tikai no 'savām' fīčām kā fully_connected bet no 'window' izmēra loga apkārt tam vārdam
def convolution_layer(input_layer, window=3, hidden_units=400, name_scope='convolution'):
    input_vector_size = input_layer.get_shape().as_list()[1]
    # pārvēršam no [vārdi x fīčas] uz [1 x 1x vārdi x fīčas] - [batch x height x vārdi x fīčas]
    batched_layer = tf.expand_dims(tf.expand_dims(input_layer, 0), 0)
    with tf.name_scope(name_scope):
        convolution = tf.Variable(tf.truncated_normal([1, window, input_vector_size, hidden_units], stddev=0.1),
                                  name='covolution_filter')
        result = tf.nn.conv2d(batched_layer, convolution, [1, 1, 1, 1], padding='SAME')
        unbatched = tf.reshape(result, [-1, hidden_units])
    return unbatched
