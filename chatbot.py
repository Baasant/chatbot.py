import tensorflow as tf
import pandas as pd
import re
import time
import numpy as np
import keras as ks
#load the data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
#the lines that we will use to train the mode
lines[:10]
#the sentment that we wii use to train the model
conv_lines[:10]
#create a dectionary to map each line's id with its text
#this mean that after each +++$+++ the sentment end
id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
#creating a list of all the the conversations' lines' ids.
convs=[]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
#print(convs)
#sort all sentments to questions as inputs ,answers as target
#how it will know that this questions or answers??
questions=[]
answers=[]
for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
#to check the data
limit = 0
for i in range(limit, limit+5):
    print(questions[i])
    print(answers[i])
    print()

#compare length od questions and answers
print(len(questions))
print(len(answers))

#clean unnecessary characters
def clean_text(text):
    text=text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text
#clean data
#clean questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

#clean answers
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))

#find the lenght of the sentements
lengths=[]
for question in clean_questions:
    lengths.append(len(question.split()))

for answer in clean_answers:
    lengths.append(len(answer.split()))

#print(lengths) to show the length of the sentment

#create a dataframe to inspect the data
#the meaning of data frame to extract the data in a specific shap
#lengths the data the i want to see
lengths=pd.DataFrame(lengths,columns=['counts'])
#print(lengths)
lengths.describe()
print(np.percentile(lengths, 80)) #to show the precentagr of 80 length in all sentments
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))

#create a frequency of the vocabulary
#we make frequency to every word to see the importance of every words
vocab={}
for question in questions:
    for word in question.split():
        if word not in vocab:
            vocab[word]=1
        else:
            vocab[word]+=1

for answer in answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word]=1
        else:
            vocab[word]+=1
#print(vocab)
# i don't khnow the meaning of that
#i think this will make accuracy low
threshold = 10
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1
print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)

#now will create a integer number for each word
questions_vocab_to_int={}
word_num=0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1


answers_vocab_to_int={}
word_num=0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += 1

# Add the unique tokens to the vocabulary dictionaries.
codes = ['<PAD>','<EOS>','<UNK>','<GO>']
for code in codes:
    questions_vocab_to_int[code]=len(questions_vocab_to_int)+1

for code in codes:
    questions_vocab_to_int[code]=len(answers_vocab_to_int)+1

#print(questions_vocab_to_int[code])
#we will create fun to convert from int to vocab
questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}#????
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}
# Check the length of the dictionaries.
#print(len(questions_vocab_to_int))
#print(len(questions_int_to_vocab))
#print(len(answers_vocab_to_int))
#print(len(answers_int_to_vocab))


#add the end of every sentment to the end of every answer
for i in range(len(answers)):
    answers[i]+='<EOS>'

#convert the text to int
#we convert every word to int now we will but the togther we will convert the text to int

questions_int = []
for question in questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_int.append(ints)

#answers_int = []
#for answer in answers:
    #ints = []
    #for word in answer.split():
        #if word not in answers_vocab_to_int:
            #ints.append(answers_vocab_to_int['<UNK>'])
        #else:
            #ints.append(answers_vocab_to_int[word])
    #answers_int.append(ints)


#convert the answers to int


#calculate the precentage of the word that we replaced with unk
word_count=0
unk_count=0
for question in questions_int:
    for word in question:
        if word == questions_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1

'''for answer in answers_int:
    for word in answer:
        if word == answers_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
        '''

unk_ratio = round(unk_count / word_count, 4) * 100

#print("Total number of words:", word_count)
#print("Number of times <UNK> is used:", unk_count)
#print("Percent of words that are <UNK>: {}%".format(round(unk_ratio, 3)))

#Sort questions and answers by the length of questions.
#this will reduce loss and make reduce padding and make the training faster

sorted_questions=[]
sorted_answers=[]
#for length in range(1, max_line_length+1):
    #for i in enumerate(questions_int):
        #if len(i[1]) == length:
            #sorted_questions.append(questions_int[i[0]])
            #sorted_answers.append(answers_int[i[0]])

#what i understand that palceholder is a variable that we store the data in to use later
def model_inputs():
    '''Create palceholders for inputs to the model'''
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob
#we use this to remove the last word from every batch and put it in the second patch
#This formatting is necessary for creating the embeddings for our decoding layer.
#we will design encoder
def process_input(target_data,vocab_to_int,batch_size):
    ending=tf.strided_slice(target_data,[0,0],[batch_size,-1],[1,1])
    dec_input = tf.concat([tf.fill([batch_size, 1],
                                   vocab_to_int['<GO>']),
                           ending], 1)
    return dec_input
#make encoder to get data in the same lenth and prepare it
#BASICLSTMCELL means the class of the RNN cell
#rnn_cell.cell_params =means A dictionary of parameters to pass to the cell class constructor.
#rnn_cell.dropout_input keep_prob=1.0 apply dropout to the(non_recurrent)input of each RNN
#rnn_cell.dropout output keep_prob=1.0 apply dropout to the(non_recurrent)input of each RNN
#rnn_cell.num_layers num of layers
#rnn_cell.residual_connections=false if true make residual connections between RNN layers in the encoder.
#define encoder layer
def encoder_layer_train(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)  #USE TO CREATE LSTM LAYER
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    enc_cell=tf.contrib.rnnMultiRNNCell([drop] * num_layers)
    enc_state=tf.rnn.bidirectional_dynamic_rnn(cell_fw=enc_cell
                                               ,cell_bw=enc_cell
                                               ,sequence_length=sequence_length,
                                               inputs=rnn_inputs,dtype=tf.float32)
    return enc_state

#define decoder layer
#contrib.layers.embed_sequence can only embed the prepared dataset before running
#we ues two decoder one to train and the second to inference




    #att_keys to be compared with the target value
    #attention_values: to be used to construct context vectors.
    #att_score_fn to compute simularity between key and target values
    #att_construct_fnto build attention states.

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob, batch_size):
    '''Decode the training data'''

    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                              attention_option="bahdanau",
                                                                                              num_units=dec_cell.output_size)

    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name="attn_dec_train")
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                              train_decoder_fn,
                                                              dec_embed_input,
                                                              sequence_length,
                                                              scope=decoding_scope)
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    return output_fn(train_pred_drop)


#decoding_layer_infer we use it to predict our results and
#GreedyEmbeddingHelper dynamically takes the output of the current step
# and give it to the next time step’s input. In order to embed the
#each input result dynamically, embedding parameter(just bunch of weight values

def decoding_layer_infer(encoder_state,dec_cell,dec_embeddings,start_of_sequence_id, end_of_sequence_id,max_length,vocab_size,decoding_scope,output_fn, keep_prob, batch_size):
    attention_state=tf.zeros([batch_size,1,dec_cell.output_size])
    att_keys, att_vals, att_score_fn, att_construct_fn=tf.contrib.seq2seq.prepare_attentionattention(
        attention_state,attention_option='bahdanau',num_units=dec_cell.output_size)
    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn,
                                                                         encoder_state[0],
                                                                         att_keys,
                                                                         att_vals,
                                                                         att_score_fn,
                                                                         att_construct_fn,
                                                                         dec_embeddings,
                                                                         start_of_sequence_id,
                                                                         end_of_sequence_id,
                                                                         max_length,
                                                                         vocab_size,
                                                                         name="attn_dec_inf")
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                                infer_decoder_fn,
                                                                scope=decoding_scope)

    return infer_logits

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    '''Create the encoding layer'''
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,
                                                   cell_bw = enc_cell,
                                                   sequence_length = sequence_length,
                                                   inputs = rnn_inputs,
                                                   dtype=tf.float32)
    return enc_state



#we define decoder to train and interfer now we will create a decoding layers
#to train we use fully connected layer and decoder cell
#by decoding layer we connect decoding process and decoding iterferance to gether
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size):
    '''Create the decoding cell and input the parameters for the training and inference decoding layers'''

    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x,
                                                                vocab_size,
                                                                None,
                                                                scope=decoding_scope,
                                                                weights_initializer=weights,
                                                                biases_initializer=biases)

        train_logits = decoding_layer_train(encoder_state,
                                            dec_cell,
                                            dec_embed_input,
                                            sequence_length,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state,
                                            dec_cell,
                                            dec_embeddings,
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'],
                                            sequence_length - 1,
                                            vocab_size,
                                            decoding_scope,
                                            output_fn, keep_prob,
                                            batch_size)

    return train_logits, infer_logits



#after what we do we need to collect them together this called seq2seq model we use it
#to build graphs,loss,optomizing
#tf.contrib.layers.embed_sequence we use it to reduce the num
#reduce the number of parameters in your network while preserving depth and implementation

def seq2seq_model(input_data,target_data,keep_prob,batch_size,sequence_length,answers_vocab_size,
    questions_vocab_size,enc_embedding_size,dec_embedding_size,rnn_size,num_layers,questions_vocab_to_int):
    enc_embed_input=tf.contrib.layers.embed_sequence(
        input_data,answers_vocab_size+1,enc_embedding_size,
        initializer=tf.random_uniform_initializer(-1,1))

    enc_state=encoding_layer(enc_embed_input,rnn_size,num_layers,keep_prob,sequence_length)


    dec_input=process_input(target_data,questions_vocab_to_int,batch_size)

    dec_embeddings = tf.Variable(
        tf.random_uniform([questions_vocab_size + 1,
                           dec_embedding_size],
                          -1, 1))
    #s used to perform parallel lookups on the list of tensors in params
    dec_embed_input=tf.nn.embedding_lookup(dec_embeddings,dec_input)
    train_logits, infer_logits = decoding_layer(
        dec_embed_input,
        dec_embeddings,
        enc_state,
        questions_vocab_size,
        sequence_length,
        rnn_size,
        num_layers,
        questions_vocab_to_int,
        keep_prob,
        batch_size)
    return train_logits, infer_logits


#set all hyperparameter
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75

#reset the graph to make sure that it's ready to train
tf.reset_default_graph()
#strart the season
sess=tf.InteractiveSession()

#load the model input
input_data,targets,lr, keep_prob = model_inputs()

#sequence length will be the max line length for each batch
max_line_length = 20
sequence_length=tf.placeholder_with_default(max_line_length, None, name='sequence_length')

#logits is function to map probablity
#find the shap of the input data to seq loss
input_shape=tf.shape(input_data)

#crate the train and interfer logits
train_logits, inference_logits=seq2seq_model(tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(answers_vocab_to_int),
    len(questions_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers,
    questions_vocab_to_int)


with tf.name_scope('optimization'):
    cost=tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))
    #optomizer
    optimizer=tf.train.AdamOptimizer(learning_rate)



    #gradient cliping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]




def batch_data(questions, answers, batch_size):

 for batch_i in range(0,len(questions)//batch_size): #batch questions and answers together
    start_i=batch_i*batch_size
    questions_batch=questions[start_i:start_i+batch_size]
    answers_batch=answers[start_i:start_i + batch_size]

    pad_questions_batch = np.array(pad_sentence_batch(questions_batch, questions_vocab_to_int))
    pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))
    yield pad_questions_batch, pad_answers_batch


#validation the traing with 10% of the data
train_valid_split=int(len(sorted_questions)*0.15)


# Split the questions and answers into training and validating data
train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_answers[train_valid_split:]


valid_questions=sorted_questions[:train_valid_split]
valid_answers=sorted_answers[:train_valid_split]

print(len(train_questions))
print(len(valid_questions))


display_step=100 #check training loss after every 100 batch
stop_early=0
stop=5 # if the validation loss does decrease in 5 consecutive checks, stop training
validation_check=(len(train_questions)//batch_size//2)-1 # Modulus for checking validation loss
total_train_loss=0 #record the training loss for display step
summary_valid_loss=[] #record the validation for saving improvements

checkpoint='best_model.ckpt'



sess.run(tf.global_variables_initializer())

for epoch_i in range(1,epochs+1):
    for batch_i,(questions_batch, answers_batch) in enumerate(
            batch_data(train_questions, train_answers, batch_size)):
        start_time=time.time()
        loss=sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability})


        total_train_loss+=loss
        end_time=time.time()
        batch_time=end_time-start_time

        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs,
                          batch_i,
                          len(train_questions) // batch_size,
                          total_train_loss / display_step,
                          batch_time * display_step))
            total_train_loss = 0

            if batch_i % validation_check == 0 and batch_i > 0:
                total_valid_loss = 0
                start_time = time.time()
                for batch_ii, (questions_batch, answers_batch) in enumerate(
                        batch_data(valid_questions, valid_answers, batch_size)):
                    valid_loss = sess.run(
                        cost, {input_data: questions_batch,
                               targets: answers_batch,
                               lr: learning_rate,
                               sequence_length: answers_batch.shape[1],
                               keep_prob: 1})
                    total_valid_loss += valid_loss
                end_time = time.time()
                batch_time = end_time - start_time
                avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
                print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

                # Reduce learning rate, but not below its minimum value
                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

                summary_valid_loss.append(avg_valid_loss)
                if avg_valid_loss<min(summary_valid_loss):
                    print('new record')
                    stop_early=0
                    saver = tf.train.Saver()
                saver.save(sess, checkpoint)

            else:
                    print('no improvment')
                    stop_early+=1
                    if stop_early==stop:
                        break
        if stop_early==stop:
            print('stop training')
            break



#prepare the question for the model
def question_to_seq(question, vocab_to_int):
    '''Prepare the question for the model'''

    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]

#create my input question
#use a question from your data as input

random=np.random.choice(len(questions))
input_question=question[random]

#pad the questions until it equals the max_line_length
input_question = input_question + [questions_vocab_to_int["<PAD>"]] * (max_line_length - len(input_question))

#add empty questions to make data correct
batch_shell=np.zeros(batch_size,max_line_length)


#set frist question to be output input question
batch_shell[0]=input_question

# Remove the padding from the Question and Answer
pad_q = questions_vocab_to_int["<PAD>"]
pad_a = answers_vocab_to_int["<PAD>"]


#run the model with input question
answer_logits=sess.run(inference_logits,{input_data:batch_shell,keep_prob:1.0})[0]
print('Question')
print('  Word Ids:      {}'.format([i for i in input_question if i != pad_q]))
print('  Input Words: {}'.format([questions_int_to_vocab[i] for i in input_question if i != pad_q]))

print('\nAnswer')
print('  Word Ids:      {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
print('  Response Words: {}'.format([answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))





