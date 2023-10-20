from tensorflow.keras import layers, regularizers, models, optimizers, losses
from data_set_params import DataSetParams
from scipy.io import wavfile
from plotly import figure_factory as ff
import tensorflow as tf
import os
import numpy as np
params = DataSetParams()

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this 
    # to work in a TensorFlow graph.
    return parts[-2]

def gen_complex_spec(waveform, sr):
    #nfft     = int(0.02*float(sr)) #params.fft_win_length
    nfft     = 48000//50 #params.fft_win_length
    #noverlap = int(0.1*nfft) #params.fft_overlap
    noverlap = nfft//10 #params.fft_overlap
    
    # window data
    step    = nfft - noverlap
    #step = 2**12
    waveform = tf.cast(waveform, tf.float32)
    
    # produces an array of complex numbers representing magnitude and phase
    complex_spec = tf.signal.stft(
        waveform, frame_length=nfft, frame_step=step)
    return complex_spec

def multi_spec_stack(complex_spec, choices = ['Mod'], stack = True):
    #possible components:
    ##Mod    modulus/absolute value (regular)
    ##Re     real component
    ##AbsRe  abs(real component)
    ##Im     imaginary component
    ##AbsIm  abs(imaginary component)
    ##Ang    angular componant
    ##AbsAng abs(angular componant)
    spec_arr = []
    for c in choices:
        def apply_func(x,c):
            if c == 'Mod':
                return tf.math.abs(x)
            if c == 'LogMod':
                return tf.math.log(tf.math.abs(x)+np.finfo(np.float32).eps)
            if c == 'Re':
                return tf.math.real(x)
            if c == 'AbsRe':
                return tf.math.abs(tf.math.real(x))
            if c == 'LogAbsRe':
                return tf.math.log(tf.math.abs(tf.math.real(x))+np.finfo(np.float32).eps)
            if c == 'Im':
                return tf.math.imag(x)
            if c == 'AbsIm':
                return tf.math.abs(tf.math.imag(x))
            if c == 'LogAbsIm':
                return tf.math.log(tf.math.abs(tf.math.imag(x))+np.finfo(np.float32).eps)
            if c == 'Ang':
                return tf.math.angle(x)
            if c == 'AbsAng':
                return tf.math.abs(tf.math.angle(x))
        spec_arr.append(apply_func(complex_spec,c))
    if stack:
        spec_arr = tf.stack(spec_arr,-1)
    return spec_arr

def multi_spec_post(spec, req_width, spec_norm, resize):
    spec_shp         = tf.shape(spec)
    spec_cutoff      = [0, 1, 0]
    spec_cutoff_size = [spec_shp[0],spec_shp[1]-1, spec_shp[2]]
    spec             = tf.slice(spec, spec_cutoff, spec_cutoff_size)
    
    # only keep the relevant bands - could do this outside
    #spec = spec[:, :100, :]
    
    spec_shp  = tf.shape(spec)
    req_width = req_width*resize
    
    if spec_shp[0] < tf.constant(req_width):
        zero_pad = tf.ones((tf.constant(req_width) - spec_shp[0], spec_shp[1], spec_shp[2]))*1e-8
        spec     = tf.concat([spec, zero_pad], axis = 0)
    else:
        spec = tf.slice(spec, [0,0,0], [req_width, spec_shp[1], spec_shp[2]]) #spec[:,:req_width,:]
        
    spec = tf.transpose(spec, perm = [1,0,2])
    
    def spec_normalize(x):
        out_tf = 255*(x - tf.math.reduce_min(x,axis=(0,1), keepdims=True)) / (tf.math.reduce_max(x,axis=(0,1), keepdims=True) - tf.math.reduce_min(x,axis=(0,1), keepdims=True)+np.finfo(np.float32).eps) 
        #out_tf = (x - tf.math.reduce_mean(x,axis=(0,1), keepdims=True)) / (tf.math.reduce_std(x,axis=(0,1), keepdims=True) +np.finfo(np.float32).eps)
        return out_tf
        
    if spec_norm:
        spec = spec_normalize(spec)
    
    return spec

def spec_unstack(spec,label):
    spec = tf.stack(spec)
    #spec = tf.unstack(spec, axis=-1)
    spec = tf.data.Dataset.zip(tuple([tf.expand_dims(x_ind, axis=-1) for x_ind in spec]))
    return spec, label

def decode_audio(audio_binary):
    audio, sr = tf.audio.decode_wav(audio_binary) # returns the WAV-encoded audio as a tensor and the sample rate
    #return tf.squeeze(audio, axis=-1) # removes dimensions of size 1 from the last axis
    #sr = sr // 5
    return audio[:,0], sr#.numpy()

def get_waveform_sr_and_label(file_path):
    label         = get_label(file_path)
    audio_binary  = tf.io.read_file(file_path)
    waveform, sr  = decode_audio(audio_binary)
    aud_tuple     = (waveform, sr)
    return aud_tuple, label

def get_spectrogram_and_label_id(audio, label, choices, categories, req_width, single_to_rgb, resize, spec_norm):
    #spectrogram = get_complex_spectrogram(audio)
    #print(dir(audio))
    aud, sr      = audio
    complex_spec = gen_complex_spec(aud, sr)
    multi_spec   = multi_spec_stack(complex_spec, choices)
    multi_spec   = multi_spec_post(multi_spec, req_width, spec_norm, resize)
    if single_to_rgb:
        multi_spec = tf.concat([multi_spec,multi_spec,multi_spec], axis=-1)
        multi_spec = tf.stack(multi_spec)
    #multi_spec = complex_spec
    spec_shape = tf.shape(multi_spec)
    if resize > 1:
        multi_spec = tf.image.resize(multi_spec, [spec_shape[0]//resize, spec_shape[1]//resize])
    label_id   = tf.argmax(label == categories)
    return multi_spec, label_id

def preprocess_dataset(files, choices, categories, req_width = 250, single_to_rgb = False, resize = 1, spec_norm = False):
    AUTOTUNE  = tf.data.experimental.AUTOTUNE
    files_ds  = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_sr_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(lambda x,y : get_spectrogram_and_label_id(audio=x, label=y,  choices = choices, 
        categories = categories, req_width = req_width, single_to_rgb = single_to_rgb, resize = resize, spec_norm = spec_norm), 
    num_parallel_calls=AUTOTUNE)
    #output_ds = output_ds.map(lambda x,y : spec_unstack(spec=x, label=y), num_parallel_calls=AUTOTUNE)
    return output_ds

def get_wf_dataset(files):
    AUTOTUNE  = tf.data.experimental.AUTOTUNE
    files_ds  = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_sr_and_label, num_parallel_calls=AUTOTUNE)
    return output_ds

def main_cnn(input_shape, num_classes):
    model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(num_classes)
    ])
    model.compile(
    optimizer = optimizers.Adam(learning_rate=0.0001),
    loss      = losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics   = 'accuracy'
    )
    return model

def plotly_cm(cm, categories):
    z = cm

    x = categories
    y = categories

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion<br>matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    yaxis=dict(autorange='reversed'),
                    title_x=0
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.26,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    return fig

#need to copy the weights from other model
#from https://towardsdatascience.com/implementing-transfer-learning-from-rgb-to-multi-channel-imagery-f87924679166
# Expand weights dimension to match new input channels
def multify_weights(kernel, out_channels):
    mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
    tiled   = np.tile(mean_1d, (out_channels, 1))
    return(tiled)


# Loop through layers of both original model 
# and custom model and copy over weights 
# layer_modify refers to first convolutional layer
def copy_weights_tl(model_orig, custom_model, layer_modify):
    layer_to_modify = [layer_modify]

    conf = custom_model.get_config()
    layer_names   = [conf['layers'][x]['config']['name'] for x in range(len(conf['layers']))]
    input_channel = conf["layers"][0]["config"]["batch_input_shape"][-1]
    #old_input_channel = model_orig.get_config()["layers"][0]["config"]["batch_input_shape"][-1]

    for layer in model_orig.layers:
        if layer.name in layer_names:
            if layer.get_weights() != []:
                target_layer = custom_model.get_layer(layer.name)

                if layer.name in layer_to_modify:    
                    kernels = layer.get_weights()[0]
                    biases  = layer.get_weights()[1]
            
                    kernels_extra_channel = multify_weights(kernels, input_channel)
                    #print('kernels_extra_channel', kernels_extra_channel.shape)                                
                    target_layer.set_weights([kernels_extra_channel, biases])
                    target_layer.trainable = False

                else:
                    target_layer.set_weights(layer.get_weights())
                    target_layer.trainable = False

def concat_model(input_shape, num_channels, num_classes):
    from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, Dropout
    from keras.models import Model
    import numpy as np


    def part_model(input_a):
        #x1 = layers.Resizing(100, 150)(input_a)
        x1 = Conv2D(16, (3, 3), activation = 'relu')(input_a)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(16, (3, 3))(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(32, (3, 3))(x1)
        x1 = MaxPooling2D()(x1)
        out_a = Flatten()(x1)
        return(out_a)

    out_list   = []
    input_list = []
    for c in range(num_channels):
        input_a = Input(shape=input_shape)
        input_list.append(input_a)
        out_list.append(part_model(input_a))
    #print(input_list)
    #print(out_list)

    concatenated = concatenate(out_list, axis=-1)
    x = Dropout(0.5)(concatenated)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes)(x)
    model = Model(input_list, out)
    #print(model.summary())
    model.compile(
        optimizer = optimizers.Adam(learning_rate=0.0001),
        loss      = losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics   = 'accuracy'
        )
    return model

def concat_model2(input_shape, num_channels, num_classes):
    from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, Dropout
    from keras.models import Model
    import numpy as np


    def part_model(input_a):
        #x1 = layers.Resizing(100, 150)(input_a)
        x1 = Conv2D(16, (3, 3), activation = 'relu')(input_a)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(16, (3, 3))(x1)
        x1 = MaxPooling2D()(x1)
        return(x1)

    out_list   = []
    input_list = []
    for c in range(num_channels):
        input_a = Input(shape=input_shape)
        input_list.append(input_a)
        out_list.append(part_model(input_a))
    #print(input_list)
    #print(out_list)

    concatenated = concatenate(out_list, axis=-1)
    x = Conv2D(64, (3, 3))(concatenated)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3))(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes)(x)
    model = Model(input_list, out)
    #print(model.summary())
    model.compile(
        optimizer = optimizers.Adam(learning_rate=0.0001),
        loss      = losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics   = 'accuracy'
        )
    return model

def concat_model3(input_shape, num_channels, num_classes):
    from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, Dropout
    from keras.models import Model
    import numpy as np


    def part_model(input_a):
        #x1 = layers.Resizing(100, 150)(input_a)
        x1 = Conv2D(16, (3, 3), activation = 'relu')(input_a)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(16, (3, 3))(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(32, (3, 3))(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Flatten()(x1)
        x1 = Dropout(0.5)(x1)
        x1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x1)
        out_a = Dropout(0.5)(x1)
        return(out_a)

    out_list   = []
    input_list = []
    for c in range(num_channels):
        input_a = Input(shape=input_shape)
        input_list.append(input_a)
        out_list.append(part_model(input_a))
    #print(input_list)
    #print(out_list)

    concatenated = concatenate(out_list)
    out = Dense(num_classes)(concatenated)
    model = Model(input_list, out)
    #print(model.summary())
    model.compile(
        optimizer = optimizers.Adam(learning_rate=0.0001),
        loss      = losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics   = 'accuracy'
        )
    return model

def concat_model4(input_shape, num_channels, num_classes):
    from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, Dropout
    from keras.models import Model
    import numpy as np


    def part_model(input_a):
        #x1 = layers.Resizing(100, 150)(input_a)
        x1 = Conv2D(16, (3, 3), activation = 'relu')(input_a)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(16, (3, 3))(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(32, (3, 3))(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Conv2D(32, (3, 3))(x1)
        x1 = MaxPooling2D()(x1)
        x1 = Flatten()(x1)
        x1 = Dropout(0.5)(x1)
        x1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x1)
        out_a = Dropout(0.5)(x1)
        return(out_a)

    out_list   = []
    input_list = []
    for c in range(num_channels):
        input_a = Input(shape=input_shape)
        input_list.append(input_a)
        out_list.append(part_model(input_a))
    #print(input_list)
    #print(out_list)

    concatenated = concatenate(out_list)
    out = Dense(num_classes)(concatenated)
    model = Model(input_list, out)
    #print(model.summary())
    model.compile(
        optimizer = optimizers.Adam(learning_rate=0.0001),
        loss      = losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics   = 'accuracy'
        )
    return model
