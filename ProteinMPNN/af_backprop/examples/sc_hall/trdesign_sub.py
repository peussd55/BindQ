import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v1.keras.backend as K1
tf1.disable_eager_execution()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Lambda, Layer, Concatenate

def get_TrR_weights(filename):
  weights = [np.squeeze(w) for w in np.load(filename, allow_pickle=True)]
  # remove weights for beta-beta pairing
  del weights[-4:-2]
  return weights

def get_TrR(blocks=12, trainable=False, weights=None, name="TrR"):
  ex = {"trainable":trainable}
  # custom layer(s)
  class PSSM(Layer):
    # modified from MRF to only output tiled 1D features
    def __init__(self, diag=0.4, use_entropy=False):
      super(PSSM, self).__init__()
      self.diag = diag
      self.use_entropy = use_entropy
    def call(self, inputs):
      x,y = inputs
      _,_,L,A = [tf.shape(y)[k] for k in range(4)]
      with tf.name_scope('1d_features'):
        # sequence
        x_i = x[0,0,:,:20]
        # pssm
        f_i = y[0,0]
        # entropy
        if self.use_entropy:
          h_i = K.sum(-f_i * K.log(f_i + 1e-8), axis=-1, keepdims=True)
        else:
          h_i = tf.zeros((L,1))
        # tile and combined 1D features
        feat_1D = tf.concat([x_i,f_i,h_i], axis=-1)
        feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
        feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

      with tf.name_scope('2d_features'):
        ic = self.diag * tf.eye(L*A)
        ic = tf.reshape(ic,(L,A,L,A))
        ic = tf.transpose(ic,(0,2,1,3))
        ic = tf.reshape(ic,(L,L,A*A))
        i0 = tf.zeros([L,L,1])
        feat_2D = tf.concat([ic,i0], axis=-1)

      feat = tf.concat([feat_1D_tile_A, feat_1D_tile_B, feat_2D],axis=-1)
      return tf.reshape(feat, [1,L,L,442+2*42])
      
  class instance_norm(Layer):
    def __init__(self, axes=(1,2),trainable=True):
      super(instance_norm, self).__init__()
      self.axes = axes
      self.trainable = trainable
    def build(self, input_shape):
      self.beta  = self.add_weight(name='beta',shape=(input_shape[-1],),
                                  initializer='zeros',trainable=self.trainable)
      self.gamma = self.add_weight(name='gamma',shape=(input_shape[-1],),
                                  initializer='ones',trainable=self.trainable)
    def call(self, inputs):
      mean, variance = tf.nn.moments(inputs, self.axes, keepdims=True)
      return tf.nn.batch_normalization(inputs, mean, variance, self.beta, self.gamma, 1e-6)

  ## INPUT ##
  inputs = Input((None,None,21),batch_size=1)
  A = PSSM()([inputs,inputs])
  A = Dense(64, **ex)(A)
  A = instance_norm(**ex)(A)
  A = Activation("elu")(A)

  ## RESNET ##
  def resnet(X, dilation=1, filters=64, win=3):
    Y = Conv2D(filters, win, dilation_rate=dilation, padding='SAME', **ex)(X)
    Y = instance_norm(**ex)(Y)
    Y = Activation("elu")(Y)
    Y = Conv2D(filters, win, dilation_rate=dilation, padding='SAME', **ex)(Y)
    Y = instance_norm(**ex)(Y)
    return Activation("elu")(X+Y)

  for _ in range(blocks):
    for dilation in [1,2,4,8,16]:
      A = resnet(A, dilation)
  A = resnet(A, dilation=1)
  
  ## OUTPUT ##
  A_input   = Input((None,None,64))
  p_theta   = Dense(25, activation="softmax", **ex)(A_input)
  p_phi     = Dense(13, activation="softmax", **ex)(A_input)
  A_sym     = Lambda(lambda x: (x + tf.transpose(x,[0,2,1,3]))/2)(A_input)
  p_dist    = Dense(37, activation="softmax", **ex)(A_sym)
  p_omega   = Dense(25, activation="softmax", **ex)(A_sym)
  A_model   = Model(A_input,Concatenate()([p_theta,p_phi,p_dist,p_omega]))

  ## MODEL ##
  model = Model(inputs, A_model(A),name=name)
  if weights is not None: model.set_weights(weights)
  return model

def get_TrR_model(L=None, exclude_theta=False, use_idx=False, use_bkg=False, models_path="models"):
  def gather_idx(x):
    idx = x[1][0]
    return tf.gather(tf.gather(x[0],idx,axis=-2),idx,axis=-3)

  def get_cce_loss(x, eps=1e-8, only_dist=False):
    if only_dist:
      true_x = split_feat(x[0])["dist"]
      pred_x = split_feat(x[1])["dist"]
      loss = -tf.reduce_mean(tf.reduce_sum(true_x*tf.math.log(pred_x + eps),-1),[-1,-2])
      return loss * 4
      
    elif exclude_theta:
      true_x = split_feat(x[0])
      pred_x = split_feat(x[1])
      true_x = tf.concat([true_x[k] for k in ["phi","dist","omega"]],-1)
      pred_x = tf.concat([pred_x[k] for k in ["phi","dist","omega"]],-1)
      loss = -tf.reduce_mean(tf.reduce_sum(true_x*tf.math.log(pred_x + eps),-1),[-1,-2])
      return loss * 4/3

    else:
      return -tf.reduce_mean(tf.reduce_sum(x[0]*tf.math.log(x[1] + eps),-1),[-1,-2])
  
  def get_bkg_loss(x, eps=1e-8):
    return -tf.reduce_mean(tf.reduce_sum(x[1]*(tf.math.log(x[1]+eps)-tf.math.log(x[0]+eps)),-1),[-1,-2])      

  def prep_seq(x_logits):
    x_soft = tf.nn.softmax(x_logits,-1)
    x_hard = tf.one_hot(tf.argmax(x_logits,-1),20)
    x = tf.stop_gradient(x_hard - x_soft) + x_soft
    x = tf.pad(x,[[0,0],[0,0],[0,1]])
    return x[None]

  I_seq_logits = Input((L,20),name="seq_logits")
  seq = Lambda(prep_seq,name="seq")(I_seq_logits)
  I_true = Input((L,L,100),name="true")
  if use_bkg:
    I_bkg = Input((L,L,100),name="bkg")
  if use_idx:
    I_idx = Input((None,),dtype=tf.int32,name="idx")
    I_idx_true = Input((None,),dtype=tf.int32,name="idx_true")
  
  pred = []
  for nam in ["xaa","xab","xac","xad","xae"]:
    print(nam)
    TrR = get_TrR(weights=get_TrR_weights(f"{models_path}/model_{nam}.npy"),name=nam)
    pred.append(TrR(seq))
  pred = sum(pred)/len(pred)

  if use_idx:
    pred_sub = Lambda(gather_idx, name="pred_sub")([pred,I_idx])
    true_sub = Lambda(gather_idx, name="true_sub")([I_true,I_idx_true])
  else:
    pred_sub = pred
    true_sub = I_true
  
  cce_loss = Lambda(get_cce_loss,name="cce_loss")([true_sub, pred_sub])
  if use_bkg:
    bkg_loss = Lambda(get_bkg_loss,name="bkg_loss")([I_bkg, pred])
    loss = Lambda(lambda x: x[0]+0.1*x[1])([cce_loss,bkg_loss])
  else:
    loss = cce_loss
  grad = Lambda(lambda x: tf.gradients(x[0],x[1]), name="grad")([loss,I_seq_logits])

  # setup model
  inputs = [I_seq_logits, I_true]
  outputs = [cce_loss]
  if use_bkg:
    inputs += [I_bkg]
    outputs += [bkg_loss]
  if use_idx: inputs += [I_idx, I_idx_true]  
  model = Model(intputs, outputs + [grad, pred], name="TrR_model")
  
  TrR_model(seq, true, **kwargs):
    i = [seq[None],true[None]]
    if use_bkg:
      i += [kwargs["bkg"][None]]
    if use_idx:
      pos_idx = kwargs["pos_idx"]
      if "pos_idx_ref" not in kwargs or kwargs["pos_idx_ref"] is None:
        pos_idx_ref = pos_idx
      else:
        pos_idx_ref = kwargs["pos_idx_ref"]
      i += [pos_idx[None],pos_idx_ref[None]]
    
    *o = model.predict(i)
    r = {"cce_loss":o[0][0],"grad":o[-1][0],"pred":o[-2][0]}
    if use_bkg: r["bkg_loss"] = o[1][0]
    return r
  
  return TrR_model
