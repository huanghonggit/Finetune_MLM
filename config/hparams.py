
# CONFIG -----------------------------------------------------------------------------------------------------------#

# Here are the input and output data paths (Note: you can override input/outputpath in __main__.py)
input_path = ''
output_path = ''
save_runs_path = ''


# Mlm params ------------------------------------------------------------------------------------------------------#
model_path = 'src/data/bert_ep0.model'
embed_dim = 256
hidden = 256
attn_layers = 4
attn_heads = 4
enc_maxlen = 1024
pos_dropout_rate = 0.1
enc_conv1d_dropout_rate = 0.2
enc_conv1d_layers = 3
enc_conv1d_kernel_size = 5
enc_ffn_dropout_rate = 0.1

self_att_dropout_rate = 0.1
self_att_block_res_dropout = 0.1


# Optimizer params -----------------------------------------------------------------------------------------------------#
lr = 2e-5
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.1
mlm_clip_grad_norm = 1.0
clip_grad_norm = True
warmup_steps = 100
warmup = 0.1


# Trainer params -----------------------------------------------------------------------------------------------------#
mode = 'train'
epochs = 3
log_freq = 5
save_train_loss = 20
save_valid_loss = 200
save_model = 2000
save_checkpoint = 2000
save_runs = 200
batch_size = 32
valid_size = 32
total_steps = 100000


# ------------------------------------------------------------------------------------------------------------------#

