import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
import random
from .transformer import build_transformer

class MATR(nn.Module):
    def __init__(self, args):
        super(MATR, self).__init__()
        self.args = args
        self.training = args.training
        self.n_feature = args.feat_dim
        n_class = args.num_of_class
        n_embedding_dim = args.hidden_dim
        self.n_embedding_dim = n_embedding_dim
        n_enc_layer = args.enc_layers
        n_enc_head = args.e_nheads
        n_dec_layer = args.dec_layers
        n_dec_head = args.d_nheads
        n_seglen = args.num_frame
        self.n_seglen = n_seglen
        self.num_queries = args.num_queries
        self.rgb = args.rgb
        self.flow = args.flow
        self.dropout=args.dropout
        
        self.use_flag = args.use_flag
        self.flag_threshold = args.flag_threshold
        self.max_memory_len = args.max_memory_len
        
        # FC layers for multi-modals
        if self.rgb and self.flow:
            self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
            self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        else:
            self.feature_reduction_rgb = nn.Linear(self.n_feature, n_embedding_dim)
            self.feature_reduction_flow = nn.Linear(self.n_feature, n_embedding_dim)
        
        # separte attention
        self.segment_encoder, self.segment_decoder = build_transformer(args)
        self.memory_encoder, self.memory_decoder = build_transformer(args)

        self.classification_head = nn.Sequential(nn.Linear(n_embedding_dim*2,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,n_class)) 
        self.stcls_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,self.max_memory_len+2))
        self.streg_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.Tanh(), nn.Linear(n_embedding_dim,self.max_memory_len+2))
        self.edcls_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,1))
        self.edreg_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.Tanh(), nn.Linear(n_embedding_dim,1))
        
        if self.use_flag:
            self.flag_token = nn.Parameter(torch.randn(1, 1, n_embedding_dim))
            self.segment_flag_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, 1), nn.Sigmoid())     
        
        self.decoder_token = nn.Parameter(torch.randn(self.num_queries*2, 1, n_embedding_dim))
        
        # memory queue
        self.dropout = nn.Dropout(args.dropout)
        self.memory_sampler = args.memory_sampler
        # self.compress_layer = nn.Identity()
        # self.compressed_size = n_seglen
        # self.compressed_ratio = 1
        
        self.pos_token = nn.Parameter(torch.randn(self.num_queries, 1, n_embedding_dim))
        self.segment_pos_encoding = PositionalEncoding_segment(n_embedding_dim, args.dropout, maxlen=400)
        self.memory_pos_encoding = PositionalEncoding_memory_flag(n_embedding_dim, args.dropout, maxlen=400)       
            
        self.video_name = None
        self.memory_queue = None
        self.memory_queue_index = None
    
    def forward(self, inputs, device):
        # inputs - batch x seq_len x featsize
        self.device = device
        infos = inputs['infos']
        inputs = inputs['inputs']     
        st = infos['st']
        ed = infos['ed']
        bs = inputs.shape[0]
        
        base_x = self.input_projection(inputs) # seq_len x batch x featsize
        
        if self.use_flag:
            flag_token = self.flag_token.expand(-1, base_x.shape[1], -1)
            base_x = torch.cat((base_x, flag_token), dim=0)
        pos_x = self.segment_pos_encoding(base_x)
        
        encoded_x = self.segment_encoder(base_x, pos=pos_x)
        
        
        if self.use_flag:
            anc_flag = self.segment_flag_head(encoded_x[-1]) # batch x 1
        else:
            anc_flag = torch.ones((bs, 1)).to(device)
            
        base_x = base_x[:self.n_seglen]
        pos_x = pos_x[:self.n_seglen]
        encoded_x = encoded_x[:self.n_seglen]
            
        self.check_nan(encoded_x)

        memory_queue = self.memory_queue
        memory_queue_index = self.memory_queue_index
        if self.args.training or self.use_flag != True:
            memory_args = (bs, infos['video_name'], infos['current_frame'], infos['segment_flag'])
        else:
            memory_args = (bs, infos['video_name'], infos['current_frame'], (anc_flag > self.flag_threshold).int())
        
        memory_feature, memory_feature_index, memory_queue, memory_queue_index = self.memory_update(memory_args=memory_args,
                                                   memory_queue=memory_queue,
                                                   memory_queue_index=memory_queue_index, 
                                                   current_segment=base_x)

        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        pos_token = self.pos_token.repeat(2, decoder_token.shape[1], 1)

        decoded_x, att_seg = self.segment_decoder(decoder_token, encoded_x, query_pos=pos_token, pos=pos_x)
        
        end_cls_feature, anc_end_offset = self.end_offset_process(decoded_x)
        
        memory_feature_index = memory_feature_index.permute([1,0]) # batch x max len
        memory_feature_index = (infos['current_frame'].unsqueeze(dim=1) - memory_feature_index) / self.n_seglen
        memory_feature_index[memory_feature_index < 0] = -1
        mem_pos = self.memory_pos_encoding_process(memory_feature, memory_feature_index)

        decoded_x, att_mem = self.memory_decoder(decoded_x, memory_feature, query_pos=pos_token, pos=mem_pos)
        decoded_x = decoded_x.permute([1,0,2]) # batch x queires len x featsize
        
        self.check_nan(decoded_x)
        
        decoded_x_cls = decoded_x[:,:self.num_queries]
        decoded_x_reg = decoded_x[:,self.num_queries:]
        
        decoded_x_cls = torch.cat([end_cls_feature, decoded_x_cls], dim=2)
        anc_cls = self.classification_head(decoded_x_cls)

        anc_stcls = self.stcls_head(decoded_x_reg)
        anc_start_offset = self.streg_head(decoded_x_reg)        
        anc_reg = torch.cat([anc_start_offset, anc_end_offset], dim=2)
        
        if memory_queue != None:
            self.memory_queue = memory_queue.detach()
        if memory_queue_index != None:
            self.memory_queue_index = memory_queue_index
            
        self.video_name = infos['video_name']
        
        out = {
            "pred_cls": anc_cls,
            "pred_reg": anc_reg
        }
        out["pred_stcls"] = anc_stcls
        out["att_1"] = att_seg
        out["att_2"] = att_mem
            
        if self.use_flag:
            out['pred_flag'] = anc_flag

        return out
            
    def input_projection(self, inputs):
        if self.rgb and self.flow:
            base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
            base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
            base_x = torch.cat([base_x_rgb,base_x_flow],dim=-1)
        elif self.rgb:
            base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature])
            base_x = base_x_rgb
        else:
            base_x_flow = self.feature_reduction_flow(inputs[:,:,:self.n_feature])
            base_x = base_x_flow
        return base_x.permute([1,0,2])
    
    def check_nan(self, values):
        if (values.isnan() == True).any():
            import pdb;pdb.set_trace()
    
    def memory_update(self, memory_args, memory_queue, memory_queue_index, current_segment):  
        bs, video_names, cur_frames, flags = memory_args      
        # memory initialization
        if memory_queue == None:
            memory_queue = torch.zeros((self.max_memory_len * self.n_seglen, bs, self.n_embedding_dim)).to(self.device) # max_len*seg len x batch x featsize
            memory_queue_index = 10000*torch.ones((self.max_memory_len, bs)).to(self.device) # max_memory_len x batch
        
        # reset memory when input segment belongs to another video.
        elif self.video_name != None:
            for b, video_name in enumerate(video_names):  
                if self.video_name[b] != video_name:
                    memory_queue[:, b:] = 0
                    memory_queue_index[:, b:] = 10000                    
        
        # memory features index        
        memory_feature_index = torch.cat((memory_queue_index,cur_frames.unsqueeze(dim=0)), dim=0) # max_len+1 x batch
        
        # memory queue + current segment
        current_segment = current_segment.detach()
        _memory_queue = torch.cat((memory_queue[:,:bs], current_segment), dim=0) # max_len+1*seg len x batch x featsize
        
        if self.use_flag:
            memory_queue = memory_queue.permute([1,0,2]) # batch x max_len*seg len x featsize
            memory_queue_index = memory_queue_index.permute([1,0]) # batch x max_len
            
            for i, b_memory_queue in enumerate(memory_queue[:bs]): # b_memory_queue: max_len*seg len x featsize
                if flags[i] == 1:
                    memory_queue[i] = torch.cat((b_memory_queue, current_segment[i]), dim=0)[self.n_seglen:]
                    memory_queue_index[i] = torch.cat((memory_queue_index[i], cur_frames[i].unsqueeze(dim=0)), dim=0)[1:]                 
            memory_queue = memory_queue.permute([1,0,2]).contiguous() # max_len*seg len x batch x featsize
            memory_queue_index = memory_queue_index.permute([1,0]).contiguous() # max_len x batch
        else:
            memory_queue = _memory_queue[self.n_seglen:]
            memory_queue_index = memory_feature_index[1:]
             
        # sampled memory
        memory_feature = self.sample_memory(_memory_queue)

        return memory_feature, memory_feature_index, memory_queue, memory_queue_index
    
    def end_offset_process(self, decoded_x):
        decoded_x_cls = decoded_x.permute([1,0,2])[:,:self.num_queries]
        decoded_x_end = decoded_x.permute([1,0,2])[:,self.num_queries:]

        endreg = self.edreg_head(decoded_x_end)

            
        return decoded_x_cls, endreg
    
    def memory_pos_encoding_process(self, memory_feature, memory_feature_index):         
        between_memory_index = memory_feature_index.repeat_interleave(self.n_seglen, dim=1)
        inside_memory_index = torch.arange(self.n_seglen-1,-1,-1).repeat(self.max_memory_len+1)
        inside_memory_index = inside_memory_index.unsqueeze(0).expand(between_memory_index.size(0),-1)
        mem_pos = self.memory_pos_encoding(between_memory_index, inside_memory_index, self.memory_sampler)

        return mem_pos
    
    def sample_memory(self, memory):
        if memory == None:
            return None
        if "gap" in self.memory_sampler:
            gap_size = int(self.memory_sampler[-1])
            return memory[0::gap_size]
        elif self.memory_sampler == 'all':
            return memory

class PositionalEncoding_segment(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.3,
                 maxlen: int = 750,
                 batch_first=False):
        super(PositionalEncoding_segment, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        self.register_buffer(
            'position_ids',
            torch.arange(maxlen).expand((1, -1))
            )
    def forward(self, token_embedding: torch.Tensor): 
        return self.pos_embedding[:token_embedding.size(0), :].flip(dims=(0,))
    
class PositionalEncoding_memory_flag(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.3,
                 maxlen: int = 750
                 ):
        super(PositionalEncoding_memory_flag, self).__init__()
        emb_size = int(emb_size/2)
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / (emb_size))
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        self.register_buffer(
            'position_ids',
            torch.arange(maxlen))
    def forward(self, between_memory_index, inside_memory_index, memory_sampler): 
        if memory_sampler == 'all':
            between_pos_embedding = self.pos_embedding[between_memory_index.long()].permute([1,0,2])
            inside_pos_embedding = self.pos_embedding[inside_memory_index.long()].permute([1,0,2])     
        elif 'gap' in str(memory_sampler):
            gap_size = int(memory_sampler[-1])
            between_pos_embedding = self.pos_embedding[between_memory_index[:,::gap_size].long()].permute([1,0,2])
            inside_pos_embedding = self.pos_embedding[inside_memory_index[:,::gap_size].long()].permute([1,0,2])
                
        return torch.cat((between_pos_embedding, inside_pos_embedding), dim=2)