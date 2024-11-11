
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data.data_utils import computeFFT
from model.cell import DCGRUCell
from torch.autograd import Variable
import utils
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.sparse as sp
from torch.nn.modules.module import Module


def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor
    """
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x)
                     for x in tup)
    else:
        return fn(tup)


def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor
    """
    if isinstance(tups[0], tuple):
        return tuple(
            (torch.cat(
                xs,
                dim) if isinstance(
                xs[0],
                torch.Tensor) else xs[0]) for xs in zip(
                *
                tups))
    else:
        return torch.cat(tups, dim)


class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='laplacian',
                 device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        num_nodes = inputs.shape[2]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = torch.zeros((self.num_rnn_layers,batch_size,num_nodes*self.hid_dim), device=self._device)
        # (num_layers, batch_size, num_nodes * rnn_units)
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            # (seq_len, batch_size, num_nodes * rnn_units)
            output_inner = torch.zeros((seq_length,batch_size,num_nodes*self.hid_dim), device=self._device)
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    supports, current_inputs[t, ...], hidden_state)
                output_inner[t] = hidden_state
            current_inputs = output_inner
            output_hidden[i_layer] = hidden_state

        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, dcgru_activation=None,
                 filter_type='laplacian', device=None, dropout=0.0):
        super(DCGRUDecoder, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device
        self.dropout = dropout

        cell = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                         max_diffusion_step=max_diffusion_step,
                         num_nodes=num_nodes, nonlinearity=dcgru_activation,
                         filter_type=filter_type)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            decoding_cells.append(cell)

        self.decoding_cells = nn.ModuleList(decoding_cells)
        self.projection_layer = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)  # dropout before projection layer

    def forward(
            self,
            inputs,
            initial_hidden_state,
            supports,
            teacher_forcing_ratio=None):
        """
        Args:
            inputs: shape (seq_len, batch_size, num_nodes, output_dim)
            initial_hidden_state: the last hidden state of the encoder, shape (num_layers, batch, num_nodes * rnn_units)
            supports: list of supports from laplacian or dual_random_walk filters
            teacher_forcing_ratio: ratio for teacher forcing
        Returns:
            outputs: shape (seq_len, batch_size, num_nodes * output_dim)
        """
        seq_length, batch_size, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        go_symbol = torch.zeros(
            (batch_size,
             self.num_nodes *
             self.output_dim)).to(
            self._device)

        # tensor to store decoder outputs
        outputs = torch.zeros(
            seq_length,
            batch_size,
            self.num_nodes *
            self.output_dim).to(
            self._device)

        current_input = go_symbol  # (batch_size, num_nodes * input_dim)
        for t in range(seq_length):
            next_input_hidden_state = []
            next_input_hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.num_nodes*self.hid_dim),device=self._device)
            for i_layer in range(0, self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    supports, current_input, hidden_state)
                current_input = output
                next_input_hidden_state[i_layer] = hidden_state
            initial_hidden_state = next_input_hidden_state

            projected = self.projection_layer(self.dropout(
                output.reshape(batch_size, self.num_nodes, -1)))
            projected = projected.reshape(
                batch_size, self.num_nodes * self.output_dim)
            outputs[t] = projected

            if teacher_forcing_ratio is not None:
                teacher_force = random.random() < teacher_forcing_ratio  # a bool value
                current_input = (inputs[t] if teacher_force else projected)
            else:
                current_input = projected

        return outputs
class ReadoutLayer1(Module):
  def __init__(self, D_in, D_out, D_node, D_len, device, drop = 0.5):
    super(ReadoutLayer1, self).__init__()
    """
    textING ReadoutLayer
    """
    self.num_nodes = D_node

    self.dropout = nn.Dropout(p = drop)
    self.len = D_len
    self._device = device

    self.tanh = nn.Tanh()
    self.sigmoid = nn.Sigmoid()

    D_hid = int(D_in/2)
    self.attention1 = nn.Linear(D_in, D_hid)
    self.attention2 = nn.Linear(D_in, D_hid)
    self.fc2 = nn.Linear(D_in, D_out)
    self.relu = nn.ReLU()
    self.weight_len = nn.Parameter(torch.zeros(size=(D_hid, 1))) # [48/2, 1]
    nn.init.xavier_uniform_(self.weight_len.data, gain=1.414)
    self.weight_node = nn.Parameter(torch.zeros(size=(D_hid, 1))) # [48/2, 1]
    nn.init.xavier_uniform_(self.weight_node.data, gain=1.414)
    self.attention = nn.Linear(D_in, D_in)
    self.embedding = nn.Linear(D_in, D_in)
    self.tanh = nn.Tanh()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x, seq_lengths):
    # soft attention
    # h: (batch_size, self.len, self.num_nodes, self.rnn_units)
    # generate sequence tensor : [0,1,2,...,11]
    x = torch.transpose(x, dim0=2, dim1=0) # x : [19, 12, 40, 48]
    num_nodes, seq_len, batch, input_dim = x.size()
    output_list = []
    r = torch.as_tensor(range(self.len)).to(self._device)
    mask = r.repeat(batch,1) # [batch,sel_len]
    mask = torch.unsqueeze(mask,2) # [batch, seq_len, 1]
    ones_tensor = torch.ones(batch, seq_len, 1).float().to(self._device)
    lengths = seq_lengths.reshape(batch, 1, 1).float()
    thres = torch.matmul(ones_tensor, lengths)
    # final mask: [batch, seq_len, 1], value 1 for seizure time step, value -10 for non-seizure time step
    final_mask = 11*torch.lt(mask,thres) - 10 

    # attention over time/seq_len
    for i in range(num_nodes):
        Xi = x[i,:] # [12, 40, 48]     
        output = Xi.permute(1,2,0).contiguous() # [40,48,12]
        # mask attention
        hc = torch.transpose(output, dim0=1, dim1=2) # [40, 12, 48]
        output_C = self.relu(self.attention1(hc)) # (batch_size, seq_len, D_hid)
        attention_C = torch.matmul(output_C, self.weight_len) # (batch_size, seq_len, 1)
        # mask attention
        attention_C = torch.mul(attention_C,final_mask) # element-wise mul
        # attention score
        attention_C = F.softmax(attention_C,dim=1) # (batch_size, seq_len, 1)
        hc = torch.squeeze(torch.matmul(output, attention_C)) # (batch_size, C_in) [40,48]
        output_list.append(hc) # [40,48]
    output_tensor = torch.stack(output_list) # [19, 40, 48]
    output_tensor = torch.transpose(output_tensor, dim0=1, dim1=0) # [40,19,48]

    # attention over num_nodes
    att = self.sigmoid(self.attention(output_tensor))
    emb = self.tanh(self.embedding(output_tensor))
    # graph summation, equation (6)
    g = torch.mul(att, emb) # g: batch, num_nodes, num_hidden

    # g: batch, num_nodes, num_hidden
    g = self.dropout(g)

    # classification
    # g: batch, num_nodes, num_class
    g = self.fc2(g)

    # max-pooling
    m, _ = torch.max(g, dim = 1)
    # average + max-pooling, equation (7)
    final_logits = torch.sum(g, dim = 1) / num_nodes + m 

    return final_logits

########## Model for seizure classification/detection ##########
class DCRNN_all_classification(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(DCRNN_all_classification, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        max_diffusion_step = args.max_diffusion_step
        self.top_k = args.top_k

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes
        self.len = args.max_seq_len

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=self._device)

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.GRU = nn.GRU(enc_input_dim, self.rnn_units)
        D_hid = int(self.rnn_units/2)
        self.attention = nn.Linear(self.rnn_units, D_hid)
        self.weight_len_C = nn.Parameter(torch.zeros(size=(D_hid, 1))) # [48/2, 1]
        nn.init.xavier_uniform_(self.weight_len_C.data, gain=1.414)
        self.weight_key = nn.Parameter(torch.zeros(size=(self.rnn_units, int(self.rnn_units/2)))) # [48,24]
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.rnn_units, int(self.rnn_units/2)))) # [48, 24]
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.readout = ReadoutLayer1(D_in = self.rnn_units, 
                                    D_out = self.num_classes, 
                                    D_node = self.num_nodes,
                                    D_len = self.len,
                                    device = self._device,
                                    drop = args.dropout)


    def calculate_random_walk_matrix(self, adj_mx):
        """
        State transition matrix D_o^-1W in paper.
        """
        # degree of each node
        d = torch.sum(adj_mx, 1)
        # 1/degree
        d_inv = torch.pow(d,-1)
        # set +/- inf value to zero
        d_inv[torch.isinf(d_inv)] = 0.
        # diagonal matrix
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.matmul(d_mat_inv,adj_mx) # [19,19]
        return random_walk_mx

    # graph construction: GRU (bidirectional=False) -> MLP -> Attention
    def self_graph_attention(self, x, seq_lengths):
        # x : shape (batch, seq_len, num_nodes, input_dim) [40, 12, 19, 100]
        # seq_length: real lengths for each EEG clip (batch_size,)
        # Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # GRU input: L - sequence length, N - batch size, Hin - input size
        x = torch.transpose(x, dim0=2, dim1=0) # x : [19, 12, 40, 100]
        num_nodes, seq_len, batch, input_dim = x.size()
        output_list = []

        # generate sequence tensor : [0,1,2,...,11]
        r = torch.as_tensor(range(self.len)).to(self._device)
        mask = r.repeat(batch,1) # [batch,sel_len]
        mask = torch.unsqueeze(mask,2) # [batch, seq_len, 1]
        ones_tensor = torch.ones(batch, seq_len, 1).float().to(self._device)
        lengths = seq_lengths.reshape(batch, 1, 1).float()
        thres = torch.matmul(ones_tensor, lengths)
        # final mask: [batch, seq_len, 1], value 1 for seizure time step, -10 for non-seizure time step
        final_mask = 11*torch.lt(mask,thres) - 10 

        # attention over time/seq_len
        for i in range(num_nodes):
            Xi = x[i,:] # [12, 40, 100]     
            output, h_n = self.GRU(Xi) 
            # GRU ouput: L - sequence length, N - batch size, Hout - hidden size, [12, 40, 48]
            output = output.permute(1,2,0).contiguous() # [40,48,12]
            # TODO: mask attention
            #output = self.attention(output)# [40,48,1]
            hc = torch.transpose(output, dim0=1, dim1=2) # [40, 12, 48]
            output_C = self.relu(self.attention(hc)) # (batch_size, seq_len, D_hid)
            attention_C = torch.matmul(output_C, self.weight_len_C) # (batch_size, seq_len, 1)
            # mask attention
            attention_C = torch.mul(attention_C,final_mask) # element-wise mul
            # attention score
            attention_C = F.softmax(attention_C,dim=1) # (batch_size, seq_len, 1)
            hc = torch.squeeze(torch.matmul(output, attention_C)) # (batch_size, C_in) [40,48]
            output_list.append(hc) # [40,48]
        output_tensor = torch.stack(output_list) # [19, 40, 48]
        output_tensor = torch.transpose(output_tensor, dim0=1, dim1=0) # [40,19,48]

        # attention over space/nodes
        key = torch.matmul(output_tensor, self.weight_key) # [40, 19, 48]* [48, 24] = [40, 19, 24]
        query = torch.matmul(output_tensor, self.weight_query) # [40, 19, 48]* [48, 24] = [40, 19, 24]

        query = query.permute(0,2,1).contiguous() # [40, 24, 19]
        data = torch.matmul(key,query)/np.sqrt(self.rnn_units) # [40, 19, 19]

        attention = F.softmax(data, dim = 2) # [40, 19, 19]
        # keep top k neighbors
        # sort each row in descending order and select topk
        a,_ = attention.topk(k = self.top_k,dim = 2) # [40, 19, k]
        # minimum value of each top k
        a_min = torch.min(a,dim=-1).values # [40,19]
        a_min = a_min.unsqueeze(-1).repeat(1,1,num_nodes) # [40, 19, 19]
        # compare attention with a_min
        ge = torch.ge(attention,a_min)
        zero = torch.zeros_like(attention)
        # ge: condition, a: attention, b: zero
        result = torch.where(ge,attention,zero)
        diag = torch.eye(num_nodes).to(self._device) # [19,19]
        diag = diag.unsqueeze(0).repeat(batch,1,1) # [40,19,19]
        result = torch.add(result,diag) # (batch, N, N)

        # calculate supports with Bidirectional random walk
        supports = []
        supports1 = torch.zeros((batch,num_nodes,num_nodes), device=self._device) # (batch, N, N)
        supports2 = torch.zeros((batch,num_nodes,num_nodes), device=self._device) # (batch, N, N)
        for i in range(batch):
            adj_mat = result[i]
            supports1[i] = self.calculate_random_walk_matrix(adj_mat).T
            supports2[i] = self.calculate_random_walk_matrix(adj_mat.T).T

        supports.append(supports1)
        supports.append(supports2)

        return supports

    def forward(self, input_seq, seq_lengths):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # construct graph with self-attention layer
        supports = self.self_graph_attention(input_seq, seq_lengths)

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(
            batch_size).to(self._device)

        # last hidden state of the encoder is the context
        # (max_seq_len, batch, rnn_units*num_nodes)
        _, final_hidden = self.encoder(input_seq, init_hidden_state, supports)

        # Readout
        # (batch_size, max_seq_len, rnn_units*num_nodes)
        output = torch.transpose(final_hidden, dim0=0, dim1=1)
        output = output.view(batch_size, self.len, self.num_nodes, self.rnn_units)

        pool_logits = self.readout(output,seq_lengths)

        return pool_logits



########## Model for next time prediction ##########
class DCRNNModel_nextTimePred(nn.Module):
    def __init__(self, args, device=None):
        super(DCRNNModel_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        dec_input_dim = args.output_dim
        output_dim = args.output_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = args.num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.output_dim = output_dim
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=self._device)
        self.decoder = DCGRUDecoder(input_dim=dec_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes, hid_dim=rnn_units,
                                    output_dim=output_dim,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=device,
                                    dropout=args.dropout)

        self.GRU = nn.GRU(enc_input_dim, self.rnn_units)
        self.relu = nn.ReLU()
        D_hid = int(self.rnn_units/2)
        self.attention = nn.Linear(self.rnn_units, D_hid)
        self.weight_len_C = nn.Parameter(torch.zeros(size=(D_hid, 1))) # [48/2, 1]
        nn.init.xavier_uniform_(self.weight_len_C.data, gain=1.414)
        self.weight_key = nn.Parameter(torch.zeros(size=(self.rnn_units, int(self.rnn_units/2)))) # [48,24]
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.rnn_units, int(self.rnn_units/2)))) # [48, 24]
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.top_k = args.top_k


    def calculate_random_walk_matrix(self, adj_mx):
        """
        State transition matrix D_o^-1W in paper.
        """
        # degree of each node
        d = torch.sum(adj_mx, 1)
        # 1/degree
        d_inv = torch.pow(d,-1)
        # set +/- inf value to zero
        d_inv[torch.isinf(d_inv)] = 0.
        # diagonal matrix
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.matmul(d_mat_inv,adj_mx) # [19,19]
        return random_walk_mx

    # graph construction: GRU (bidirectional=False) -> MLP -> Attention
    def self_graph_attention(self, x):
        # x : shape (batch, seq_len, num_nodes, input_dim) [40, 12, 19, 100]
        # seq_length: real lengths for each EEG clip (batch_size,)
        # Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # GRU input: L - sequence length, N - batch size, Hin - input size
        x = torch.transpose(x, dim0=2, dim1=0) # x : [19, 12, 40, 100]
        num_nodes, seq_len, batch, input_dim = x.size()
        output_tensor = torch.zeros((num_nodes,batch,self.rnn_units),device=self._device)

        # attention over time/seq_len
        for i in range(num_nodes):
            Xi = x[i,:] # [12, 40, 100]     
            output, h_n = self.GRU(Xi) 
            # GRU ouput: L - sequence length, N - batch size, Hout - hidden size, [12, 40, 48]
            output = output.permute(1,2,0).contiguous() # [40,48,12]
            # TODO: mask attention
            #output = self.attention(output)# [40,48,1]
            hc = torch.transpose(output, dim0=1, dim1=2) # [40, 12, 48]
            output_C = self.relu(self.attention(hc)) # (batch_size, seq_len, D_hid)
            attention_C = torch.matmul(output_C, self.weight_len_C) # (batch_size, seq_len, 1)
            # attention score
            attention_C = F.softmax(attention_C,dim=1) # (batch_size, seq_len, 1)
            hc = torch.squeeze(torch.matmul(output, attention_C)) # (batch_size, C_in) [40,48]
            output_tensor[i] = hc
        output_tensor = torch.transpose(output_tensor, dim0=1, dim1=0) # [40,19,48]

        # attention over space/nodes
        key = torch.matmul(output_tensor, self.weight_key) # [40, 19, 48]* [48, 24] = [40, 19, 24]
        query = torch.matmul(output_tensor, self.weight_query) # [40, 19, 48]* [48, 24] = [40, 19, 24]

        query = query.permute(0,2,1).contiguous() # [40, 24, 19]
        data = torch.matmul(key,query)/np.sqrt(self.rnn_units) # [40, 19, 19]

        attention = F.softmax(data, dim = 2) # [40, 19, 19]
        # keep top k neighbors
        # sort each row in descending order and select topk
        a,_ = attention.topk(k = self.top_k,dim = 2) # [40, 19, k]
        # minimum value of each top k
        a_min = torch.min(a,dim=-1).values # [40,19]
        a_min = a_min.unsqueeze(-1).repeat(1,1,num_nodes) # [40, 19, 19]
        # compare attention with a_min
        ge = torch.ge(attention,a_min)
        zero = torch.zeros_like(attention)
        # ge: condition, a: attention, b: zero
        result = torch.where(ge,attention,zero)
        diag = torch.eye(num_nodes).to(self._device) # [19,19]
        diag = diag.unsqueeze(0).repeat(batch,1,1) # [40,19,19]
        result = torch.add(result,diag) # (batch, N, N)

        # calculate supports with Bidirectional random walk
        supports = []
        supports1 = torch.zeros((batch,num_nodes,num_nodes), device=self._device) # (batch, N, N)
        supports2 = torch.zeros((batch,num_nodes,num_nodes), device=self._device) # (batch, N, N)
        for i in range(batch):
            adj_mat = result[i]
            supports1[i] = self.calculate_random_walk_matrix(adj_mat).T
            supports2[i] = self.calculate_random_walk_matrix(adj_mat.T).T

        supports.append(supports1)
        supports.append(supports2)

        return supports

    def forward(
            self,
            encoder_inputs,
            decoder_inputs,
            batches_seen=None):
        """
        Args:
            encoder_inputs: encoder input sequence, shape (batch, input_seq_len, num_nodes, input_dim)
            encoder_inputs: decoder input sequence, shape (batch, output_seq_len, num_nodes, output_dim)
            batches_seen: number of examples seen so far, for teacher forcing
        Returns:
            outputs: predicted output sequence, shape (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape


        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(batch_size).cuda()

        # construct graph with self-attention layer
        # supports: list of supports from dual_random_walk filters
        supports = self.self_graph_attention(encoder_inputs)
        # (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # encoder
        # (num_layers, batch, rnn_units*num_nodes)
        encoder_hidden_state, _ = self.encoder(
            encoder_inputs, init_hidden_state, supports)

        # decoder
        if self.training and self.use_curriculum_learning and (
                batches_seen is not None):
            teacher_forcing_ratio = utils.compute_sampling_threshold(
                self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None
        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio)  # (seq_len, batch_size, num_nodes * output_dim)
        # (seq_len, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs
########## Model for next time prediction ##########
