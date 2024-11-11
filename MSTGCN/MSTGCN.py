import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import LayerNorm
import numpy as np

class TemporalAttention(nn.Module):
    def __init__(self, num_of_timesteps, num_of_vertices,num_of_features):#num_of_features
        super(TemporalAttention, self).__init__()
        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features

        self.U_1 = nn.Parameter(torch.empty(num_of_vertices, 1))
        self.U_2 = nn.Parameter(torch.empty(num_of_features, num_of_vertices))
        self.U_3 = nn.Parameter(torch.empty(num_of_features))
        self.b_e = nn.Parameter(torch.empty(1, num_of_timesteps, num_of_timesteps))
        self.V_e = nn.Parameter(torch.empty(num_of_timesteps, num_of_timesteps))

        nn.init.uniform_(self.U_1)
        nn.init.uniform_(self.U_2)
        nn.init.uniform_(self.U_3)
        nn.init.uniform_(self.b_e)
        nn.init.uniform_(self.V_e)

    def forward(self, x):
        #print("mstgcnx",x.size())
        #print(self.U_1.size())
        batch_size, T, V, F = x.size()

        # lhs operations
        lhs = torch.matmul(x.permute(0, 1, 3, 2), self.U_1)  # Permute to adjust dimensions

        lhs = lhs.view(batch_size, T, F)

        lhs = torch.matmul(lhs, self.U_2)#self.U_2
        #print("3:",lhs.size()) 1 5 26
        # rhs operations
        rhs = torch.matmul(self.U_3, x.permute(2, 0, 3, 1))#SELF.U_3
        #print("4:",rhs.size())
        rhs = rhs.permute(1, 0, 2)

        # Product and attention scores calculation
        product = torch.bmm(lhs, rhs)
        S = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e).permute(1, 2, 0)).permute(2, 0, 1)

        # normalization
        S = S - torch.max(S, dim=1, keepdim=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)
        #print("S:",S_normalized.size())
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])



class SpatialAttention(nn.Module):
    '''
        compute spatial attention scores
        --------
        Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
        Output: (batch_size, num_of_vertices, num_of_vertices)
        '''

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features):
        super(SpatialAttention, self).__init__()
        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features

        self.W_1 = nn.Parameter(torch.empty(num_of_timesteps, 1))
        self.W_2 = nn.Parameter(torch.empty(num_of_features, num_of_timesteps))
        self.W_3 = nn.Parameter(torch.empty(num_of_features,))
        self.b_s = nn.Parameter(torch.empty(1, num_of_vertices, num_of_vertices))
        self.V_s = nn.Parameter(torch.empty(num_of_vertices, num_of_vertices))

        nn.init.uniform_(self.W_1)
        nn.init.uniform_(self.W_2)
        nn.init.uniform_(self.W_3)
        nn.init.uniform_(self.b_s)
        nn.init.uniform_(self.V_s)

    def forward(self, x):
        batch_size, T, V, Fea = x.size()
        #print("x:",x.size())
        #print("W1:",self.W_1.size())
        #print(x.permute(0, 2, 3, 1).size())
        # lhs operations
        lhs = torch.matmul(x.permute(0, 2, 3, 1), self.W_1)
        #print("1:",lhs.size())
        lhs = lhs.view(batch_size, V, Fea)#降维
        lhs = torch.matmul(lhs, self.W_2)#.

        # rhs operations
        rhs = torch.matmul(self.W_3, x.permute(1, 0, 3, 2))#self.
        rhs = rhs.permute(1, 0, 2)

        # Product and attention scores calculation
        product = torch.bmm(lhs, rhs)
        S = torch.matmul(self.V_s, torch.sigmoid(product + self.b_s).permute(1, 2, 0)).permute(2, 0, 1)

        # normalization
        S = S - torch.max(S, dim=1, keepdim=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)

        return S_normalized



class Graph_Learn(nn.Module):
    '''
        Graph structure learning (based on the middle time slice)
        --------
        Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
        Output: (batch_size, num_of_vertices, num_of_vertices)
        '''
    def __init__(self, num_of_vertices, num_of_features, alpha):
        super(Graph_Learn, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.alpha = alpha
        self.a = nn.Parameter(torch.rand(num_of_features, 1))

    def forward(self, x):
        #print("zbc",x.size())
        batch_size, num_of_timesteps, num_of_vertices , num_feature= x.shape
        S_all = []
        for time_step in range(num_of_timesteps):
            xt = x[:, time_step, :, :]
            diff = xt.unsqueeze(2) - xt.unsqueeze(1)
            abs_diff = torch.abs(diff)
            diff_exp = torch.exp(-torch.sum(abs_diff * self.a.squeeze(-1), dim=-1))#self.a
            S = diff_exp / torch.sum(diff_exp, dim=2, keepdim=True)
            S_all.append(S.unsqueeze(1))

        S_stack = torch.cat(S_all, dim=1)
        return S_stack

    # def extra_repr(self):
    #     return 'alpha={}, num_of_vertices={}, num_of_features={}'.format(self.alpha, self.num_of_vertices,
    #                                                                      self.num_of_features)



class ChebConvWithAttGL(nn.Module):
    def __init__(self, num_of_filters, K, F):
        super(ChebConvWithAttGL, self).__init__()
        self.K = K
        self.F = F
        self.num_of_filters = num_of_filters
        self.Theta = nn.Parameter(torch.empty(K, F, num_of_filters))
        nn.init.uniform_(self.Theta, -1., 1.)
    def reset_parameters(self):
        for theta in self.Theta:
            stdv = 1. / (theta.size(0) ** 0.5)
            theta.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        x, A, S = inputs
        batch_size, T, V, num_features = x.shape


        # Ensure that the inputs are lists and have the correct length
        assert isinstance(inputs, list) and len(
            inputs) == 3, 'Input error: ChebConvWithAttGL expects a list of three elements (x, A, S)'
        #print("SORIGINAL:",S.size())#1 26 9
        batch_size, num_of_vertices = x.shape[0], x.shape[2]
        #print("SSS:", S.size()) 1 5 26 26
        # Symmetrize S
        S = torch.min(S, S.transpose(2, 3))
        #print("SSS:",S.size())
        out = []
        for time_step in range(x.shape[1]):
            graph_signal = x[:, time_step, :, :]
            Tx_0 = graph_signal #1 26 9
            #print("S:",S.size())
            #print(S[:, time_step, :, :].size())1,26,26
            Tx_1 = torch.matmul(S[:, time_step, :, :], graph_signal)
            #print("TX1:",Tx_1.size())1,26,9
            #print("Theta:",self.Theta[0].size())1,10   26x9 and 1x10
            out_time_step = torch.matmul(Tx_1, self.Theta[0])
            #print("ots:", out_time_step.size())
            for k in range(2, self.K):
                Tx_2 = 2 * torch.matmul(S[:, time_step, :, :], Tx_1) - Tx_0
                out_time_step += torch.matmul(Tx_2, self.Theta[k - 1])
                Tx_0, Tx_1 = Tx_1, Tx_2

            out.append(out_time_step.unsqueeze(1))

        return F.relu(torch.cat(out, dim=1))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 3
        x_shape = input_shape[0]
        return (x_shape[0], x_shape[1], x_shape[2], self.num_of_filters)




class ChebConvWithAttStatic(nn.Module):
    def __init__(self, num_of_filters, k, cheb_polynomials, num_features):
        super(ChebConvWithAttStatic, self).__init__()
        self.num_features = num_features
        self.k = k
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = [torch.tensor(p, dtype=torch.float32) for p in cheb_polynomials]
        self.Theta = nn.ParameterList([nn.Parameter(torch.randn(num_features, num_of_filters)) for _ in range(k)])

    def forward(self, inputs):
        x, Att = inputs
        x.to(Att.device)
        batch_size, T, V, num_features = x.shape

        outputs = []
        for time_step in range(T):
            graph_signal = x[:, time_step, :, :]  # (batch_size, V, F)
            output = torch.zeros((batch_size, V, self.num_of_filters), device=x.device)

            for kk in range(self.k):
                T_k = self.cheb_polynomials[kk]  # (V, V)
                T_k_with_at = T_k * Att  # Element-wise multiplication (batch_size, V, V)
                #print("T_k_with_at:",T_k_with_at.size())
                T_k_with_at = F.dropout(T_k_with_at, 0.6)  # Apply dropout

                # Batch matrix multiplication
                rhs = torch.matmul(T_k_with_at, graph_signal)  # (batch_size, V, F)
                output += torch.matmul(rhs, self.Theta[kk])  # (batch_size, V, num_of_filters)

            outputs.append(output.unsqueeze(-1))

        # Concatenate along the new time axis and apply ReLU
        outputs = torch.cat(outputs, dim=-1)  # (batch_size, V, num_of_filters, T)
        outputs = outputs.permute(0, 3, 1, 2)  # (batch_size, T, V, num_of_filters)
        return F.relu(outputs)

    def reset_parameters(self):
        for theta in self.Theta:
            stdv = 1. / (theta.size(0)**0.5)
            theta.data.uniform_(-stdv, stdv)





class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradReverse.apply(x, self.alpha)



class MSTGCN_Block(nn.Module):
    def __init__(self, num_features, num_vertices, K, num_of_chev_filters, num_of_time_filters, time_conv_kernel, time_conv_strides, num_of_timesteps, GLaplha,cheb_polynomials):
        super(MSTGCN_Block, self).__init__()
        self.num_of_timesteps = num_of_timesteps
        self.num_features = num_features
        self.GLaplha = GLaplha
        self.num_vertices = num_vertices
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.num_of_chev_filters = num_of_chev_filters
        self.num_of_time_filters = num_of_time_filters
        self.time_conv_kernel = time_conv_kernel
        self.time_conv_strides = time_conv_strides
        self.temporal_attention = TemporalAttention(num_of_timesteps, num_vertices,num_features)#, num_features
        self.spatial_attention = SpatialAttention(num_of_timesteps, num_vertices, num_features)
        self.graph_learn = Graph_Learn(num_vertices,num_features,alpha=GLaplha)
        self.cheb_conv_att_gl = ChebConvWithAttGL(num_of_chev_filters, K, num_features)
        self.cheb_conv_att_static = ChebConvWithAttStatic(num_of_chev_filters, K, cheb_polynomials, num_features)
        self.time_conv_gl = nn.Conv2d(num_of_chev_filters, num_of_time_filters, (time_conv_kernel, 1), stride=(time_conv_strides, 1), padding='same')
        self.time_conv_sd = nn.Conv2d(num_of_chev_filters, num_of_time_filters, (time_conv_kernel, 1), stride=(time_conv_strides, 1), padding='same')
        self.layer_norm_gl = LayerNorm(num_of_time_filters)
        self.layer_norm_sd = LayerNorm(num_of_time_filters)

    def forward(self, x):

        x = x.to(self.cheb_polynomials[0].device)
        temporal_att = self.temporal_attention(x)
        x_TAtt = reshape_dot(x, temporal_att)
        spatial_att = self.spatial_attention(x_TAtt)
        S = self.graph_learn(x)
        #print("S:",S.type())
        S = F.dropout(S, 0.3)
        #print("SS:",S.size())
        spatial_gcn_GL = self.cheb_conv_att_gl([x, spatial_att, S])
        spatial_gcn_SD = self.cheb_conv_att_static([x, spatial_att])
        spatial_gcn_GL = spatial_gcn_GL.permute(0, 3, 1, 2)
        time_conv_output_GL = self.time_conv_gl(spatial_gcn_GL)
        spatial_gcn_SD = spatial_gcn_SD.permute(0, 3, 1, 2)
        time_conv_output_SD = self.time_conv_sd(spatial_gcn_SD)
        time_conv_output_GL = time_conv_output_GL.permute(0, 2, 3, 1)
        end_output_GL = self.layer_norm_gl(time_conv_output_GL)
        time_conv_output_SD = time_conv_output_SD.permute(0, 2, 3, 1)
        end_output_SD = self.layer_norm_sd(time_conv_output_SD)
        return end_output_GL, end_output_SD

class MSTGCN(nn.Module):
    def __init__(self, num_vertices, num_features, num_timesteps, num_blocks,num_of_chev_filters ,num_of_time_filters,time_conv_kernel,time_conv_strides, K, GLaplha, cheb_polynomials, num_class, domin):
        super(MSTGCN, self).__init__()
        self.time_conv_strides = time_conv_strides
        self.cheb_polynomials = cheb_polynomials
        self.time_conv_kernel = time_conv_kernel
        self.num_vertices = num_vertices
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.num_of_chev_filters = num_of_chev_filters
        self.num_of_time_filters = num_of_time_filters
        self.num_blocks = num_blocks
        self.K = K
        self.GLaplha = GLaplha
        self.num_domains = domin
        self.num_class = num_class
        #####################
        self.mstgcn_block = nn.ModuleList([MSTGCN_Block(num_features, num_vertices, K, num_of_chev_filters, num_of_time_filters, time_conv_kernel , time_conv_strides, num_timesteps, GLaplha,cheb_polynomials)
                                            for _ in range(num_blocks)])

        self.grl = GradientReversal()
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_of_time_filters * num_timesteps * num_vertices * 2, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_domains)
        )

        self.feature_classifier = nn.Sequential(
            nn.Linear(2 * num_of_time_filters * num_timesteps * num_vertices, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_class)  # Assuming binary classification for features
        )

    def forward(self, x):
        B, V, Fea = x.size()
        T = self.num_timesteps
        x = x.view(B, T, V, Fea)
        x = x.to(self.cheb_polynomials[0].device)

        # Process the input through the first block
        block_out_GL, block_out_SD = self.mstgcn_block[0](x)

        # Process the output through the rest of the blocks
        for block in self.mstgcn_block[1:]:
            block_out_GL, block_out_SD = block(block_out_GL)

            # Concatenate outputs from the last block processed
        block_out = torch.cat([block_out_GL, block_out_SD], dim=1)
        block_out = block_out.view(block_out.size(0), -1)  # Flatten the output for further processing or a final layer

        #print("block_out (before dropout):", block_out.size(), block_out.type())  # Ensure block_out is a tensor
        #assert isinstance(block_out, torch.Tensor), "block_out is not a tensor"


        block_out = F.dropout(block_out, 0.5, training=self.training)

        block_out_grl = self.grl(block_out)
        feature_output = self.feature_classifier(block_out)  # Perform classification
        domain_output = self.domain_classifier(block_out_grl)  # Perform domain classification
        return feature_output, domain_output

        #block_out_reserve = self.grl(block_out)
        #feature_output = self.feature_classifier(block_out)  # Perform classification
        #domain_output = self.domain_classifier(self.grl(block_out))  # Perform domain classification



        #return feature_output#, domain_output



# Instantiate the model


def diff_loss(output, target):
    return torch.mean((output - target) ** 2)

def F_norm_loss(matrix):
    return torch.norm(matrix, p='fro')

# def reshape_dot(x, y):
#     print("x shape before reshape:", x.shape)  # 打印x的原始形状
#     print("y shape before reshape:", y.shape)  # 打印y的原始形状
#     print("x shape after reshape:", x.view(x.size(0), 1, -1).shape)  # 打印x的形状
#     print("y shape after reshape:", y.view(y.size(0), -1, 1).shape)  # 打印y的形状
#     return torch.bmm(x.view(x.size(0), 1, -1), y.view(y.size(0), -1, 1)).squeeze()


def reshape_dot(x, TAtt):
    # x and TAtt are expected to be torch tensors
    # x: shape [batch_size, num_timesteps, num_vertices, num_features]
    # TAtt: shape should be compatible for batch matrix multiplication with reshaped x
    # Transposing x to bring features dimension to second-to-last position for matrix multiplication
    x_transposed = x.permute(0, 2, 3, 1)  # shape [batch_size, num_vertices, num_features, num_timesteps]
    # Flattening the first three dimensions of x for batch dot product
    x_reshaped = x_transposed.reshape(x.size(0), -1, x.size(1))  # shape [batch_size, num_vertices*num_features, num_timesteps]
    # Performing batch matrix multiplication
    # TAtt expected shape: [batch_size, num_timesteps, ?] - the second dimension of TAtt must align with the last of x_reshaped
    result = torch.bmm(x_reshaped, TAtt)
    # Reshaping result back to original x dimensions [batch_size, num_timesteps, num_vertices, num_features]
    result_reshaped = result.reshape(-1, x.shape[1], x.shape[2], x.shape[3])
    return result_reshaped





