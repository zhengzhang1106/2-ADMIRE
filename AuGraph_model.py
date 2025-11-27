from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.utils import get_filter_config

torch, nn = try_import_torch()

class AuGraphModel(TorchModelV2, nn.Module):
    """
    状态空间：'link'9*9矩阵  'request'信息
    动作空间：5个连续动作从0-255*5，分别表示 0.疏导边; 1.发射器边; 2.接收器边; 3.光路边; 4.波长链路边的权重

    数据流向：
    （一）
    1：'link'经过两层CNN，最后经过flatten层展平
    2：'request'经过一层全连接

    （二）
    将上述两个输入拼接，传入actor和critic网络的隐藏层fc
    需要把输出变为一维的，作为actor和critic网络的输入
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.original_space = obs_space.original_space

        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # print("========num_outputs========", num_outputs)  # 256

        # =======================CNN===================================
        concat_size = 0  # 记录处理后连接起来的总长度
        # 第一步：CNN处理'link'
        cnn_config = {
            "conv_filters": model_config["conv_filters"]
            if "conv_filters" in model_config else
            get_filter_config(obs_space.shape),
            "conv_activation": model_config.get("conv_activation"),
            "post_fcnet_hiddens": []
        }

        layers_cnn = []     # 存cnn模型
        (dim, w, h,) = self.original_space['phylink'].shape   # (72,9,9)
        # print('开始输出self.original_space[link].shape')
        # print("dim,w,h",dim,w,h)

        in_size_cnn = [w, h]    # (9,9)
        in_channels = dim     # 输入是72维

        for out_channels, kernel, stride in cnn_config["conv_filters"][:-1]:  # 除了最后一个卷积核都进行了操作
            padding, out_size_cnn = same_padding(in_size_cnn, kernel, stride)
            layers_cnn.append(
                SlimConv2d(
                    in_channels=in_channels,    # 输入维度
                    out_channels=out_channels,   # 输出维度（滤波器个数）
                    kernel=kernel,  # 卷积核
                    stride=stride,  # 步长
                    padding=padding,  # 填边
                    activation_fn=cnn_config["conv_activation"]    # 激活函数
                )
            )
            in_channels = out_channels  # 下一个CNN的输入维度是上一个的输出维度
            in_size_cnn = out_size_cnn  # CNN的长和宽

        out_channels, kernel, stride = cnn_config["conv_filters"][-1]  # 开始处理最后一个卷积核
        layers_cnn.append(
            SlimConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                padding=None,  # padding=valid
                activation_fn=cnn_config["conv_activation"]
                )
        )
        # 将CNN展平，此处直接用flatten的前提是最后的形状为[batch, number_of_filters, 1, 1]，也就是说不用管滤波器的展平层了
        layers_cnn.append(nn.Flatten())

        self._convs = nn.Sequential(*layers_cnn)  # 组成CNN神经网络序列，卷积层+激活层+展平层...，会得到CNN输出
        # print("self._convs",self._convs)
        concat_size += out_channels*8*8   # FC输入神经元个数,out_channels=5,concat_size=5*8*8,根据输出结果调整的
        # print("concat_size_init",concat_size)
        # 第二步：处理'request_index'：1
        # concat_size += len(self.original_space['request_index'])    # 不处理
        # 第三步：one_hot处理'request_src'：9
        concat_size += (self.original_space['request_src'].high - self.original_space['request_src'].low)[0] + 1
        # 第四步：one_hot处理'request_dest'：9
        concat_size += (self.original_space['request_dest'].high - self.original_space['request_dest'].low)[0] + 1
        # 第五步：处理'request_traffic'：1
        concat_size += self.original_space['request_traffic'].shape[0]
        # print('流量concate_size',self.original_space['request_traffic'].shape[0]) #24
        # 第六步：此处表示前面经过CNN、ONE_HOT之后需要送到多层的（目前就一层）全连接层中，应该定义一个这样的全连接层
        # 这个FC的输出要作为actor和critic隐藏层FC的输入
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens"),
            "fcnet_activation": model_config.get("post_fcnet_activation"),
        }

        post_fc_layers = []
        in_size = concat_size   # FC输入层神经元个数，
        # print("concat_size_end", concat_size)  #426
        for i, out_size in enumerate(post_fc_stack_config['fcnet_hiddens']):
            post_fc_layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn=post_fc_stack_config["fcnet_activation"],
                    # if i < len(post_fc_stack_config['fcnet_hiddens']) - 1 else None,  # 除了最后一层都加激活函数
                    initializer=normc_initializer(1.0)
                )
            )
            in_size = out_size
        self._hidden = nn.Sequential(*post_fc_layers)   # 全连接层输出
        # print(self._hidden)

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None
        # # Actions and value heads
        # if num_outputs:
        #     print('我进来了！！！！！！！！！！！！')
        #     print(num_outputs)  # 256
        #     self.logits_layer = SlimFC(
        #         in_size=out_size,
        #         out_size=num_outputs,
        #         activation_fn=None,
        #     )
        #     # Create the value branch model.
        #     self.value_layer = SlimFC(
        #         in_size=out_size,
        #         out_size=1,
        #         activation_fn=None,
        #         initializer=normc_initializer(0.01))  # 正态分布初始化

    @override(ModelV2)  # 子类覆写父类函数
    def forward(self, input_dict, state, seq_lens):
        # print('已进入forward阶段！')
        phylink = input_dict["obs"]["phylink"]
        # print(phylink)
        # print("phylink",phylink.shape)  # torch.Size([32, 1, 3, 9, 9])
        request_index = input_dict["obs"]["request_index"]
        # print(request_index)
        # print("request_index",request_index.shape)  # torch.Size([32, 1])
        request_src = input_dict["obs"]["request_src"]
        # print(request_src)
        # print("request_src",request_src.shape)  # torch.Size([32, 1])
        request_dest = input_dict["obs"]["request_dest"]
        # print(request_dest)
        # print("request_dest",request_dest.shape)  # torch.Size([32, 1])
        request_traffic = input_dict["obs"]["request_traffic"]
        # print("业务大小",request_traffic)
        # print("request_traffic",request_traffic.shape)  # torch.Size([32, 1])

        outs = []   # 记录输出，送入fc
        out_cnn = self._convs(phylink)
        # print('现在输出卷积层输出尺寸')
        # print(out_cnn.shape)
        outs.append(out_cnn)
        # outs.append(request_index)

        # 源目的节点one-hot编码
        out_onehot_src = nn.functional.one_hot(request_src.long(), (self.original_space['request_src'].high - self.original_space['request_src'].low)[0] + 1)
        out_onehot_src = out_onehot_src.squeeze(dim=1).float()  # 去除空维度
        # print('现在输出onehot输出尺寸')  #) torch.Size([32, 9])
        # print(out_onehot_src.shape)
        outs.append(out_onehot_src)

        out_onehot_dest = nn.functional.one_hot(request_dest.long(), (self.original_space['request_dest'].high - self.original_space['request_dest'].low)[0] + 1)
        out_onehot_dest = out_onehot_dest.squeeze(dim=1).float()  # 去除空维度
        # print('现在输出onehot输出尺寸')
        # print('out_onehot_dest',out_onehot_dest)
        # print('out_onehot_dest_shape',out_onehot_dest.shape)
        outs.append(out_onehot_dest)
        # print('request_traffic', request_traffic)
        outs.append(request_traffic)    # 目前流量就直接加进去了

        out_add = torch.cat(outs, dim=1)  # 拼接在一起
        # print('现在输出三个参量拼接后的输出尺寸')
        # print(out_add.shape)  # torch.Size([1, 426])

        out = self._hidden(out_add)
        # print('现在输出全连接的输出尺寸')
        # print(out)
        # print(out.shape) # torch.Size([1, 256])
        # if not self.logits_layer is None:
        #     logits, values = self.logits_layer(out), self.value_layer(out)
        #     print(logits.shape)   # torch.Size([1, 256])
        #     self._value_out = torch.reshape(values, [-1])  # 表示将矩阵形式的value展平
        #     # return logits + inf_mask, []
        #     return logits, []
        # else:
        return out, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out