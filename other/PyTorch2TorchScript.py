import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

class CustomLlamaModel(LlamaModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids,
            attention_mask,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        hidden_states = outputs[0]
        next_cache = outputs[1] if use_cache else None
        all_hidden_states = outputs.hidden_states if output_hidden_states else None
        all_self_attns = outputs.attentions if output_attentions else None

        if not return_dict:
            output_tuple = (hidden_states, next_cache)
            if output_hidden_states:
                output_tuple += (all_hidden_states,)
            if output_attentions:
                output_tuple += (all_self_attns,)
            return output_tuple
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

# 使用自定义模型加载预训练权重
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
config = LlamaConfig.from_pretrained(model_name)
custom_model = CustomLlamaModel.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # 根据模型输出格式调整

model_wrapper = ModelWrapper(custom_model)

# 创建一个示例输入
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt")

# 使用script进行转换
scripted_model = torch.jit.script(model_wrapper)

# 保存TorchScript模型
scripted_model.save("TinyLlama-1.1B-Chat-v1.0-scripted.pt")
