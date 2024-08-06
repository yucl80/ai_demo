from transformers import AutoTokenizer, AutoModelForPreTraining
import torch
model_id="JetBrains-Research/cmg-codereviewer-without-history"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForPreTraining.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float).cpu()
input_text = "    代码:\n        ```java\n        \n     public class OrderService{\n   private double getOrderAmount(Order order){\n        double totalAmount = calcTotalAmout(order);\n        return totalAmount * getOrderDiscounts(totalAmount);\n    }   \n\n    private double getOrderDiscounts(double totalAmount) {\n        if (totalAmount >= 1000) {\n            return 0.85;\n        } else if (totalAmount > 500) {\n            return 0.8;\n        } else {\n            return 0.99;\n        }\n\n    }\n    }\n    \n        ```\n"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))