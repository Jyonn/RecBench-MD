from model.lc_model import LongContextModel


class QWenModel(LongContextModel):
    PEFT_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


class QWen2TH7BModel(QWenModel):
    KEY = 'Qwen/Qwen2-7B-Instruct'


class QWen2TH1D5BModel(QWenModel):
    KEY = 'Qwen/Qwen2-1.5B-Instruct'


class QWen2TH0D5BModel(QWenModel):
    KEY = 'Qwen/Qwen2-0.5B-Instruct'


class DeepSeekR1QWen7BModel(LongContextModel):
    KEY = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
