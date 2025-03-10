from .runner import run_net
from .runner import test_net
from .runner_BERT_pretrain import run_net as BERT_pretrain_run_net
from .runner_BERT_finetune import run_net as BERT_finetune_run_net
from .runner_BERT_fewshot import run_net as BERT_fewshot_run_net
from .runner_BERT_finetune import test_net as BERT_test_run_net
from .runner_BERT_fewshot import test_net as BERT_fewshot_test_net