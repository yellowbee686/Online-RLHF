import json
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import re

tqdm.pandas()


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer



@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    url: Optional[str] = field(
        default="http://localhost",
        metadata={"help": "url of the model response"},
    )
    tokenizer: Optional[str] = field(
        default="HuggingFaceH4/mistral-7b-sft-beta",
        metadata={"help": "the tokenizer to use"},
    )
    ports: List[str] = field(default_factory=lambda: ["8000"], metadata={"help": "ports of the model response"})
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    dataset_name_or_path: Optional[str] = field(
        default="cornfieldrm/iterative-prompt-v1-iter1-2K",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="uf_split0_responses_K8.json",
        metadata={"help": "the location of the output file"},
    )
    bos_format: Optional[str] = field(
        default="",
        metadata={"help": "the format of the beginning of the sentence"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    max_workers: Optional[int] = field(
        default=1024,
        metadata={"help": "the number of workers"},
    )
    ds_split: Optional[str] = field(
        default="train",
        metadata={"help": "the split of dataset"},
    )
    eval_gsm: Optional[bool] = field(
        default=False,
        metadata={"help": "if run the eval gsm mode"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ds_dir = script_args.dataset_name_or_path
output_dir = script_args.output_dir
K = script_args.K
ports = script_args.ports

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)


def change_of_format(prom):
    # To be modified according to the reward model and the LLM you use
    # Be careful about multi-turn conversions
    """
    prom = prom.replace("<s>GPT4 Correct User: ", "").replace("<|end_of_turn|>GPT4 Correct Assistant:", "")

    final_resp = resp.split("GPT4 Correct User")[0]
    """
    prom = prom.replace(tokenizer.bos_token, "")

    prompt_5_shots = f"""
    your task is to solve grade school math problems, here are some examples, you should think step by step, and write the final answer clearly and follow the answer format
    question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10
    question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
    answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5
    question: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?
    answer: There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers. So in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers. Purple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers. That means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers. So in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden. #### 35
    question: Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?
    answer: To the initial 2 pounds of jelly beans, he added enough brownies to cause the weight to triple, bringing the weight to 2*3=<<2*3=6>>6 pounds. Next, he added another 2 pounds of jelly beans, bringing the weight to 6+2=<<6+2=8>>8 pounds. And finally, he added enough gummy worms to double the weight once again, to a final weight of 8*2=<<8*2=16>>16 pounds. #### 16
    next the question you should answer:
    question: {prom}
    answer: 
    """

    message = [
        {"role": "user", "content": prompt_5_shots}
    ]
    return message

def query_model(prompt, args, port):
    json = {
        **args,
        "prompt": prompt,
    }
    response = requests.post(url=script_args.url + ":" + str(port) + "/generate", json=json)
    response_json = response.json()
    return [response_json["text"][i][len(prompt) :] for i in range(len(response_json["text"]))]


default_args = {
    "use_beam_search": script_args.use_beam_search,
    "n": script_args.K,
    "temperature": script_args.temperature,
    "max_tokens": script_args.max_new_tokens,
    "seed": script_args.seed,
    "top_p": 1.0,
    "top_k": -1,
    "stop_token_ids": [tokenizer.eos_token_id] + script_args.eos_ids,
}

print(default_args)

ds = load_dataset(ds_dir, name='main', split=script_args.ds_split)
# load_dataset("json", data_files=ds_dir, split="train", field="instances")
print(ds)

# use tokenizer.apply_template to apply the template to the prompt
ds = ds.map(
    lambda x: {
        "prompt": tokenizer.apply_chat_template(change_of_format(x[script_args.dataset_key]), tokenize=False, add_generation_prompt=True),
        "answer": x["answer"]
    }
)


with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
    result = [
        executor.submit(query_model, ds[i]["prompt"], default_args, ports[i % len(ports)]) for i in range(len(ds))
    ]
    # use tqdm to show progress
    for _ in tqdm(as_completed(result), total=len(result)):
        pass

    responses = [r.result() for r in result]


gathered_data = []
length = 0
correct_cnt = 0
for i in range(len(ds)):
    gt_answer = extract_answer(ds[i]["answer"])
    res_answer = extract_answer(responses[i][0])
    
    is_correct = res_answer == gt_answer
    if res_answer == gt_answer:
        correct_cnt += 1
    tmp_data = {"prompt": ds[i]["prompt"], "responses": responses[i], "res_answer": res_answer, "gt_answer": gt_answer, "is_correct": is_correct}
    for resp in responses[i]:
        length += len(resp)
    gathered_data.append(tmp_data)
    print(tmp_data)

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = gathered_data
cnt = len(gathered_data) * K
length /= cnt
correct_rate = correct_cnt * 1.0 / cnt
print(f"I collect {cnt} samples, avg_len:{length}, correct_rate:{correct_rate}")


# with open(output_dir, "w", encoding="utf8") as f:
#     json.dump(output_eval_dataset, f, ensure_ascii=False)
