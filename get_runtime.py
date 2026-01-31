from huggingface_hub import HfApi
api = HfApi(token=open('hf_token.txt').read().strip())
print(api.get_space_runtime('ithamarspitz/adhd-noise-classifier'))
